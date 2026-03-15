#!/usr/bin/env python3
"""
Lane Follower Node
==================

A lightweight ROS 2 node that follows detected lane boundaries
without any global planning or CARLA dependency.

Pipeline:
  mapping publishes boundaries → this node subscribes
  → LaneMidlineGenerator computes midline → trajectory published to control

Topics subscribed:
  /mapping/boundary/left   (nav_msgs/Path, base_link)
  /mapping/boundary/right  (nav_msgs/Path, base_link)
  /carla/ego_vehicle/ground_truth/odometry  (nav_msgs/Odometry)

Topics published:
  /planning/trajectory     (autoiq_msgs/Trajectory, map frame)
  /planning/lane_midline   (nav_msgs/Path, map frame — visualisation)
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from typing import List, Optional

from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped
from autoiq_msgs.msg import Trajectory, TrajectoryPoint

from autoiq_common.types import VehicleState
from autoiq_common.types import TrajectoryPoint as TrajectoryPointDC
from autoiq_common.converters import vehicle_state_from_odometry

from planning.core.road_knowledge import RoadKnowledge
from planning.core.lane_midline_generator import LaneMidlineGenerator


class LaneFollowerNode(Node):
    def __init__(self):
        super().__init__('lane_follower_node')

        # --- Road knowledge (from YAML) ---
        self.road_knowledge = RoadKnowledge.from_yaml(RoadKnowledge.default_path())
        self.midline_generator = LaneMidlineGenerator(self.road_knowledge)
        self.get_logger().info(f'Road knowledge: {self.road_knowledge}')

        # --- State ---
        self.vehicle_state = VehicleState()
        self._left_boundary: Optional[np.ndarray] = None
        self._right_boundary: Optional[np.ndarray] = None
        self._boundary_stamp = None
        self._boundary_timeout: float = 1.0  # seconds

        # --- Subscribers ---
        self.create_subscription(
            Odometry,
            '/carla/ego_vehicle/ground_truth/odometry',
            self._odom_cb, 1)

        boundary_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(
            Path, '/mapping/boundary/left',
            self._left_boundary_cb, boundary_qos)
        self.create_subscription(
            Path, '/mapping/boundary/right',
            self._right_boundary_cb, boundary_qos)

        # --- Publishers ---
        self.traj_pub = self.create_publisher(Trajectory, '/planning/trajectory', 1)
        self.midline_pub = self.create_publisher(Path, '/planning/lane_midline', 1)

        # --- Timer (10 Hz) ---
        self.create_timer(0.1, self._loop)
        self.get_logger().info('Lane Follower Node ready')

    # ==================== Callbacks ============================

    def _odom_cb(self, msg: Odometry):
        self.vehicle_state = vehicle_state_from_odometry(msg)

    def _left_boundary_cb(self, msg: Path):
        self.get_logger().info('Left boundary received')
        self._left_boundary = self._path_to_array(msg)
        self._boundary_stamp = self.get_clock().now()

    def _right_boundary_cb(self, msg: Path):
        self.get_logger().info('Right boundary received')
        self._right_boundary = self._path_to_array(msg)
        self._boundary_stamp = self.get_clock().now()

    @staticmethod
    def _path_to_array(msg: Path) -> Optional[np.ndarray]:
        if len(msg.poses) < 3:
            return None
        return np.array([[p.pose.position.x, p.pose.position.y]
                         for p in msg.poses])

    # ==================== Main Loop ============================

    def _loop(self):
        if not self._boundaries_valid():
            self.get_logger().warn('No valid boundaries', throttle_duration_sec=2.0)
            return
        self.get_logger().info('Boundaries valid, generating midline', throttle_duration_sec=2.0)
        # Generate midline in base_link
        midline_bl, status_code = self.midline_generator.generate(
            self._left_boundary, self._right_boundary)
        
        status_msg = self.midline_generator.STATUS_MSGS.get(status_code, "Unknown status")
        
        if status_code == self.midline_generator.SUCCESS:
            self.get_logger().info(f'{status_msg} ({len(midline_bl)} points)', throttle_duration_sec=2.0)
        elif status_code > 0:
            self.get_logger().warn(f'{status_msg} ({len(midline_bl)} points)', throttle_duration_sec=2.0)
        else:
            self.get_logger().error(status_msg)
            return

        # Transform base_link → map
        midline_map = self._base_link_to_map(midline_bl)

        # Publish visualisation
        self._publish_midline(midline_map)

        # Build & publish trajectory
        trajectory = self._trajectory_from_midline(midline_map)
        if trajectory:
            self._publish_trajectory(trajectory)

    def _boundaries_valid(self) -> bool:
        if self._boundary_stamp is None:
            return False
        age = (self.get_clock().now() - self._boundary_stamp).nanoseconds * 1e-9
        if age > self._boundary_timeout:
            return False
        return (self._left_boundary is not None or
                self._right_boundary is not None)

    # ==================== Geometry =============================

    def _base_link_to_map(self, pts_bl: np.ndarray) -> np.ndarray:
        """Rigid-body transform (N,2) base_link → map."""
        c = math.cos(self.vehicle_state.yaw)
        s = math.sin(self.vehicle_state.yaw)
        x_bl, y_bl = pts_bl[:, 0], pts_bl[:, 1]
        mx = self.vehicle_state.x + x_bl * c - y_bl * s
        my = self.vehicle_state.y + x_bl * s + y_bl * c
        return np.column_stack([mx, my])

    def _trajectory_from_midline(
        self, midline_map: np.ndarray
    ) -> List[TrajectoryPointDC]:
        """Convert (N,2) map-frame midline → trajectory with speed profile."""
        n = len(midline_map)
        if n < 2:
            return []

        target_v = self.road_knowledge.local_follow_speed / 3.6
        curv_factor = self.road_knowledge.curvature_speed_factor

        dx = np.diff(midline_map[:, 0])
        dy = np.diff(midline_map[:, 1])
        yaw = np.arctan2(dy, dx)
        yaw = np.append(yaw, yaw[-1])

        # Curvature via finite differences
        curvature = np.zeros(n)
        for i in range(1, n - 1):
            x0, y0 = midline_map[i - 1]
            x1, y1 = midline_map[i]
            x2, y2 = midline_map[i + 1]
            d1x, d1y = x1 - x0, y1 - y0
            d2x, d2y = x2 - x1, y2 - y1
            cross = d1x * d2y - d1y * d2x
            l1 = math.hypot(d1x, d1y)
            l2 = math.hypot(d2x, d2y)
            if l1 > 0 and l2 > 0:
                curvature[i] = 2.0 * cross / (l1 * l2 * (l1 + l2))

        # Speed: reduce in curves
        speed = np.full(n, target_v)
        for i in range(n):
            k = abs(curvature[i])
            if k > 1e-3:
                speed[i] = min(speed[i], math.sqrt(curv_factor / k))
        speed[0] = max(self.vehicle_state.speed, 0.5)

        # Cumulative time
        dists = np.hypot(dx, dy)
        t = np.zeros(n)
        for i in range(1, n):
            avg_v = max(0.5 * (speed[i - 1] + speed[i]), 0.1)
            t[i] = t[i - 1] + dists[i - 1] / avg_v

        return [
            TrajectoryPointDC(
                x=float(midline_map[i, 0]),
                y=float(midline_map[i, 1]),
                yaw=float(yaw[i]),
                speed=float(speed[i]),
                curvature=float(curvature[i]),
                time_from_start=float(t[i]),
            )
            for i in range(n)
        ]

    # ==================== Publishers ===========================

    def _publish_trajectory(self, trajectory: List[TrajectoryPointDC]):
        msg = Trajectory()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.behavior_state = Trajectory.LANE_FOLLOWING
        msg.target_lane = 0
        for pt in trajectory:
            tp = TrajectoryPoint()
            tp.x = pt.x
            tp.y = pt.y
            tp.yaw = pt.yaw
            tp.speed = pt.speed
            tp.curvature = pt.curvature
            tp.time_from_start = pt.time_from_start
            msg.points.append(tp)
        self.traj_pub.publish(msg)

    def _publish_midline(self, midline_map: np.ndarray):
        msg = Path()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y in midline_map:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.midline_pub.publish(msg)


def main():
    rclpy.init()
    node = LaneFollowerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
