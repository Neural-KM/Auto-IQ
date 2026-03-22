#!/usr/bin/env python3
"""
Lane Follower Node
==================

A lightweight ROS 2 node that follows detected lane boundaries
without any global planning or CARLA dependency.

Pipeline:
  mapping publishes LocalLaneData → this node subscribes
  → selects preferred lane → trajectory published to control

Topics subscribed:
  /mapping/local_lane_data  (autoiq_msgs/LocalLaneData, base_link)
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
from autoiq_msgs.msg import Trajectory, TrajectoryPoint, LocalLaneData

from autoiq_common.types import VehicleState
from autoiq_common.types import TrajectoryPoint as TrajectoryPointDC
from autoiq_common.converters import vehicle_state_from_odometry

from planning.core.road_knowledge import RoadKnowledge


class LaneFollowerNode(Node):
    def __init__(self):
        super().__init__('lane_follower_node')

        # --- Road knowledge (from YAML) - for speed profile and lane selection ---
        self.road_knowledge = RoadKnowledge.from_yaml(RoadKnowledge.default_path())
        self.get_logger().info(f'Road knowledge: {self.road_knowledge}')

        # --- State ---
        self.vehicle_state = VehicleState()
        self._local_lane_data: Optional[LocalLaneData] = None
        self._local_lane_data_stamp = None
        self._local_lane_data_timeout: float = 1.0  # seconds

        # Temporal fallback for robustness
        self._last_midline_bl: Optional[np.ndarray] = None

        # --- Subscribers ---
        self.create_subscription(
            Odometry,
            '/carla/ego_vehicle/ground_truth/odometry',
            self._odom_cb, 1)

        drivable_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(
            LocalLaneData, '/mapping/local_lane_data',
            self._local_lane_data_cb, drivable_qos)

        # --- Publishers ---
        self.traj_pub = self.create_publisher(Trajectory, '/planning/trajectory', 1)
        self.midline_pub = self.create_publisher(Path, '/planning/lane_midline', 1)

        # --- Timer (10 Hz) ---
        self.create_timer(0.1, self._loop)
        self.get_logger().info('Lane Follower Node ready (using LocalLaneData message)')

    # ==================== Callbacks ============================

    def _odom_cb(self, msg: Odometry):
        self.vehicle_state = vehicle_state_from_odometry(msg)

    def _local_lane_data_cb(self, msg: LocalLaneData):
        self.get_logger().info(
            f'LocalLaneData received: valid={msg.valid}, lanes={len(msg.lanes)}',
            throttle_duration_sec=2.0)
        self._local_lane_data = msg
        self._local_lane_data_stamp = self.get_clock().now()

    # ==================== Main Loop ============================

    def _loop(self):
        if not self._local_lane_data_valid():
            self.get_logger().warn('No valid local lane data', throttle_duration_sec=2.0)
            return

        # Get midline from selected lane
        midline_bl = self._select_lane_midline()

        if midline_bl is None or len(midline_bl) < 3:
            # Fallback to previous midline
            if self._last_midline_bl is not None:
                midline_bl = self._last_midline_bl
                self.get_logger().warn('Using previous midline (fallback)', throttle_duration_sec=2.0)
            else:
                self.get_logger().error('No valid midline available')
                return
        else:
            # Store for fallback
            self._last_midline_bl = midline_bl
            self.get_logger().info(
                f'Midline selected ({len(midline_bl)} points)',
                throttle_duration_sec=2.0)

        # Transform base_link → map
        midline_map = self._base_link_to_map(midline_bl)

        # Publish visualisation
        self._publish_midline(midline_map)

        # Build & publish trajectory
        trajectory = self._trajectory_from_midline(midline_map)
        if trajectory:
            self._publish_trajectory(trajectory)

    def _local_lane_data_valid(self) -> bool:
        if self._local_lane_data is None or self._local_lane_data_stamp is None:
            return False
        age = (self.get_clock().now() - self._local_lane_data_stamp).nanoseconds * 1e-9
        if age > self._local_lane_data_timeout:
            return False
        return self._local_lane_data.valid

    def _select_lane_midline(self) -> Optional[np.ndarray]:
        """
        Select the preferred lane's centerline from the local lane data.

        Returns (N, 2) numpy array of (x, y) in base_link frame, or None.
        """
        if not self._local_lane_data.lanes:
            return None

        # Get preferred lane index from road knowledge
        preferred_idx = self.road_knowledge.preferred_lane

        # Clamp to available lanes
        if preferred_idx >= len(self._local_lane_data.lanes):
            preferred_idx = len(self._local_lane_data.lanes) - 1

        lane = self._local_lane_data.lanes[preferred_idx]

        # If preferred lane invalid, find first valid lane
        if not lane.is_valid:
            for l in self._local_lane_data.lanes:
                if l.is_valid:
                    lane = l
                    self.get_logger().warn(
                        f'Preferred lane {preferred_idx} invalid, using lane {l.lane_index}',
                        throttle_duration_sec=2.0)
                    break

        if not lane.centerline:
            return None

        # Convert geometry_msgs/Point[] to numpy array
        centerline = np.array([[p.x, p.y] for p in lane.centerline])
        return centerline

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
