#!/usr/bin/env python3
"""
Lane Follower Node
==================

A lightweight ROS 2 node that follows detected lane boundaries
without any global planning or CARLA dependency.

Pipeline:
  mapping publishes EstimatedLanes → this node subscribes
  → selects preferred lane → trajectory published to control

Two operating modes
-------------------
``local_only=False`` (default, CARLA bringups):
  Subscribes to ``/carla/ego_vehicle/ground_truth/odometry`` for the
  vehicle pose, transforms the lane midline base_link→map, and publishes
  the trajectory in the ``map`` frame. Pure-pursuit chases that pose.

``local_only=True`` (real-hardware bringups -- no localisation):
  Skips odometry entirely. The vehicle is assumed to sit at the origin of
  ``base_frame``, the midline is published as-is in that frame, and the
  trajectory header carries ``base_frame`` instead of ``map``. The node
  also republishes the target speed of the first trajectory point on
  ``/target_velocity`` (``std_msgs/Float32``) so ``cmd_to_can_node`` can
  forward it directly to the firmware's velocity PID without needing the
  control package.

Topics subscribed:
  /mapping/estimated_lanes               (autoiq_msgs/EstimatedLanes)
  /carla/ego_vehicle/ground_truth/odometry  (nav_msgs/Odometry, map mode only)

Topics published:
  /planning/trajectory     (autoiq_msgs/Trajectory, frame depends on mode)
  /planning/lane_midline   (nav_msgs/Path, frame depends on mode)
  /target_velocity         (std_msgs/Float32, local_only mode only)
"""

import math
import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from ament_index_python.packages import get_package_share_directory
from typing import List, Optional, Dict, Any

from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped
from autoiq_msgs.msg import Trajectory, TrajectoryPoint, EstimatedLanes

from autoiq_common.types import VehicleState
from autoiq_common.types import TrajectoryPoint as TrajectoryPointDC
from autoiq_common.converters import vehicle_state_from_odometry


class LaneFollowerNode(Node):
    def __init__(self):
        super().__init__('lane_follower_node')

        # --- Parameters ---
        # `local_only=True` strips out odometry/CARLA dependencies and
        # publishes everything in `base_frame` (no map frame, no
        # localisation). Use this on real hardware where there is no
        # global pose source.
        self.declare_parameter('local_only', False)
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('odom_topic', '/carla/ego_vehicle/ground_truth/odometry')
        self.declare_parameter('target_velocity_topic', '/target_velocity')
        self.declare_parameter(
            'estimated_lanes_topic', '/mapping/estimated_lanes')
        self.declare_parameter('trajectory_topic', '/planning/trajectory')
        self.declare_parameter('midline_topic', '/planning/lane_midline')

        self.local_only: bool = bool(
            self.get_parameter('local_only').value)
        self.base_frame: str = str(
            self.get_parameter('base_frame').value)
        odom_topic: str = str(self.get_parameter('odom_topic').value)
        target_velocity_topic: str = str(
            self.get_parameter('target_velocity_topic').value)
        estimated_lanes_topic: str = str(
            self.get_parameter('estimated_lanes_topic').value)
        trajectory_topic: str = str(
            self.get_parameter('trajectory_topic').value)
        midline_topic: str = str(self.get_parameter('midline_topic').value)

        # In local_only mode the trajectory is consumed by a controller
        # that also runs in `base_frame`, so we tag both the Trajectory
        # and the visualisation Path with that frame instead of `map`.
        self.output_frame: str = self.base_frame if self.local_only else 'map'

        # --- Road knowledge (from YAML) - for speed profile and lane selection ---
        self.road_knowledge = self._load_road_knowledge()
        self.get_logger().info(f"Road knowledge: preferred_lane={self.road_knowledge['preferred_lane']}, "
                               f"local_follow_speed={self.road_knowledge['local_follow_speed']} km/h")

        # --- State ---
        self.vehicle_state = VehicleState()
        self._estimated_lanes: Optional[EstimatedLanes] = None
        self._estimated_lanes_stamp = None
        self._estimated_lanes_timeout: float = 1.0  # seconds

        # Temporal fallback for robustness
        self._last_midline_bl: Optional[np.ndarray] = None

        # --- Subscribers ---
        # Skip the odometry subscription entirely in local_only mode --
        # we don't have a global pose source on the real vehicle and the
        # default vehicle_state of (0, 0, 0, speed=0) is exactly what
        # _base_link_to_map needs to behave as identity.
        if not self.local_only:
            self.create_subscription(
                Odometry,
                odom_topic,
                self._odom_cb, 1)

        drivable_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(
            EstimatedLanes, estimated_lanes_topic,
            self._estimated_lanes_cb, drivable_qos)

        # --- Publishers ---
        self.traj_pub = self.create_publisher(Trajectory, trajectory_topic, 1)
        self.midline_pub = self.create_publisher(Path, midline_topic, 1)

        # In local_only mode, also push the target speed straight to the
        # firmware's velocity PID via cmd_to_can_node so we don't have to
        # compose a full VehicleControl pipeline just to set a setpoint.
        self.target_velocity_pub = (
            self.create_publisher(Float32, target_velocity_topic, 1)
            if self.local_only else None
        )

        # --- Timer (10 Hz) ---
        self.create_timer(0.1, self._loop)
        self.get_logger().info(
            f'Lane Follower Node ready (local_only={self.local_only}, '
            f'output_frame={self.output_frame})')

    def _load_road_knowledge(self) -> Dict[str, Any]:
        """Load road knowledge config from YAML file."""
        config_path = f"{get_package_share_directory('planning')}/config/road_knowledge.yaml"

        defaults = {
            'preferred_lane': 1,
            'local_follow_speed': 10.0,
            'curvature_speed_factor': 2.0,
        }

        try:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f) or {}
            return {**defaults, **data}
        except FileNotFoundError:
            self.get_logger().warn(f'road_knowledge.yaml not found at {config_path}. Using defaults.')
            return defaults
        except Exception as e:
            self.get_logger().warn(f'Failed to load road_knowledge.yaml: {e}. Using defaults.')
            return defaults

    # ==================== Callbacks ============================

    def _odom_cb(self, msg: Odometry):
        self.vehicle_state = vehicle_state_from_odometry(msg)

    def _estimated_lanes_cb(self, msg: EstimatedLanes):
        self.get_logger().info(
            f'EstimatedLanes received: valid={msg.valid}, lanes={len(msg.lanes)}',
            throttle_duration_sec=2.0)
        self._estimated_lanes = msg
        self._estimated_lanes_stamp = self.get_clock().now()

    # ==================== Main Loop ============================

    def _loop(self):
        if not self._estimated_lanes_valid():
            self.get_logger().warn('No valid estimated lanes', throttle_duration_sec=2.0)
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

        # In local_only mode the controller also runs in base_frame, so
        # we skip the rigid-body transform entirely. Otherwise the
        # midline gets pushed into the global map frame using the latest
        # odometry estimate.
        if self.local_only:
            midline_out = midline_bl
        else:
            midline_out = self._base_link_to_map(midline_bl)

        # Publish visualisation
        self._publish_midline(midline_out)

        # Build & publish trajectory
        trajectory = self._trajectory_from_midline(midline_out)
        if trajectory:
            self._publish_trajectory(trajectory)
            # Push the very first segment's target speed onto
            # /target_velocity so the firmware's velocity PID has a fresh
            # setpoint even if the host-side controller decides to skip
            # this tick (e.g. lookahead failure).
            if self.target_velocity_pub is not None:
                msg = Float32()
                msg.data = float(trajectory[0].speed)
                self.target_velocity_pub.publish(msg)

    def _estimated_lanes_valid(self) -> bool:
        if self._estimated_lanes is None or self._estimated_lanes_stamp is None:
            return False
        age = (self.get_clock().now() - self._estimated_lanes_stamp).nanoseconds * 1e-9
        if age > self._estimated_lanes_timeout:
            return False
        return self._estimated_lanes.valid

    def _select_lane_midline(self) -> Optional[np.ndarray]:
        """
        Select the preferred lane's centerline from the estimated lanes.

        Returns (N, 2) numpy array of (x, y) in base_link frame, or None.
        """
        if not self._estimated_lanes.lanes:
            return None

        # Get preferred lane index from road knowledge
        preferred_idx = self.road_knowledge['preferred_lane']

        # Clamp to available lanes
        if preferred_idx >= len(self._estimated_lanes.lanes):
            preferred_idx = len(self._estimated_lanes.lanes) - 1

        lane = self._estimated_lanes.lanes[preferred_idx]

        # If preferred lane invalid, find first valid lane
        if not lane.is_valid:
            for l in self._estimated_lanes.lanes:
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

        target_v = self.road_knowledge['local_follow_speed'] / 3.6
        curv_factor = self.road_knowledge['curvature_speed_factor']

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
        msg.header.frame_id = self.output_frame
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
        msg.header.frame_id = self.output_frame
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
