#!/usr/bin/env python3
"""
CARLA Visualization Node
========================

ROS2 node for drawing debug shapes in CARLA simulator.

Subscribes to:
- /carla/ego_vehicle/ground_truth/odometry (nav_msgs/Odometry)
- /planning/trajectory (autoiq_msgs/Trajectory)
- /planning/global_route (nav_msgs/Path)
- /control/predicted_trajectory (nav_msgs/Path)
- /perception/obstacles (autoiq_msgs/ObstacleArray)
- /perception/ground_truth/obstacles (autoiq_msgs/ObstacleArray)
- /carla/ego_vehicle/control (autoiq_msgs/VehicleControl)

Also publishes RViz markers for obstacles and trajectory.
"""

import json
import math
import os
from dataclasses import replace
from typing import List, Tuple

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path, Odometry
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import PoseStamped, PointStamped

from autoiq_msgs.msg import ObstacleArray as ObstacleArrayMsg, Trajectory as TrajectoryMsg
from autoiq_msgs.msg import VehicleControl

from autoiq_common.types import VehicleState, Trajectory, ObstacleArray, Obstacle
from autoiq_common.converters import (
    vehicle_state_from_odometry,
    trajectory_from_ros,
    obstacle_array_from_ros,
)
from autoiq_common.utils import quaternion_to_yaw

import carla
from ament_index_python.packages import get_package_share_directory

from visualization.carla_drawer import CarlaDrawer


class CarlaVizNode(Node):
    """
    ROS2 Node for CARLA debug drawing and RViz marker publishing.
    
    Draws trajectories, routes, obstacles, and other debug shapes
    in CARLA simulator. Also publishes MarkerArrays for RViz.
    """
    
    def __init__(self):
        super().__init__('carla_viz_node')
        
        # ==================== Parameters ====================
        self.declare_parameter('carla_host', 'localhost')
        self.declare_parameter('carla_port', 2000)
        self.declare_parameter('carla_timeout', 10.0)
        self.declare_parameter('viz_rate', 20.0)  # Hz
        self.declare_parameter('draw_lifetime', 0.07)  # seconds
        self.declare_parameter('objects_file', '')
        
        # Comma-separated list of objects to draw in CARLA debug view
        # Options: trajectory, global_route, predicted_traj, ground_truth_obstacles, detected_obstacles, speed, sensors
        self.declare_parameter('carla_debug_draw', 'trajectory,global_route,predicted_traj,speed')
        
        carla_host = self.get_parameter('carla_host').value
        carla_port = self.get_parameter('carla_port').value
        carla_timeout = self.get_parameter('carla_timeout').value
        viz_rate = self.get_parameter('viz_rate').value
        self.draw_lifetime = self.get_parameter('draw_lifetime').value
        objects_file_param = self.get_parameter('objects_file').value
        
        # Parse carla_debug_draw string into a set of items
        carla_debug_draw_str = self.get_parameter('carla_debug_draw').value
        self.carla_draw_items = set()
        if carla_debug_draw_str:
            self.carla_draw_items = set(item.strip() for item in carla_debug_draw_str.split(',') if item.strip())
        
        # ==================== CARLA Connection ====================
        self.carla_drawer = None
        try:
            self.get_logger().info(f'Connecting to CARLA at {carla_host}:{carla_port}...')
            client = carla.Client(carla_host, carla_port)
            client.set_timeout(carla_timeout)
            world = client.get_world()
            self.carla_drawer = CarlaDrawer(world)
            self.get_logger().info('CARLA visualization enabled')
        except Exception as e:
            self.get_logger().error(f'Failed to connect to CARLA: {e}')
            self.get_logger().error('CARLA visualization disabled - node will only publish RViz markers')
        
        # ==================== Data Storage ====================
        self.vehicle_state: VehicleState = None
        self.target_speed = 0.0
        self.current_steer = 0.0
        self.current_throttle = 0.0
        self.current_brake = 0.0
        
        self.trajectory = []  # List of (x, y)
        self.global_route = []  # List of (x, y, z)
        self.predicted_trajectory = []  # List of (x, y, yaw, speed)
        self.current_waypoint = None  # Tuple of (x, y, z) or None
        
        self.ground_truth_obstacles: List[Obstacle] = []
        self.detected_obstacles: List[Obstacle] = []
        
        # Load sensor mounts from objects.json
        self.sensor_mounts = self._load_sensor_mounts(objects_file_param)
        
        # ==================== Publishers for RViz ====================
        self.gt_obstacles_marker_pub = self.create_publisher(
            MarkerArray, '/visualization/ground_truth_obstacles', 10)
        self.detected_obstacles_marker_pub = self.create_publisher(
            MarkerArray, '/visualization/detected_obstacles', 10)
        self.trajectory_viz_pub = self.create_publisher(
            Path, '/visualization/trajectory', 10)
        self.current_waypoint_marker_pub = self.create_publisher(
            Marker, '/visualization/current_waypoint', 10)
        
        # ==================== Subscribers ====================
        self.odom_sub = self.create_subscription(
            Odometry, '/carla/ego_vehicle/ground_truth/odometry',
            self.odom_callback, 10)
        
        self.trajectory_sub = self.create_subscription(
            TrajectoryMsg, '/planning/trajectory',
            self.trajectory_callback, 10)
        
        self.global_route_sub = self.create_subscription(
            Path, '/planning/global_route',
            self.global_route_callback, 10)
        
        self.predicted_sub = self.create_subscription(
            Path, '/control/predicted_trajectory',
            self.predicted_callback, 10)
        
        self.gt_obstacles_sub = self.create_subscription(
            ObstacleArrayMsg, '/perception/ground_truth/obstacles',
            self.ground_truth_obstacles_callback, 10)
        
        self.detected_obstacles_sub = self.create_subscription(
            ObstacleArrayMsg, '/perception/obstacles',
            self.detected_obstacles_callback, 10)
        
        self.current_waypoint_sub = self.create_subscription(
            PointStamped, '/planning/current_waypoint',
            self.current_waypoint_callback, 10)
        
        self.control_sub = self.create_subscription(
            VehicleControl, '/carla/ego_vehicle/control',
            self.control_callback, 10)
        
        # ==================== Visualization Timer ====================
        viz_period = 1.0 / viz_rate
        self.viz_timer = self.create_timer(viz_period, self.visualization_loop)
        
        self.get_logger().info(f'CARLA Viz Node initialized ({viz_rate}Hz)')
    
    # ==================== Sensor Mount Utilities ====================
    
    def _load_sensor_mounts(self, objects_file_param: str):
        """Load sensor spawn offsets from objects.json."""
        objects_path = objects_file_param
        if not objects_path:
            try:
                pkg_share = get_package_share_directory('autoiq_bridge')
                objects_path = os.path.join(pkg_share, 'config', 'objects.json')
            except Exception as e:
                self.get_logger().warn(f'Could not resolve autoiq_bridge share for objects.json: {e}')
                return []

        if not os.path.exists(objects_path):
            self.get_logger().warn(f'objects.json not found at {objects_path}')
            return []

        try:
            with open(objects_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            self.get_logger().warn(f'Failed to load objects.json: {e}')
            return []

        mounts = []
        for obj in data.get('objects', []):
            if obj.get('id') != 'ego_vehicle':
                continue
            for sensor in obj.get('sensors', []):
                sp = sensor.get('spawn_point', {})
                mounts.append({
                    'id': sensor.get('id', 'sensor'),
                    'type': sensor.get('type', ''),
                    'x': float(sp.get('x', 0.0)),
                    'y': float(sp.get('y', 0.0)),
                    'z': float(sp.get('z', 0.0)),
                    'yaw': float(sp.get('yaw', 0.0)),
                })

        if mounts:
            self.get_logger().info(f'Loaded {len(mounts)} sensor mounts from objects.json')

        return mounts
    
    # ==================== Callbacks ====================
    
    def odom_callback(self, msg: Odometry):
        """Update vehicle state from odometry."""
        self.vehicle_state = vehicle_state_from_odometry(msg)
    
    def trajectory_callback(self, msg: TrajectoryMsg):
        """Receive local trajectory from planning node."""
        traj = trajectory_from_ros(msg)
        waypoints = []
        
        if traj.points:
            self.target_speed = traj.points[0].speed
            
        for point in traj.points:
            waypoints.append((point.x, point.y))
        self.trajectory = waypoints
        
        # Publish as Path for RViz
        self._publish_trajectory_path(traj)
    
    def global_route_callback(self, msg: Path):
        """Receive global route from planning node."""
        waypoints = []
        for pose in msg.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            z = pose.pose.position.z
            waypoints.append((x, y, z))
        self.global_route = waypoints
    
    def current_waypoint_callback(self, msg: PointStamped):
        """Receive current waypoint from planning node."""
        self.current_waypoint = (msg.point.x, msg.point.y, msg.point.z)
        self._publish_current_waypoint_marker()
    
    def predicted_callback(self, msg: Path):
        """Receive MPC predicted trajectory from control node."""
        states = []
        for pose in msg.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            quat = pose.pose.orientation
            yaw = quaternion_to_yaw(quat)
            states.append((x, y, yaw, 0.0))
        self.predicted_trajectory = states
    
    def control_callback(self, msg: VehicleControl):
        """Receive vehicle control command."""
        self.current_steer = msg.steer
        self.current_throttle = msg.throttle
        self.current_brake = msg.brake
    
    def ground_truth_obstacles_callback(self, msg: ObstacleArrayMsg):
        """Receive ground truth obstacles."""
        obstacle_array = obstacle_array_from_ros(msg)
        self.ground_truth_obstacles = self._transform_obstacles_to_world(obstacle_array)
        
        marker_array = self._obstacles_to_marker_array(
            self.ground_truth_obstacles, 'ground_truth_obstacles', (0.0, 1.0, 0.0))
        self.gt_obstacles_marker_pub.publish(marker_array)
    
    def detected_obstacles_callback(self, msg: ObstacleArrayMsg):
        """Receive detected obstacles."""
        obstacle_array = obstacle_array_from_ros(msg)
        self.detected_obstacles = self._transform_obstacles_to_world(obstacle_array)
        
        marker_array = self._obstacles_to_marker_array(
            self.detected_obstacles, 'detected_obstacles', (1.0, 1.0, 0.0))
        self.detected_obstacles_marker_pub.publish(marker_array)
    
    def _transform_obstacles_to_world(self, obstacle_array: ObstacleArray) -> List[Obstacle]:
        """Transform obstacles from ego frame to world frame."""
        if self.vehicle_state is None:
            return []
        
        result = []
        vx, vy, vyaw = self.vehicle_state.x, self.vehicle_state.y, self.vehicle_state.yaw
        c, s = math.cos(vyaw), math.sin(vyaw)
        
        for obs in obstacle_array.obstacles:
            world_x = vx + obs.x * c - obs.y * s
            world_y = vy + obs.x * s + obs.y * c
            world_yaw = vyaw + obs.yaw
            world_obs = replace(obs, x=world_x, y=world_y, yaw=world_yaw)
            result.append(world_obs)
        
        return result
    
    def _obstacles_to_marker_array(self, obstacles: List[Obstacle], namespace: str,
                                    color: Tuple[float, float, float]) -> MarkerArray:
        """Convert obstacle list to MarkerArray for RViz."""
        marker_array = MarkerArray()
        
        clear_marker = Marker()
        clear_marker.header.frame_id = 'map'
        clear_marker.header.stamp = self.get_clock().now().to_msg()
        clear_marker.ns = namespace
        clear_marker.id = 0
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        
        for i, obs in enumerate(obstacles):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = namespace
            marker.id = i + 1
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            marker.pose.position.x = obs.x
            marker.pose.position.y = obs.y
            marker.pose.position.z = obs.height / 2
            
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = math.sin(obs.yaw / 2)
            marker.pose.orientation.w = math.cos(obs.yaw / 2)
            
            marker.scale.x = obs.length
            marker.scale.y = obs.width
            marker.scale.z = obs.height
            
            marker.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=0.7)
            marker.lifetime.nanosec = 200000000
            
            marker_array.markers.append(marker)
            
            # Text label
            text_marker = Marker()
            text_marker.header = marker.header
            text_marker.ns = f'{namespace}_labels'
            text_marker.id = i + 1
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = obs.x
            text_marker.pose.position.y = obs.y
            text_marker.pose.position.z = obs.height + 0.5
            text_marker.text = f"{obs.type}"
            text_marker.scale.z = 0.5
            text_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.9)
            text_marker.lifetime.nanosec = 200000000
            marker_array.markers.append(text_marker)
        
        return marker_array
    
    def _publish_trajectory_path(self, traj: Trajectory):
        """Convert Trajectory dataclass to Path for RViz."""
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        for point in traj.points:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = point.x
            pose.pose.position.y = point.y
            pose.pose.position.z = 0.0
            pose.pose.orientation.z = math.sin(point.yaw / 2)
            pose.pose.orientation.w = math.cos(point.yaw / 2)
            path_msg.poses.append(pose)
        
        self.trajectory_viz_pub.publish(path_msg)
    
    def _publish_current_waypoint_marker(self):
        """Publish current waypoint as a sphere marker for RViz."""
        if self.current_waypoint is None:
            return
        
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'current_waypoint'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = self.current_waypoint[0]
        marker.pose.position.y = self.current_waypoint[1]
        marker.pose.position.z = self.current_waypoint[2] + 1.0
        
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 2.0
        marker.scale.y = 2.0
        marker.scale.z = 2.0
        
        marker.color = ColorRGBA(r=1.0, g=0.0, b=1.0, a=0.9)
        marker.lifetime.nanosec = 200000000
        
        self.current_waypoint_marker_pub.publish(marker)
    
    # ==================== Visualization Loop ====================
    
    def visualization_loop(self):
        """Draw debug shapes in CARLA."""
        if self.vehicle_state is None or self.carla_drawer is None:
            return
        
        lt = self.draw_lifetime
        items = self.carla_draw_items
        
        # Draw global route
        if 'global_route' in items and self.global_route:
            self.carla_drawer.draw_global_route(self.global_route, lifetime=lt)
        
        # Draw local trajectory
        if 'trajectory' in items and self.trajectory:
            self.carla_drawer.draw_trajectory(self.trajectory, lifetime=lt)
        
        # Draw MPC predicted trajectory
        if 'predicted_traj' in items and self.predicted_trajectory:
            self.carla_drawer.draw_predicted_trajectory(self.predicted_trajectory, lifetime=lt)
        
        # Draw speed indicator
        if 'speed' in items:
            self.carla_drawer.draw_speed_indicator(
                self.vehicle_state.x, self.vehicle_state.y,
                self.vehicle_state.speed * 3.6,
                self.target_speed * 3.6,
                lifetime=lt
            )
        
        # Draw sensor mounts
        if 'sensors' in items and self.sensor_mounts:
            self.carla_drawer.draw_sensor_mounts(
                self.vehicle_state.x, self.vehicle_state.y,
                self.vehicle_state.yaw,
                self.sensor_mounts,
                lifetime=lt
            )
        
        # Ground truth obstacles (green boxes)
        if 'ground_truth_obstacles' in items:
            for obs in self.ground_truth_obstacles:
                self.carla_drawer.draw_obstacle_box(
                    obs.x, obs.y, obs.yaw,
                    obs.length, obs.width, obs.height,
                    is_vehicle=(obs.type == 'vehicle'),
                    lifetime=lt,
                    color=(0, 255, 0)
                )
        
        # Detected obstacles (yellow boxes)
        if 'detected_obstacles' in items:
            for obs in self.detected_obstacles:
                self.carla_drawer.draw_obstacle_box(
                    obs.x, obs.y, obs.yaw,
                    obs.length, obs.width, obs.height,
                    is_vehicle=(obs.type == 'vehicle'),
                    lifetime=lt,
                    color=(255, 255, 0)
                )


def main(args=None):
    rclpy.init(args=args)
    node = CarlaVizNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
