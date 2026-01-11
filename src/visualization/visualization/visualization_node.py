#!/usr/bin/env python3
"""
Visualization Node
==================

ROS2 node for visualizing Auto-IQ data in both CARLA (3D debug drawings)
and RViz (MarkerArrays).

Subscribes to:
- /carla/ego_vehicle/ground_truth/odometry (nav_msgs/Odometry) - Vehicle state
- /planning/trajectory (autoiq_msgs/Trajectory) - Local trajectory
- /planning/global_route (nav_msgs/Path) - Global route
- /control/predicted_trajectory (nav_msgs/Path) - MPC predicted path
- /perception/obstacles (autoiq_msgs/ObstacleArray) - Detected obstacles
- /perception/ground_truth/obstacles (autoiq_msgs/ObstacleArray) - Ground truth obstacles

Publishes:
- /visualization/trajectory (nav_msgs/Path) - Trajectory for RViz
- /visualization/ground_truth_obstacles (MarkerArray) - Ground truth obstacle markers
- /visualization/detected_obstacles (MarkerArray) - Detected obstacle markers

Connects to CARLA for debug drawing (optional).
"""

import math
import numpy as np
from typing import List, Tuple

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path, Odometry
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point, PoseStamped

from autoiq_msgs.msg import ObstacleArray, Trajectory

import carla

from visualization.carla_drawer import CarlaDrawer
from visualization.live_2d_visualizer import Live2DVisualizerROS


def quaternion_to_yaw(q) -> float:
    """Convert quaternion to yaw angle in radians."""
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class VisualizationNode(Node):
    """
    ROS2 Visualization Node.
    
    Subscribes to planning, control, and perception topics and visualizes
    the data in CARLA (3D) and matplotlib (2D).
    """
    
    def __init__(self):
        super().__init__('visualization_node')
        
        # ==================== Parameters ====================
        self.declare_parameter('carla_host', 'localhost')
        self.declare_parameter('carla_port', 2000)
        self.declare_parameter('carla_timeout', 10.0)
        self.declare_parameter('enable_2d_viz', False)
        self.declare_parameter('viz_rate', 10.0)  # Hz
        self.declare_parameter('draw_lifetime', 0.15)  # seconds
        
        # Comma-separated list of objects to draw in CARLA debug view
        # Options: trajectory, global_route, predicted, ground_truth_obstacles, detected_obstacles, speed
        self.declare_parameter('carla_debug_draw', 'trajectory,global_route,predicted,speed')
        
        carla_host = self.get_parameter('carla_host').value
        carla_port = self.get_parameter('carla_port').value
        carla_timeout = self.get_parameter('carla_timeout').value
        self.enable_2d_viz = self.get_parameter('enable_2d_viz').value
        viz_rate = self.get_parameter('viz_rate').value
        self.draw_lifetime = self.get_parameter('draw_lifetime').value
        
        # Parse carla_debug_draw string into a set of items
        carla_debug_draw_str = self.get_parameter('carla_debug_draw').value
        self.carla_draw_items = set()
        if carla_debug_draw_str:
            self.carla_draw_items = set(item.strip() for item in carla_debug_draw_str.split(',') if item.strip())
        self.enable_carla_viz = len(self.carla_draw_items) > 0
        
        # ==================== CARLA Connection ====================
        self.carla_drawer = None
        if self.enable_carla_viz:
            try:
                self.get_logger().info(f'Connecting to CARLA at {carla_host}:{carla_port}...')
                client = carla.Client(carla_host, carla_port)
                client.set_timeout(carla_timeout)
                world = client.get_world()
                self.carla_drawer = CarlaDrawer(world)
                self.get_logger().info('CARLA visualization enabled')
            except Exception as e:
                self.get_logger().warn(f'Failed to connect to CARLA: {e}')
                self.get_logger().warn('CARLA visualization disabled')
                self.enable_carla_viz = False
        
        # ==================== 2D Visualizer ====================
        self.viz_2d = None
        if self.enable_2d_viz:
            self.viz_2d = Live2DVisualizerROS(title="Auto-IQ Live 2D", refresh_hz=viz_rate)
            self.viz_2d.start()
            self.get_logger().info('2D matplotlib visualization enabled')
        
        # ==================== Data Storage ====================
        # Vehicle state (ROS coordinates - right-handed)
        self.vehicle_x = None
        self.vehicle_y = None
        self.vehicle_yaw = None
        self.vehicle_speed = 0.0
        self.target_speed = 0.0
        
        # Paths (ROS coordinates)
        self.trajectory = []  # List of (x, y)
        self.global_route = []  # List of (x, y, z)
        self.predicted = []  # List of (x, y, yaw, speed)
        
        # Obstacles (ROS world coordinates)
        self.ground_truth_obstacles = []  # List of dicts
        self.detected_obstacles = []  # List of dicts
        
        # ==================== Publishers for RViz ====================
        # Publish MarkerArrays for obstacles visualization
        self.gt_obstacles_marker_pub = self.create_publisher(
            MarkerArray, '/visualization/ground_truth_obstacles', 10)
        self.detected_obstacles_marker_pub = self.create_publisher(
            MarkerArray, '/visualization/detected_obstacles', 10)
        # Publish trajectory as Path for RViz
        self.trajectory_viz_pub = self.create_publisher(
            Path, '/visualization/trajectory', 10)
        
        # ==================== Subscribers ====================
        self.odom_sub = self.create_subscription(
            Odometry, '/carla/ego_vehicle/ground_truth/odometry',
            self.odom_callback, 10)
        
        self.trajectory_sub = self.create_subscription(
            Trajectory, '/planning/trajectory',
            self.trajectory_callback, 10)
        
        self.global_route_sub = self.create_subscription(
            Path, '/planning/global_route',
            self.global_route_callback, 10)
        
        self.predicted_sub = self.create_subscription(
            Path, '/control/predicted_trajectory',
            self.predicted_callback, 10)
        
        # Subscribe to obstacle topics - needed for both CARLA and RViz visualization
        self.gt_obstacles_sub = self.create_subscription(
            ObstacleArray, '/perception/ground_truth/obstacles',
            self.ground_truth_obstacles_callback, 10)
        self.get_logger().info('Subscribed to ground truth obstacles')
        
        self.detected_obstacles_sub = self.create_subscription(
            ObstacleArray, '/perception/obstacles',
            self.detected_obstacles_callback, 10)
        self.get_logger().info('Subscribed to detected obstacles')
        
        # ==================== Visualization Timer ====================
        viz_period = 1.0 / viz_rate
        self.viz_timer = self.create_timer(viz_period, self.visualization_loop)
        
        self.get_logger().info(f'Visualization Node initialized ({viz_rate}Hz)')
    
    # ==================== Callbacks ====================
    
    def odom_callback(self, msg: Odometry):
        """Update vehicle state from odometry."""
        pos = msg.pose.pose.position
        quat = msg.pose.pose.orientation
        vel = msg.twist.twist.linear
        
        # Store ROS coordinates (right-handed)
        self.vehicle_x = pos.x
        self.vehicle_y = pos.y
        self.vehicle_yaw = quaternion_to_yaw(quat)
        self.vehicle_speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
    
    def trajectory_callback(self, msg: Trajectory):
        """Receive local trajectory from planning node and publish as Path for RViz."""
        waypoints = []
        
        # Extract target speed from first point (if available)
        if msg.points:
            self.target_speed = msg.points[0].speed
            
        for point in msg.points:
            waypoints.append((point.x, point.y))
        self.trajectory = waypoints
        
        # Publish as Path for RViz visualization
        self._publish_trajectory_path(msg)
    
    def global_route_callback(self, msg: Path):
        """Receive global route from planning node."""
        waypoints = []
        for pose in msg.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            z = pose.pose.position.z
            waypoints.append((x, y, z))
        self.global_route = waypoints
    
    def predicted_callback(self, msg: Path):
        """Receive MPC predicted trajectory from control node."""
        states = []
        for pose in msg.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            
            # Extract yaw from quaternion
            quat = pose.pose.orientation
            yaw = quaternion_to_yaw(quat)
            
            # Speed not available in Path, use 0
            states.append((x, y, yaw, 0.0))
        self.predicted = states
    
    def _parse_obstacle_array(self, msg: ObstacleArray) -> List[dict]:
        """
        Parse ObstacleArray message into list of obstacle dicts.
        
        Converts from ego frame to world frame (ROS coordinates) for visualization.
        """
        if self.vehicle_x is None:
            return []
        
        obstacles = []
        vx, vy, vyaw = self.vehicle_x, self.vehicle_y, self.vehicle_yaw or 0.0
        c, s = math.cos(vyaw), math.sin(vyaw)
        
        for obstacle in msg.obstacles:
            # Obstacle position is in ego frame (x=forward, y=left in ROS)
            ox_ego = obstacle.pose.position.x
            oy_ego = obstacle.pose.position.y
            
            # Transform to world coordinates (ROS frame)
            ox_world = vx + ox_ego * c - oy_ego * s
            oy_world = vy + ox_ego * s + oy_ego * c
            
            # Get obstacle yaw (relative + ego)
            obs_quat = obstacle.pose.orientation
            obs_yaw_rel = quaternion_to_yaw(obs_quat)
            obs_yaw_world = vyaw + obs_yaw_rel
            
            # Get dimensions from size
            length = obstacle.size.x
            width = obstacle.size.y
            height = obstacle.size.z
            
            # Get type from message
            obs_type = obstacle.type
            
            # Get velocity
            obs_vx = obstacle.velocity.x
            obs_vy = obstacle.velocity.y
            
            obstacles.append({
                'x': ox_world,
                'y': oy_world,
                'yaw': obs_yaw_world,
                'length': length,
                'width': width,
                'height': height,
                'type': obs_type,
                'vx': obs_vx,
                'vy': obs_vy,
            })
        
        return obstacles
    
    def _obstacles_to_marker_array(self, obstacles: List[dict], namespace: str, 
                                    color: Tuple[float, float, float]) -> MarkerArray:
        """Convert obstacle list to MarkerArray for RViz."""
        marker_array = MarkerArray()
        
        # Clear previous markers
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
            
            # Obstacles are now in ROS world coordinates, use directly
            marker.pose.position.x = obs['x']
            marker.pose.position.y = obs['y']
            marker.pose.position.z = obs.get('height', 1.8) / 2
            
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = math.sin(obs['yaw'] / 2)
            marker.pose.orientation.w = math.cos(obs['yaw'] / 2)
            
            marker.scale.x = obs['length']
            marker.scale.y = obs['width']
            marker.scale.z = obs['height']
            
            marker.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=0.7)
            marker.lifetime.nanosec = 200000000
            
            marker_array.markers.append(marker)
            
            # Add text label
            text_marker = Marker()
            text_marker.header = marker.header
            text_marker.ns = f'{namespace}_labels'
            text_marker.id = i + 1
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = marker.pose.position.x
            text_marker.pose.position.y = marker.pose.position.y
            text_marker.pose.position.z = marker.pose.position.z + obs['height'] / 2 + 0.5
            text_marker.text = f"{obs['type']}"
            text_marker.scale.z = 0.5
            text_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.9)
            text_marker.lifetime.nanosec = 200000000
            marker_array.markers.append(text_marker)
        
        return marker_array
    
    def ground_truth_obstacles_callback(self, msg: ObstacleArray):
        """Receive ground truth obstacles and convert to markers."""
        self.ground_truth_obstacles = self._parse_obstacle_array(msg)
        
        # Publish MarkerArray for RViz
        marker_array = self._obstacles_to_marker_array(
            self.ground_truth_obstacles, 'ground_truth_obstacles', (0.0, 1.0, 0.0))
        self.gt_obstacles_marker_pub.publish(marker_array)
    
    def detected_obstacles_callback(self, msg: ObstacleArray):
        """Receive detected obstacles and convert to markers."""
        self.get_logger().info('Received detected obstacles')
        self.detected_obstacles = self._parse_obstacle_array(msg)
        
        # Publish MarkerArray for RViz
        marker_array = self._obstacles_to_marker_array(
            self.detected_obstacles, 'detected_obstacles', (1.0, 1.0, 0.0))
        self.detected_obstacles_marker_pub.publish(marker_array)
    
    def _publish_trajectory_path(self, msg: Trajectory):
        """Convert Trajectory message to Path for RViz visualization."""
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        for point in msg.points:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = point.x
            pose.pose.position.y = point.y
            pose.pose.position.z = 0.0  # Flat for visualization
            # Set orientation from yaw
            pose.pose.orientation.z = math.sin(point.yaw / 2)
            pose.pose.orientation.w = math.cos(point.yaw / 2)
            path_msg.poses.append(pose)
        
        self.trajectory_viz_pub.publish(path_msg)
    
    # ==================== Visualization Loop ====================
    
    def visualization_loop(self):
        """Main visualization loop - draws in CARLA and updates 2D viz."""
        if self.vehicle_x is None:
            return
        
        # ==================== CARLA Debug Drawing ====================
        if self.enable_carla_viz and self.carla_drawer:
            self._draw_carla()
        
        # ==================== 2D Matplotlib Visualizer ====================
        if self.enable_2d_viz and self.viz_2d:
            self._update_2d_viz()
    
    def _draw_carla(self):
        """Draw debug shapes in CARLA (applies Y-flip for CARLA's left-handed coords)."""
        lt = self.draw_lifetime
        items = self.carla_draw_items
        
        # Draw global route (flip Y for CARLA)
        if 'global_route' in items and self.global_route:
            route_carla = [(x, -y, z) for x, y, z in self.global_route]
            self.carla_drawer.draw_global_route(route_carla, lifetime=lt)
        
        # Draw local trajectory (flip Y for CARLA)
        if 'trajectory' in items and self.trajectory:
            traj_carla = [(x, -y) for x, y in self.trajectory]
            self.carla_drawer.draw_trajectory(traj_carla, lifetime=lt)
        
        # Draw MPC predicted trajectory (flip Y and yaw for CARLA)
        if 'predicted' in items and self.predicted:
            pred_carla = [(x, -y, -yaw, speed) for x, y, yaw, speed in self.predicted]
            self.carla_drawer.draw_predicted_trajectory(pred_carla, lifetime=lt)
        
        # Draw speed indicator (flip Y for CARLA)
        if 'speed' in items and self.vehicle_x is not None:
            self.carla_drawer.draw_speed_indicator(
                self.vehicle_x, -self.vehicle_y,
                self.vehicle_speed * 3.6,  # Convert to km/h
                self.target_speed * 3.6,   # Target speed from trajectory
                lifetime=lt
            )
        
        # Ground truth obstacles (green boxes, flip Y and yaw for CARLA)
        if 'ground_truth_obstacles' in items:
            for obs in self.ground_truth_obstacles:
                self.carla_drawer.draw_obstacle_box(
                    obs['x'], -obs['y'], -obs['yaw'],
                    obs['length'], obs['width'], obs.get('height', 1.8),
                    is_vehicle=(obs['type'] == 'vehicle'),
                    lifetime=lt,
                    color=(0, 255, 0)  # Green for ground truth
                )
        
        # Detected obstacles (yellow boxes, flip Y and yaw for CARLA)
        if 'detected_obstacles' in items:
            for obs in self.detected_obstacles:
                self.carla_drawer.draw_obstacle_box(
                    obs['x'], -obs['y'], -obs['yaw'],
                    obs['length'], obs['width'], obs.get('height', 1.8),
                    is_vehicle=(obs['type'] == 'vehicle'),
                    lifetime=lt,
                    color=(255, 255, 0)  # Yellow for detected
                )
    
    def _update_2d_viz(self):
        """Update 2D matplotlib visualizer."""
        # Update vehicle
        if self.vehicle_x is not None:
            self.viz_2d.update_vehicle(
                self.vehicle_x, self.vehicle_y, 
                self.vehicle_yaw or 0.0, self.vehicle_speed
            )
        
        # Update trajectory
        if self.trajectory:
            self.viz_2d.update_trajectory(self.trajectory)
        
        # Update global route
        if self.global_route:
            route_2d = [(p[0], p[1]) for p in self.global_route]
            self.viz_2d.update_global_route(route_2d)
        
        # Update predicted
        if self.predicted:
            self.viz_2d.update_predicted(self.predicted)
        
        # Update obstacles - combine both lists for 2D viz with type tag
        all_obstacles = []
        if 'ground_truth_obstacles' in self.carla_draw_items:
            for obs in self.ground_truth_obstacles:
                obs_copy = obs.copy()
                obs_copy['source'] = 'ground_truth'
                all_obstacles.append(obs_copy)
        if 'detected_obstacles' in self.carla_draw_items:
            for obs in self.detected_obstacles:
                obs_copy = obs.copy()
                obs_copy['source'] = 'detected'
                all_obstacles.append(obs_copy)
        if all_obstacles:
            self.viz_2d.update_obstacles(all_obstacles)
    
    def destroy_node(self):
        """Cleanup on shutdown."""
        if self.viz_2d:
            self.viz_2d.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VisualizationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
