#!/usr/bin/env python3
"""
Visualization Node
==================

ROS2 node for visualizing Auto-IQ data in both CARLA (3D debug drawings)
and matplotlib (2D plot window).

Subscribes to:
- /carla/ego_vehicle/ground_truth/odometry (nav_msgs/Odometry) - Vehicle state
- /planning/trajectory (nav_msgs/Path) - Local trajectory
- /planning/global_route (nav_msgs/Path) - Global route
- /control/predicted_trajectory (nav_msgs/Path) - MPC predicted path
- /perception/obstacles (MarkerArray) - Detected obstacles (optional)
- /perception/ground_truth/obstacles (MarkerArray) - Ground truth obstacles (optional)

Connects to CARLA for debug drawing (optional).
Runs matplotlib visualizer in a separate thread.
"""

import math
import numpy as np
from typing import List, Tuple

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path, Odometry
from visualization_msgs.msg import MarkerArray

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
        self.declare_parameter('enable_carla_viz', True)
        self.declare_parameter('enable_2d_viz', False)
        self.declare_parameter('viz_rate', 10.0)  # Hz
        self.declare_parameter('draw_lifetime', 0.15)  # seconds
        
        # Obstacle visualization options - can enable both independently
        self.declare_parameter('draw_ground_truth_obstacles', False)
        self.declare_parameter('draw_detected_obstacles', True)
        
        carla_host = self.get_parameter('carla_host').value
        carla_port = self.get_parameter('carla_port').value
        carla_timeout = self.get_parameter('carla_timeout').value
        self.enable_carla_viz = self.get_parameter('enable_carla_viz').value
        self.enable_2d_viz = self.get_parameter('enable_2d_viz').value
        viz_rate = self.get_parameter('viz_rate').value
        self.draw_lifetime = self.get_parameter('draw_lifetime').value
        
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
        # Vehicle state (CARLA coordinates)
        self.vehicle_x = None
        self.vehicle_y = None
        self.vehicle_yaw = None
        self.vehicle_speed = 0.0
        
        # Paths (CARLA coordinates)
        self.trajectory = []  # List of (x, y)
        self.global_route = []  # List of (x, y, z)
        self.predicted = []  # List of (x, y, yaw, speed)
        
        # Obstacles (world coordinates) - separate for ground truth and detected
        self.ground_truth_obstacles = []  # List of dicts
        self.detected_obstacles = []  # List of dicts
        
        # ==================== Subscribers ====================
        self.odom_sub = self.create_subscription(
            Odometry, '/carla/ego_vehicle/ground_truth/odometry',
            self.odom_callback, 10)
        
        self.trajectory_sub = self.create_subscription(
            Path, '/planning/trajectory',
            self.trajectory_callback, 10)
        
        self.global_route_sub = self.create_subscription(
            Path, '/planning/global_route',
            self.global_route_callback, 10)
        
        self.predicted_sub = self.create_subscription(
            Path, '/control/predicted_trajectory',
            self.predicted_callback, 10)
        
        # Subscribe to obstacle topics based on parameters
        self.draw_ground_truth_obstacles = self.get_parameter('draw_ground_truth_obstacles').value
        self.draw_detected_obstacles = self.get_parameter('draw_detected_obstacles').value
        
        if self.draw_ground_truth_obstacles:
            self.gt_obstacles_sub = self.create_subscription(
                MarkerArray, '/perception/ground_truth/obstacles',
                self.ground_truth_obstacles_callback, 10)
            self.get_logger().info('Subscribed to ground truth obstacles')
        
        if self.draw_detected_obstacles:
            self.detected_obstacles_sub = self.create_subscription(
                MarkerArray, '/perception/obstacles',
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
        
        # Convert to CARLA coordinates
        self.vehicle_x = pos.x
        self.vehicle_y = -pos.y  # Y-axis inversion
        
        yaw_rad = quaternion_to_yaw(quat)
        self.vehicle_yaw = -yaw_rad  # Yaw inversion for CARLA
        
        self.vehicle_speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
    
    def trajectory_callback(self, msg: Path):
        """Receive local trajectory from planning node."""
        waypoints = []
        for pose in msg.poses:
            x = pose.pose.position.x
            y = -pose.pose.position.y  # Y-axis inversion to CARLA
            waypoints.append((x, y))
        self.trajectory = waypoints
    
    def global_route_callback(self, msg: Path):
        """Receive global route from planning node."""
        waypoints = []
        for pose in msg.poses:
            x = pose.pose.position.x
            y = -pose.pose.position.y  # Y-axis inversion to CARLA
            z = pose.pose.position.z
            waypoints.append((x, y, z))
        self.global_route = waypoints
    
    def predicted_callback(self, msg: Path):
        """Receive MPC predicted trajectory from control node."""
        states = []
        for pose in msg.poses:
            x = pose.pose.position.x
            y = -pose.pose.position.y  # Y-axis inversion to CARLA
            
            # Extract yaw from quaternion
            quat = pose.pose.orientation
            yaw = -quaternion_to_yaw(quat)  # Yaw inversion for CARLA
            
            # Speed not available in Path, use 0
            states.append((x, y, yaw, 0.0))
        self.predicted = states
    
    def obstacles_callback(self, msg: MarkerArray):
        """Receive obstacles from perception node (generic parser)."""
        if self.vehicle_x is None:
            return
        
        obstacles = []
        vx, vy, vyaw = self.vehicle_x, self.vehicle_y, self.vehicle_yaw or 0.0
        c, s = math.cos(vyaw), math.sin(vyaw)
        
        for marker in msg.markers:
            # Marker position is in ego frame (x=forward, y=left)
            ox_ego = marker.pose.position.x
            oy_ego = marker.pose.position.y
            
            # Transform to world coordinates
            ox_world = vx + ox_ego * c - oy_ego * s
            oy_world = vy + ox_ego * s + oy_ego * c
            
            # Get obstacle yaw (relative + ego)
            obs_quat = marker.pose.orientation
            obs_yaw_rel = quaternion_to_yaw(obs_quat)
            obs_yaw_world = vyaw + obs_yaw_rel
            
            # Get dimensions from marker scale
            length = marker.scale.x
            width = marker.scale.y
            height = marker.scale.z
            
            # Determine type from namespace or color
            obs_type = 'vehicle' if 'vehicle' in marker.ns.lower() else 'pedestrian'
            
            obstacles.append({
                'x': ox_world,
                'y': oy_world,
                'yaw': obs_yaw_world,
                'length': length,
                'width': width,
                'height': height,
                'type': obs_type
            })
        
        return obstacles
    
    def ground_truth_obstacles_callback(self, msg: MarkerArray):
        """Receive ground truth obstacles."""
        self.ground_truth_obstacles = self.obstacles_callback(msg) or []
    
    def detected_obstacles_callback(self, msg: MarkerArray):
        """Receive detected obstacles from perception."""
        self.detected_obstacles = self.obstacles_callback(msg) or []
    
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
        """Draw debug shapes in CARLA."""
        lt = self.draw_lifetime
        
        # Draw global route
        if self.global_route:
            self.carla_drawer.draw_global_route(self.global_route, lifetime=lt)
        
        # Draw local trajectory
        if self.trajectory:
            self.carla_drawer.draw_trajectory(self.trajectory, lifetime=lt)
        
        # Draw MPC predicted trajectory
        if self.predicted:
            self.carla_drawer.draw_predicted_trajectory(self.predicted, lifetime=lt)
        
        # Draw speed indicator
        if self.vehicle_x is not None:
            self.carla_drawer.draw_speed_indicator(
                self.vehicle_x, self.vehicle_y,
                self.vehicle_speed * 3.6,  # Convert to km/h
                30.0,  # Target speed (could be parameter)
                lifetime=lt
            )
        
        # Draw obstacles
        # Ground truth obstacles (green boxes)
        if self.draw_ground_truth_obstacles:
            for obs in self.ground_truth_obstacles:
                self.carla_drawer.draw_obstacle_box(
                    obs['x'], obs['y'], obs['yaw'],
                    obs['length'], obs['width'], obs.get('height', 1.8),
                    is_vehicle=(obs['type'] == 'vehicle'),
                    lifetime=lt,
                    color=(0, 255, 0)  # Green for ground truth
                )
        
        # Detected obstacles (yellow boxes)
        if self.draw_detected_obstacles:
            for obs in self.detected_obstacles:
                self.carla_drawer.draw_obstacle_box(
                    obs['x'], obs['y'], obs['yaw'],
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
        if self.draw_ground_truth_obstacles:
            for obs in self.ground_truth_obstacles:
                obs_copy = obs.copy()
                obs_copy['source'] = 'ground_truth'
                all_obstacles.append(obs_copy)
        if self.draw_detected_obstacles:
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
