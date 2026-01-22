#!/usr/bin/env python3
"""
Live 2D Visualizer Node - ROS2 node for 2D matplotlib visualization of vehicle state, trajectory,
obstacles, and routes.
"""
import math
from dataclasses import replace
from typing import List

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path, Odometry

from autoiq_msgs.msg import ObstacleArray as ObstacleArrayMsg, Trajectory as TrajectoryMsg

from autoiq_common.types import VehicleState, Trajectory, ObstacleArray, Obstacle
from autoiq_common.converters import (
    vehicle_state_from_odometry,
    trajectory_from_ros,
    obstacle_array_from_ros,
)
from autoiq_common.utils import quaternion_to_yaw

from visualization.live_2d_visualizer import Live2DVisualizerROS


class Live2DVizNode(Node):
    """
    ROS2 Node for 2D matplotlib visualization.
    
    Updates the visualizer immediately upon receiving new ROS messages.
    """
    
    def __init__(self):
        super().__init__('live_2d_viz_node')
        
        # ==================== Parameters ====================
        self.declare_parameter('plot_refresh_rate', 20.0)  # Hz for GUI refresh
        self.declare_parameter('show_ground_truth_obstacles', True)
        self.declare_parameter('show_detected_obstacles', True)
        
        plot_refresh_rate = self.get_parameter('plot_refresh_rate').value
        self.show_ground_truth_obstacles = self.get_parameter('show_ground_truth_obstacles').value
        self.show_detected_obstacles = self.get_parameter('show_detected_obstacles').value
        
        # ==================== 2D Visualizer ====================
        self.viz_2d = Live2DVisualizerROS(title="Auto-IQ Live 2D", refresh_hz=plot_refresh_rate)
        self.viz_2d.start()
        self.get_logger().info('2D matplotlib visualization started (Event-Driven Mode)')
        
        # ==================== Data Storage ====================
        # We need to persist these because obstacles require vehicle state for transformation,
        # and we need to merge GT and Detected lists before sending to viz.
        self.vehicle_state: VehicleState = None
        self.ground_truth_obstacles: List[Obstacle] = []
        self.detected_obstacles: List[Obstacle] = []
        
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
        
        self.get_logger().info('Live 2D Viz Node initialized')
    
    # ==================== Callbacks ====================
    
    def odom_callback(self, msg: Odometry):
        """Update vehicle state and immediately trigger visualization update."""
        self.vehicle_state = vehicle_state_from_odometry(msg)
        
        # Immediate update
        self.viz_2d.update_vehicle(
            self.vehicle_state.x, self.vehicle_state.y,
            self.vehicle_state.yaw, self.vehicle_state.speed
        )
    
    def trajectory_callback(self, msg: TrajectoryMsg):
        """Receive and visualize local trajectory."""
        traj = trajectory_from_ros(msg)
        waypoints = [(point.x, point.y) for point in traj.points]
        
        # Immediate update
        self.viz_2d.update_trajectory(waypoints)
    
    def global_route_callback(self, msg: Path):
        """Receive and visualize global route."""
        waypoints = []
        for pose in msg.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            waypoints.append((x, y))
            
        # Immediate update
        self.viz_2d.update_global_route(waypoints)
    
    def predicted_callback(self, msg: Path):
        """Receive and visualize MPC predicted trajectory."""
        states = []
        for pose in msg.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            quat = pose.pose.orientation
            yaw = quaternion_to_yaw(quat)
            states.append((x, y, yaw, 0.0))
            
        # Immediate update
        self.viz_2d.update_predicted_trajectory(states)
    
    def ground_truth_obstacles_callback(self, msg: ObstacleArrayMsg):
        """Receive GT obstacles, update internal list, and push to viz."""
        obstacle_array = obstacle_array_from_ros(msg)
        self.ground_truth_obstacles = self._transform_obstacles_to_world(obstacle_array)
        self._update_viz_obstacles()
    
    def detected_obstacles_callback(self, msg: ObstacleArrayMsg):
        """Receive Detected obstacles, update internal list, and push to viz."""
        obstacle_array = obstacle_array_from_ros(msg)
        self.detected_obstacles = self._transform_obstacles_to_world(obstacle_array)
        self._update_viz_obstacles()
    
    # ==================== Helpers ====================

    def _update_viz_obstacles(self):
        """Merge GT and Detected obstacles and push to visualizer."""
        all_obstacles = []
        
        if self.show_ground_truth_obstacles:
            all_obstacles.extend(self.ground_truth_obstacles)
            
        if self.show_detected_obstacles:
            all_obstacles.extend(self.detected_obstacles)
            
        if all_obstacles:
            # Convert to dict format expected by Live2DVisualizer
            obs_dicts = [{
                'x': o.x, 
                'y': o.y, 
                'yaw': o.yaw, 
                'length': o.length,
                'width': o.width, 
                'type': o.type
            } for o in all_obstacles]
            
            self.viz_2d.update_obstacles(obs_dicts)

    def _transform_obstacles_to_world(self, obstacle_array: ObstacleArray) -> List[Obstacle]:
        """Transform obstacles from ego frame to world frame using current vehicle state."""
        if self.vehicle_state is None:
            # Cannot transform if we don't know where the vehicle is
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
    
    def destroy_node(self):
        """Cleanup on shutdown."""
        if self.viz_2d:
            self.viz_2d.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Live2DVizNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()