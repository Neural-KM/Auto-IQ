#!/usr/bin/env python3
"""
MPC Control Node
================

ROS2 node for vehicle control using Model Predictive Control (MPC).
Subscribes to trajectory and odometry, publishes vehicle control commands.

Subscribes:
- /planning/trajectory (autoiq_msgs/Trajectory) - Planned trajectory from planning node
- /carla/ego_vehicle/ground_truth/odometry (nav_msgs/Odometry) - Vehicle state

Publishes:
- /carla/ego_vehicle/control (autoiq_msgs/VehicleControl) - Vehicle control
- /control/predicted_trajectory (nav_msgs/Path) - MPC predicted trajectory
- /control/status (std_msgs/String) - Control status
"""

import math
import time
import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from autoiq_msgs.msg import VehicleControl, Trajectory as TrajectoryMsg

# Import common types and converters
from autoiq_common.types import VehicleState
from autoiq_common.converters import vehicle_state_from_odometry, trajectory_from_ros
from autoiq_common.utils import yaw_to_quaternion

from control.controllers.mpc_controller import MPCController


class MPCNode(Node):
    """
    ROS2 MPC Control Node.
    
    Uses Nonlinear Model Predictive Control (NMPC) for trajectory tracking.
    The MPC optimizes over a prediction horizon to find optimal steering
    and acceleration inputs.
    """
    
    def __init__(self):
        super().__init__('mpc_node')
        
        # ==================== Parameters ====================
        self.declare_parameter('mpc_horizon', 20)
        self.declare_parameter('mpc_dt', 0.05)
        self.declare_parameter('trajectory_mode', 0)  # 0: heuristic, 1: trapezoidal
        self.declare_parameter('max_steering_diff', 0.1)
        self.declare_parameter('control_rate', 20.0)  # Hz
        
        # Get parameters
        mpc_horizon = self.get_parameter('mpc_horizon').value
        mpc_dt = self.get_parameter('mpc_dt').value
        trajectory_mode = self.get_parameter('trajectory_mode').value
        self.max_steering_diff = self.get_parameter('max_steering_diff').value
        control_rate = self.get_parameter('control_rate').value
        
        # ==================== Initialize MPC Controller ====================
        self.get_logger().info(f'Initializing MPC (horizon={mpc_horizon}, dt={mpc_dt}, mode={trajectory_mode})')
        
        self.mpc = MPCController(
            dt=mpc_dt, 
            horizon=mpc_horizon, 
            trajectory_mode=trajectory_mode
        )
        
        # ==================== State ====================
        self.vehicle_state: VehicleState = None
        self.current_waypoints = None
        self.current_target_speeds = []
        self.has_trajectory = False
        self.current_behavior_state = TrajectoryMsg.LANE_FOLLOWING
        
        self.previous_steering = 0.0
        
        # Performance monitoring
        self._last_solve_time_ms = 0.0
        self._last_optimal_control = None
        
        # ==================== Subscribers ====================
        self.trajectory_sub = self.create_subscription(
            TrajectoryMsg, '/planning/trajectory',
            self.trajectory_callback, 10)
        
        self.odom_sub = self.create_subscription(
            Odometry, '/carla/ego_vehicle/ground_truth/odometry',
            self.odom_callback, 10)
        
        # ==================== Publishers ====================
        self.control_pub = self.create_publisher(
            VehicleControl, 
            '/carla/ego_vehicle/control', 
            10)
        
        self.predicted_pub = self.create_publisher(
            Path,
            '/control/predicted_trajectory',
            10)
        
        self.status_pub = self.create_publisher(String, '/control/status', 10)
        
        # ==================== Control Loop ====================
        control_period = 1.0 / control_rate
        self.control_timer = self.create_timer(control_period, self.control_loop)
        
        self.get_logger().info(f'MPC Node initialized ({control_rate}Hz)')
        self.get_logger().info(f'  Horizon: {mpc_horizon} steps, dt: {mpc_dt}s')
    
    # ==================== Callbacks ====================
    
    def trajectory_callback(self, msg: TrajectoryMsg):
        """Receive trajectory from planning node."""
        traj = trajectory_from_ros(msg)
        
        if not traj.points:
            self.has_trajectory = False
            self.current_waypoints = None
            return
        
        # Convert Trajectory to list of (x, y) waypoints for MPC
        waypoints = []
        speeds = []
        for point in traj.points:
            waypoints.append((point.x, point.y))
            speeds.append(point.speed)
        
        self.current_waypoints = waypoints
        self.current_target_speeds = speeds
        self.has_trajectory = True
        
        # Update MPC waypoints
        self.mpc.set_waypoints(waypoints)
    
    def odom_callback(self, msg: Odometry):
        """Update vehicle state from odometry."""
        self.vehicle_state = vehicle_state_from_odometry(msg)
    
    # ==================== Control Loop ====================
    
    def control_loop(self):
        """Main control loop."""
        # Check if we have valid state
        if self.vehicle_state is None:
            return
        
        # Publish status
        self._publish_status()
        
        # Check if we have trajectory
        if not self.has_trajectory or not self.current_waypoints:
            # No trajectory - publish stop command
            self._publish_stop()
            return
        
        # Run MPC
        steering, throttle, brake = self._run_mpc()
        
        # Publish predicted trajectory for visualization
        self._publish_predicted_trajectory()
        
        # Apply steering smoothing
        steering = self._smooth_steering(steering)
        
        # Publish control
        self._publish_control(steering, throttle, brake)
    
    def _run_mpc(self):
        """Run MPC control step."""
        t_start = time.time()
        
        # Build current state vector [x, y, psi, v]
        current_state = np.array([
            self.vehicle_state.x, 
            self.vehicle_state.y,
            self.vehicle_state.yaw,
            self.vehicle_state.speed
        ])
        
        # Get target speed from trajectory (use first waypoint's speed as reference)
        target_speed_ms = 0.0
        if self.current_target_speeds:
            target_speed_ms = self.current_target_speeds[0]
        
        # Solve MPC optimization
        optimal_control = self.mpc.solve(current_state, target_speed_ms)
        
        self._last_solve_time_ms = (time.time() - t_start) * 1000
        
        if optimal_control is None:
            self.get_logger().warn('MPC solve failed - using previous steering with brake')
            return self.previous_steering, 0.0, 1.0
        
        self._last_optimal_control = optimal_control
        
        # Unpack optimal control [steering_rad, acceleration]
        steering_rad, accel = optimal_control
        
        # Convert steering angle to normalized [-1, 1]
        max_steer_rad = self.mpc.MAX_STEER
        steering = np.clip(steering_rad / max_steer_rad, -1.0, 1.0)
        
        # Convert acceleration to throttle/brake
        if accel >= 0:
            throttle = np.clip(accel / self.mpc.MAX_ACCEL, 0.0, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = np.clip(-accel / abs(self.mpc.MAX_DECEL), 0.0, 1.0)
        
        return float(steering), float(throttle), float(brake)
    
    def _smooth_steering(self, steering):
        """Apply steering smoothing to prevent jerky movements."""
        if steering > self.previous_steering + self.max_steering_diff:
            steering = self.previous_steering + self.max_steering_diff
        elif steering < self.previous_steering - self.max_steering_diff:
            steering = self.previous_steering - self.max_steering_diff
        
        self.previous_steering = steering
        return steering
    
    # ==================== Publishers ====================
    
    def _publish_control(self, steering, throttle, brake):
        """Publish vehicle control command."""
        control_msg = VehicleControl()
        control_msg.header.stamp = self.get_clock().now().to_msg()
        control_msg.throttle = float(throttle)
        control_msg.brake = float(brake)
        control_msg.steer = float(steering)
        control_msg.hand_brake = False
        control_msg.reverse = False
        control_msg.manual_gear_shift = False
        control_msg.gear = 0
        
        self.control_pub.publish(control_msg)
    
    def _publish_stop(self):
        """Publish stop command."""
        control_msg = VehicleControl()
        control_msg.header.stamp = self.get_clock().now().to_msg()
        control_msg.throttle = 0.0
        control_msg.brake = 1.0
        control_msg.steer = 0.0
        control_msg.hand_brake = False
        control_msg.reverse = False
        control_msg.manual_gear_shift = False
        control_msg.gear = 0
        
        self.control_pub.publish(control_msg)
    
    def _publish_status(self):
        """Publish control status."""
        # Get current target speed from trajectory
        current_target_speed_kmh = 0.0
        if self.current_target_speeds:
            current_target_speed_kmh = self.current_target_speeds[0] * 3.6  # Convert to km/h
            
        status = {
            'controller': 'mpc',
            'has_trajectory': self.has_trajectory,
            'num_waypoints': len(self.current_waypoints) if self.current_waypoints else 0,
            'vehicle_speed_kmh': self.vehicle_state.speed * 3.6 if self.vehicle_state else 0.0,
            'target_speed_kmh': current_target_speed_kmh,
            'solve_time_ms': self._last_solve_time_ms,
            'steering': self.previous_steering
        }
        
        status_msg = String()
        status_msg.data = str(status)
        self.status_pub.publish(status_msg)
    
    def _publish_predicted_trajectory(self):
        """Publish MPC predicted trajectory for visualization."""
        predicted = self.mpc.get_predicted_trajectory()
        
        if predicted is None or len(predicted) == 0:
            return
        
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        for state in predicted:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = state[0]
            pose.pose.position.y = state[1]
            pose.pose.position.z = 0.0
            
            # Convert yaw to quaternion
            yaw = state[2]
            pose.pose.orientation = yaw_to_quaternion(yaw)
            
            path_msg.poses.append(pose)
        
        self.predicted_pub.publish(path_msg)


def main(args=None):
    rclpy.init(args=args)
    node = MPCNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
