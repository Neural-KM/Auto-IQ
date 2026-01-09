#!/usr/bin/env python3
"""
Pure Pursuit Control Node
=========================

ROS2 node for vehicle control using Pure Pursuit (lateral) + PID (longitudinal).
Subscribes to trajectory and odometry, publishes vehicle control commands.

Subscribes:
- /planning/trajectory (nav_msgs/Path) - Planned trajectory from planning node
- /carla/ego_vehicle/ground_truth/odometry (nav_msgs/Odometry) - Vehicle state

Publishes:
- /carla/ego_vehicle/control (autoiq_msgs/VehicleControl) - Vehicle control
- /control/status (std_msgs/String) - Control status
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path, Odometry
from std_msgs.msg import String
from autoiq_msgs.msg import VehicleControl

from control.utils.pid_controller import PIDController


def quaternion_to_yaw(q):
    """Convert quaternion to yaw angle in radians."""
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class PurePursuitNode(Node):
    """
    ROS2 Pure Pursuit + PID Control Node.
    
    Uses Pure Pursuit for lateral (steering) control and PID for 
    longitudinal (speed) control. Follows trajectories received from
    the planning node.
    """
    
    def __init__(self):
        super().__init__('pure_pursuit_node')
        
        # ==================== Parameters ====================
        self.declare_parameter('target_speed', 30.0)  # km/h
        self.declare_parameter('max_steering_diff', 0.1)
        self.declare_parameter('control_rate', 20.0)  # Hz
        
        # PID parameters (longitudinal)
        self.declare_parameter('pid_kp', 0.2)
        self.declare_parameter('pid_ki', 0.2)
        self.declare_parameter('pid_kd', 0.0)
        self.declare_parameter('max_throttle', 0.75)
        self.declare_parameter('max_brake', 0.5)
        self.declare_parameter('max_steering', 1.0)
        
        # Pure Pursuit parameters (lateral)
        self.declare_parameter('wheelbase', 2.875)
        self.declare_parameter('min_lookahead', 4.0)
        self.declare_parameter('max_lookahead', 25.0)
        self.declare_parameter('lookahead_ratio', 0.5)
        
        # Get parameters
        self.target_speed = self.get_parameter('target_speed').value
        self.max_steering_diff = self.get_parameter('max_steering_diff').value
        control_rate = self.get_parameter('control_rate').value
        
        pid_kp = self.get_parameter('pid_kp').value
        pid_ki = self.get_parameter('pid_ki').value
        pid_kd = self.get_parameter('pid_kd').value
        self.max_throttle = self.get_parameter('max_throttle').value
        self.max_brake = self.get_parameter('max_brake').value
        self.max_steering = self.get_parameter('max_steering').value
        
        self.wheelbase = self.get_parameter('wheelbase').value
        self.min_lookahead = self.get_parameter('min_lookahead').value
        self.max_lookahead = self.get_parameter('max_lookahead').value
        self.lookahead_ratio = self.get_parameter('lookahead_ratio').value
        
        # ==================== Initialize Controllers ====================
        # PID for longitudinal control
        output_limits = (-self.max_brake, self.max_throttle)
        self.pid = PIDController(pid_kp, pid_ki, pid_kd, 1.0/control_rate, output_limits)
        
        # ==================== State ====================
        self.vehicle_x = None
        self.vehicle_y = None
        self.vehicle_yaw = None
        self.vehicle_speed = 0.0
        
        self.current_waypoints = None
        self.has_trajectory = False
        
        self.previous_steering = 0.0
        
        # Store current lookahead point for debugging
        self._current_lookahead_point = None
        self._current_lookahead_distance = 0.0
        
        # ==================== Subscribers ====================
        self.trajectory_sub = self.create_subscription(
            Path, '/planning/trajectory',
            self.trajectory_callback, 10)
        
        self.odom_sub = self.create_subscription(
            Odometry, '/carla/ego_vehicle/ground_truth/odometry',
            self.odom_callback, 10)
        
        # ==================== Publishers ====================
        self.control_pub = self.create_publisher(
            VehicleControl, 
            '/carla/ego_vehicle/control', 
            10)
        
        self.status_pub = self.create_publisher(String, '/control/status', 10)
        
        # ==================== Control Loop ====================
        control_period = 1.0 / control_rate
        self.control_timer = self.create_timer(control_period, self.control_loop)
        
        self.get_logger().info(f'Pure Pursuit Node initialized ({control_rate}Hz)')
        self.get_logger().info(f'  Wheelbase: {self.wheelbase}m')
        self.get_logger().info(f'  Lookahead: {self.min_lookahead}-{self.max_lookahead}m')
        self.get_logger().info(f'  Target speed: {self.target_speed} km/h')
    
    # ==================== Callbacks ====================
    
    def trajectory_callback(self, msg: Path):
        """Receive trajectory from planning node."""
        if not msg.poses:
            self.has_trajectory = False
            self.current_waypoints = None
            return
        
        # Convert Path to list of (x, y) waypoints
        # Note: Convert back from ROS to CARLA coordinates
        waypoints = []
        for pose in msg.poses:
            x = pose.pose.position.x
            y = -pose.pose.position.y  # Y-axis inversion
            waypoints.append((x, y))
        
        self.current_waypoints = waypoints
        self.has_trajectory = True
    
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
        
        # Speed in m/s
        self.vehicle_speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
    
    # ==================== Control Loop ====================
    
    def control_loop(self):
        """Main control loop."""
        # Check if we have valid state
        if self.vehicle_x is None or self.vehicle_y is None:
            return
        
        # Publish status
        self._publish_status()
        
        # Check if we have trajectory
        if not self.has_trajectory or not self.current_waypoints:
            # No trajectory - publish stop command
            self._publish_stop()
            return
        
        # Pure Pursuit for steering (lateral control)
        steering = self._compute_steering()
        
        # PID for speed (longitudinal control)
        throttle, brake = self._compute_throttle_brake()
        
        # Apply steering smoothing
        steering = self._smooth_steering(steering)
        
        # Publish control
        self._publish_control(steering, throttle, brake)
    
    def _compute_steering(self):
        """Compute steering using Pure Pursuit algorithm."""
        if not self.current_waypoints:
            return 0.0
        
        # Compute adaptive lookahead distance based on speed
        lookahead = self.min_lookahead + self.lookahead_ratio * self.vehicle_speed
        lookahead = np.clip(lookahead, self.min_lookahead, self.max_lookahead)
        self._current_lookahead_distance = lookahead
        
        # Find lookahead point on path
        lookahead_point = self._find_lookahead_point(lookahead)
        
        if lookahead_point is None:
            return 0.0
        
        self._current_lookahead_point = lookahead_point
        
        # Vector from vehicle to lookahead point
        dx = lookahead_point[0] - self.vehicle_x
        dy = lookahead_point[1] - self.vehicle_y
        ld = math.sqrt(dx**2 + dy**2)
        
        # Prevent division by zero
        if ld < self.min_lookahead * 0.5:
            return 0.0
        
        ld = max(ld, self.min_lookahead)
        
        # Transform to vehicle frame (rotate by -yaw)
        cos_yaw = math.cos(-self.vehicle_yaw)
        sin_yaw = math.sin(-self.vehicle_yaw)
        local_y = dx * sin_yaw + dy * cos_yaw
        
        # Pure pursuit formula: curvature = 2 * y / L^2
        curvature = 2.0 * local_y / (ld * ld)
        
        # Steering angle from curvature: delta = atan(L * curvature)
        steering_angle = math.atan(self.wheelbase * curvature)
        
        # Normalize to [-1, 1] range
        max_steering_angle = math.radians(35.0)  # ~35 degrees max
        steering = steering_angle / max_steering_angle
        
        return np.clip(steering, -self.max_steering, self.max_steering)
    
    def _find_lookahead_point(self, lookahead_distance):
        """Find point on path at lookahead distance."""
        if not self.current_waypoints:
            return None
        
        # Find first waypoint beyond lookahead distance
        for wp in self.current_waypoints:
            dist = math.sqrt(
                (wp[0] - self.vehicle_x)**2 + 
                (wp[1] - self.vehicle_y)**2
            )
            if dist >= lookahead_distance:
                return wp
        
        # If no point found, return last waypoint
        if self.current_waypoints:
            return self.current_waypoints[-1]
        
        return None
    
    def _compute_throttle_brake(self):
        """Compute throttle/brake using PID speed control."""
        current_speed_kmh = self.vehicle_speed * 3.6
        error = self.target_speed - current_speed_kmh
        
        accel = self.pid.step(error)
        
        if accel >= 0:
            throttle = min(accel, self.max_throttle)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(abs(accel), self.max_brake)
        
        return float(throttle), float(brake)
    
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
        status = {
            'controller': 'pure_pursuit',
            'has_trajectory': self.has_trajectory,
            'num_waypoints': len(self.current_waypoints) if self.current_waypoints else 0,
            'vehicle_speed_kmh': self.vehicle_speed * 3.6 if self.vehicle_speed else 0.0,
            'target_speed_kmh': self.target_speed,
            'lookahead_distance': self._current_lookahead_distance,
            'steering': self.previous_steering
        }
        
        status_msg = String()
        status_msg.data = str(status)
        self.status_pub.publish(status_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
