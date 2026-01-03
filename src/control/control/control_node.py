#!/usr/bin/env python3
"""
Control Node
============

ROS2 node for vehicle control (MPC or PID).
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

from control.mpc_controller import MPCController
from control.pure_pursuit_controller import PurePursuitController
from control.pid_controller import PIDController


def quaternion_to_yaw(q):
    """Convert quaternion to yaw angle in radians."""
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class ControlNode(Node):
    """
    ROS2 Control Node.
    
    Follows trajectories using MPC or PID control.
    Communication happens over ROS2 topics instead of WorldState.
    """
    
    def __init__(self):
        super().__init__('control_node')
        
        # ==================== Parameters ====================
        self.declare_parameter('controller_type', 'mpc')
        self.declare_parameter('target_speed', 30.0)  # km/h
        self.declare_parameter('mpc_horizon', 20)
        self.declare_parameter('mpc_dt', 0.05)
        self.declare_parameter('trajectory_mode', 0)
        self.declare_parameter('max_steering_diff', 0.1)
        self.declare_parameter('control_rate', 20.0)  # Hz
        
        # PID parameters
        self.declare_parameter('pid_kp', 1.0)
        self.declare_parameter('pid_ki', 0.0)
        self.declare_parameter('pid_kd', 0.0)
        self.declare_parameter('max_throttle', 0.75)
        self.declare_parameter('max_brake', 0.3)
        self.declare_parameter('max_steering', 1.0)
        
        # Pure Pursuit parameters
        self.declare_parameter('wheelbase', 2.875)
        self.declare_parameter('min_lookahead', 4.0)
        self.declare_parameter('max_lookahead', 25.0)
        self.declare_parameter('lookahead_ratio', 0.5)
        
        self.controller_type = self.get_parameter('controller_type').value
        self.target_speed = self.get_parameter('target_speed').value
        mpc_horizon = self.get_parameter('mpc_horizon').value
        mpc_dt = self.get_parameter('mpc_dt').value
        trajectory_mode = self.get_parameter('trajectory_mode').value
        self.max_steering_diff = self.get_parameter('max_steering_diff').value
        control_rate = self.get_parameter('control_rate').value
        
        pid_kp = self.get_parameter('pid_kp').value
        pid_ki = self.get_parameter('pid_ki').value
        pid_kd = self.get_parameter('pid_kd').value
        self.max_throttle = self.get_parameter('max_throttle').value
        self.max_brake = self.get_parameter('max_brake').value
        self.max_steering = self.get_parameter('max_steering').value
        
        wheelbase = self.get_parameter('wheelbase').value
        min_lookahead = self.get_parameter('min_lookahead').value
        max_lookahead = self.get_parameter('max_lookahead').value
        lookahead_ratio = self.get_parameter('lookahead_ratio').value
        
        # ==================== Initialize Controllers ====================
        if self.controller_type == 'mpc':
            self.get_logger().info(f'Using MPC control (horizon={mpc_horizon}, mode={trajectory_mode})')
            self.mpc = MPCController(
                dt=mpc_dt, 
                horizon=mpc_horizon, 
                target_speed=self.target_speed,
                trajectory_mode=trajectory_mode
            )
        else:
            self.get_logger().info('Using Pure Pursuit + PID control')
            self.mpc = None
        
        # Pure Pursuit (for PID mode) - stores state internally
        self.pp_wheelbase = wheelbase
        self.pp_min_lookahead = min_lookahead
        self.pp_max_lookahead = max_lookahead
        self.pp_lookahead_ratio = lookahead_ratio
        
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
        
        self._last_solve_time_ms = 0.0
        
        self.get_logger().info(f'Control Node initialized ({self.controller_type}, {control_rate}Hz)')
    
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
        
        # Update MPC waypoints
        if self.mpc is not None:
            self.mpc.set_waypoints(waypoints)
    
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
        
        # Run controller
        if self.controller_type == 'mpc':
            steering, throttle, brake = self._run_mpc()
        else:
            steering, throttle, brake = self._run_pid()
        
        # Apply steering smoothing
        steering = self._smooth_steering(steering)
        
        # Publish control
        self._publish_control(steering, throttle, brake)
    
    def _run_mpc(self):
        """Run MPC control step."""
        import time
        t_start = time.time()
        
        current_state = np.array([
            self.vehicle_x, 
            self.vehicle_y, 
            self.vehicle_yaw, 
            self.vehicle_speed
        ])
        
        target_speed_ms = self.target_speed / 3.6
        optimal_control = self.mpc.solve(current_state, target_speed_ms)
        
        self._last_solve_time_ms = (time.time() - t_start) * 1000
        
        if optimal_control is None:
            self.get_logger().warn('MPC solve failed')
            return self.previous_steering, 0.0, 1.0
        
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
    
    def _run_pid(self):
        """Run Pure Pursuit + PID control step."""
        if not self.current_waypoints:
            return 0.0, 0.0, 1.0
        
        # Pure Pursuit for steering
        steering = self._pure_pursuit_steering()
        
        # PID for speed
        current_speed_kmh = self.vehicle_speed * 3.6
        error = self.target_speed - current_speed_kmh
        accel = self.pid.step(error)
        
        if accel >= 0:
            throttle = min(accel, self.max_throttle)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(abs(accel), self.max_brake)
        
        return float(steering), float(throttle), float(brake)
    
    def _pure_pursuit_steering(self):
        """Compute steering using Pure Pursuit algorithm."""
        if not self.current_waypoints:
            return 0.0
        
        # Compute adaptive lookahead
        lookahead = self.pp_min_lookahead + self.pp_lookahead_ratio * self.vehicle_speed
        lookahead = np.clip(lookahead, self.pp_min_lookahead, self.pp_max_lookahead)
        
        # Find lookahead point
        lookahead_point = None
        for wp in self.current_waypoints:
            dist = math.sqrt((wp[0] - self.vehicle_x)**2 + (wp[1] - self.vehicle_y)**2)
            if dist >= lookahead:
                lookahead_point = wp
                break
        
        if lookahead_point is None and self.current_waypoints:
            lookahead_point = self.current_waypoints[-1]
        
        if lookahead_point is None:
            return 0.0
        
        # Vector to lookahead point
        dx = lookahead_point[0] - self.vehicle_x
        dy = lookahead_point[1] - self.vehicle_y
        ld = math.sqrt(dx**2 + dy**2)
        
        if ld < self.pp_min_lookahead * 0.5:
            return 0.0
        
        ld = max(ld, self.pp_min_lookahead)
        
        # Transform to vehicle frame
        cos_yaw = math.cos(-self.vehicle_yaw)
        sin_yaw = math.sin(-self.vehicle_yaw)
        local_y = dx * sin_yaw + dy * cos_yaw
        
        # Pure pursuit formula
        curvature = 2.0 * local_y / (ld * ld)
        steering_angle = math.atan(self.pp_wheelbase * curvature)
        
        # Normalize
        max_steering_angle = math.radians(35.0)
        steering = steering_angle / max_steering_angle
        
        return np.clip(steering, -self.max_steering, self.max_steering)
    
    def _smooth_steering(self, steering):
        """Apply steering smoothing."""
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
            'controller_type': self.controller_type,
            'has_trajectory': self.has_trajectory,
            'num_waypoints': len(self.current_waypoints) if self.current_waypoints else 0,
            'vehicle_speed_kmh': self.vehicle_speed * 3.6 if self.vehicle_speed else 0.0,
            'target_speed_kmh': self.target_speed,
            'solve_time_ms': self._last_solve_time_ms
        }
        
        status_msg = String()
        status_msg.data = str(status)
        self.status_pub.publish(status_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
