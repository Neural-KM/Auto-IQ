"""
Vehicle Controller
==================

Unified vehicle control with MPC or PID modes (selectable via argument).
MPC: Full NMPC for trajectory tracking with obstacle avoidance
PID: Pure Pursuit (lateral) + PID (longitudinal) for waypoint following
"""

import carla
import numpy as np
import math

from control.pure_pursuit_controller import PurePursuitController
from control.longitudinal_controller import LongitudinalController
from config.settings import Settings

# MPC components
from control.mpc_controller import MPCController
from control.trajectory_generator import TrajectoryGenerator
from control.vehicle_model import VehicleModel

class VehicleController:
    """
    Combined vehicle controller with MPC or PID modes.
    
    MPC: Smooth, predictive control with obstacle avoidance
    PID: Pure Pursuit lateral + PID longitudinal for waypoint following
    """
    
    def __init__(self, vehicle, controller_type='mpc', mpc_horizon=20, mpc_dt=0.05):
        """
        Initialize vehicle controller.
        
        Args:
            vehicle: CARLA vehicle actor
            controller_type: 'mpc' or 'pid'
            mpc_horizon: MPC prediction horizon steps
            mpc_dt: MPC timestep in seconds
        """
        self.vehicle = vehicle
        self.controller_type = controller_type.lower()
        
        if self.controller_type == 'mpc':
            raise NotImplementedError("MPC controller is currently disabled.")
            print("[VehicleController] Using MPC control (path tracking, no obstacles)")
            self.mpc = MPCController(dt=mpc_dt, horizon=mpc_horizon)
            self.trajectory_gen = TrajectoryGenerator(target_speed=Settings.TARGET_SPEED)
            self.model = VehicleModel()
            self._mpc_horizon = mpc_horizon
            self._mpc_dt = mpc_dt
            
        elif self.controller_type == 'pid':
            print("[VehicleController] Using Pure Pursuit + PID control")
            self.lateral = PurePursuitController(vehicle)
            self.longitudinal = LongitudinalController(
                vehicle,
                k_p=Settings.LONGITUDINAL_KP,
                k_i=Settings.LONGITUDINAL_KI,
                k_d=Settings.LONGITUDINAL_KD,
                dt=Settings.FIXED_DELTA_SECONDS
            )
        else:
            raise ValueError(f"Unknown controller type: {controller_type}. Use 'mpc' or 'pid'")
        
        # Steering smoothing
        self.previous_steering = 0.0
        self.max_steering_diff = Settings.MAX_STEERING_DIFF
        
        # Previous control
        self._prev_control = np.array([0.0, 0.0])
        
        # Obstacles for MPC
        self._obstacles = []
        
        # Waypoints for trajectory generation
        self._waypoints = None
    
    def set_waypoints(self, waypoints):
        """
        Set waypoints for trajectory generation.
        
        Args:
            waypoints: List of carla.Waypoint or (x, y) tuples
        """
        self._waypoints = waypoints
        
        if self.controller_type == 'mpc' and waypoints:
            success = self.trajectory_gen.set_waypoints(waypoints)
            if not success:
                print("[VehicleController] Failed to set waypoints for MPC")
        elif self.controller_type == 'pid' and waypoints:
            # Pass waypoints to Pure Pursuit controller
            self.lateral.set_waypoints(waypoints)
    
    def set_obstacles(self, obstacles):
        """
        Set obstacles for MPC avoidance.
        
        Args:
            obstacles: List of (x, y, radius) tuples
        """
        self._obstacles = obstacles if obstacles else []
    
    def get_current_state(self):
        """
        Get current vehicle state [x, y, psi, v].
        
        Returns:
            numpy array [x, y, psi, v]
        """
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        
        x = transform.location.x
        y = transform.location.y
        psi = math.radians(transform.rotation.yaw)
        v = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        return np.array([x, y, psi, v])
    
    def run_step(self, target_speed):
        """
        Compute vehicle control for waypoint following.
        
        Uses internal waypoints set via set_waypoints().
        
        Args:
            target_speed: Target speed in km/h
            
        Returns:
            carla.VehicleControl object
        """
        if not self._waypoints or len(self._waypoints) == 0:
            print("[VehicleController] No waypoints set")
            return self.emergency_stop()
            
        if self.controller_type == 'mpc':
            return self._run_step_mpc(target_speed)
        else:
            return self._run_step_pid(target_speed)
    
    def _run_step_mpc(self, target_speed):
        """Run MPC control step."""
        # Get current state
        current_state = self.get_current_state()
        
        # Convert target speed to m/s
        target_speed_ms = target_speed / 3.6
        
        # Generate reference trajectory from waypoints
        if len(self._waypoints) > 3:
            # Use spline-based trajectory from waypoints
            reference = self.trajectory_gen.generate_reference(
                current_state, 
                self._mpc_horizon, 
                self._mpc_dt,
                target_speed_ms
            )
        else:
            # Generate simple reference from first waypoint
            reference = self._generate_simple_reference(
                current_state, self._waypoints[0], target_speed_ms
            )
        
        if reference is None:
            print("[VehicleController] No reference trajectory available")
            return self.emergency_stop()
        
        # Solve MPC
        optimal_control = self.mpc.solve(current_state, reference, self._obstacles)
        
        if optimal_control is None:
            print("[VehicleController] MPC solve failed")
            return self.emergency_stop()
        
        # Convert MPC output to CARLA control
        steering_rad, accel = optimal_control
        
        # Convert steering angle to normalized steering [-1, 1]
        max_steer_rad = self.model.MAX_STEER
        steering = np.clip(steering_rad / max_steer_rad, -1.0, 1.0)
        
        # Convert acceleration to throttle/brake
        if accel >= 0:
            throttle = np.clip(accel / self.model.MAX_ACCEL, 0.0, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = np.clip(-accel / abs(self.model.MAX_DECEL), 0.0, 1.0)
        
        # Create control command
        control = carla.VehicleControl()
        control.steer = float(steering)
        control.throttle = float(throttle)
        control.brake = float(brake)
        control.hand_brake = False
        control.manual_gear_shift = False
        
        # Save for debugging
        self._prev_control = optimal_control
        self.previous_steering = steering
        
        return control
    
    def _generate_simple_reference(self, current_state, target_waypoint, target_speed_ms):
        """Generate simple reference trajectory from single waypoint."""
        if target_waypoint is None:
            return None
        
        # Get target position
        if hasattr(target_waypoint, 'transform'):
            target_x = target_waypoint.transform.location.x
            target_y = target_waypoint.transform.location.y
        else:
            return None
        
        # Generate trajectory towards target
        reference = np.zeros((self._mpc_horizon + 1, 4))
        
        x, y, psi, v = current_state
        
        # Direction to target
        dx = target_x - x
        dy = target_y - y
        dist = math.sqrt(dx**2 + dy**2)
        
        if dist < 0.1:
            target_psi = psi
        else:
            target_psi = math.atan2(dy, dx)
        
        # Interpolate towards target
        for i in range(self._mpc_horizon + 1):
            t = i * self._mpc_dt
            progress = min(1.0, (target_speed_ms * t) / (dist + 0.1))
            
            ref_x = x + progress * dx
            ref_y = y + progress * dy
            ref_psi = target_psi
            ref_v = target_speed_ms
            
            reference[i] = [ref_x, ref_y, ref_psi, ref_v]
        
        return reference
    
    def _run_step_pid(self, target_speed):
        """Run PID control step."""
        # Pass waypoints to Pure Pursuit
        self.lateral.set_waypoints(self._waypoints)
        
        # Compute steering using Pure Pursuit (uses internal waypoints)
        raw_steering = self.lateral.run_step()
        steering = self.smooth_steering(raw_steering)
        
        # Compute throttle/brake
        throttle, brake = self.longitudinal.run_step(target_speed)
        
        # Create control command
        control = carla.VehicleControl()
        control.steer = float(steering)
        control.throttle = float(throttle)
        control.brake = float(brake)
        control.hand_brake = False
        control.manual_gear_shift = False
        
        return control
    
    def smooth_steering(self, steering):
        """Apply steering smoothing to prevent jerky movements."""
        if steering > self.previous_steering + self.max_steering_diff:
            steering = self.previous_steering + self.max_steering_diff
        elif steering < self.previous_steering - self.max_steering_diff:
            steering = self.previous_steering - self.max_steering_diff
        
        self.previous_steering = steering
        return steering
    
    def emergency_stop(self):
        """Return control for emergency stop."""
        control = carla.VehicleControl()
        control.steer = float(self.previous_steering)
        control.throttle = 0.0
        control.brake = 1.0
        control.hand_brake = False
        control.manual_gear_shift = False
        
        return control
    
    def get_speed(self):
        """Get current vehicle speed in km/h."""
        velocity = self.vehicle.get_velocity()
        speed_ms = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        return speed_ms * 3.6
    
    def reset(self):
        """Reset controller state."""
        self.previous_steering = 0.0
        self._prev_control = np.array([0.0, 0.0])
        self._obstacles = []
        
        if self.controller_type == 'pid':
            self.lateral.reset()
            self.longitudinal.reset()
        elif self.controller_type == 'mpc':
            self.mpc.reset()
    
    def get_mpc_info(self):
        """Get MPC debug info."""
        if self.controller_type != 'mpc':
            return None
        
        return {
            'solve_time_ms': self.mpc.get_avg_solve_time(),
            'predicted_trajectory': self.mpc.get_predicted_trajectory(),
            'planned_controls': self.mpc.get_planned_controls(),
            'num_obstacles': len(self._obstacles)
        }
    
    def get_pure_pursuit_info(self):
        """Get Pure Pursuit debug info."""
        if self.controller_type != 'pid':
            return None
        
        return {
            'lookahead_point': self.lateral.get_lookahead_point(),
            'lookahead_distance': self.lateral.get_current_lookahead_distance()
        }
    
    def is_using_mpc(self):
        """Check if MPC is active."""
        return self.controller_type == 'mpc'
