"""
Vehicle Controller
==================

Unified vehicle control with MPC or PID modes (selectable via argument).
MPC: Full NMPC for trajectory tracking
PID: Pure Pursuit (lateral) + PID (longitudinal) for waypoint following
"""

import carla
import numpy as np
import math

from control.pure_pursuit_controller import PurePursuitController
from control.longitudinal_controller import LongitudinalController
from config.config import CARLA, CONTROL

# MPC (unified controller with integrated vehicle model and trajectory generation)
from control.mpc_controller import MPCController

class VehicleController:
    """
    Combined vehicle controller with MPC or PID modes.
    
    MPC: Smooth, predictive control with obstacle avoidance
    PID: Pure Pursuit lateral + PID longitudinal for waypoint following
    """
    
    def __init__(self, vehicle, controller_type='mpc', mpc_horizon=20, mpc_dt=0.05, trajectory_mode=0):
        """
        Initialize vehicle controller.
        
        Args:
            vehicle: CARLA vehicle actor
            controller_type: 'mpc' or 'pid'
            mpc_horizon: MPC prediction horizon steps
            mpc_dt: MPC timestep in seconds
            trajectory_mode: 0 for heuristic, 1 for trapezoidal (MPC only)
        """
        self.vehicle = vehicle
        self.controller_type = controller_type.lower()
        
        if self.controller_type == 'mpc':
            print(f"[VehicleController] Using MPC control (mode={trajectory_mode})")
            self.mpc = MPCController(dt=mpc_dt, horizon=mpc_horizon, 
                                   target_speed=CONTROL['planning']['target_speed'],
                                   trajectory_mode=trajectory_mode)
            self._mpc_horizon = mpc_horizon
            self._mpc_dt = mpc_dt
            
        elif self.controller_type == 'pid':
            print("[VehicleController] Using Pure Pursuit + PID control")
            self.lateral = PurePursuitController(vehicle)
            self.longitudinal = LongitudinalController(
                vehicle,
                k_p=CONTROL['longitudinal']['kp'],
                k_i=CONTROL['longitudinal']['ki'],
                k_d=CONTROL['longitudinal']['kd'],
                dt=CARLA['simulation']['fixed_delta_seconds']
            )
        else:
            raise ValueError(f"Unknown controller type: {controller_type}. Use 'mpc' or 'pid'")
        
        # Steering smoothing
        self.previous_steering = 0.0
        self.max_steering_diff = CONTROL['lateral']['max_steering_diff']
        
        # Previous control
        self._prev_control = np.array([0.0, 0.0])
        
        # Waypoints for trajectory generation
        self._waypoints = None
    
    # def update_obstacles(self, obstacles):
    #     """Update detected obstacles for the controller (for MPC or PID)."""
    #     self._obstacles = obstacles
    #     if self.controller_type == 'mpc':
    #         self.mpc.update_obstacles(obstacles)
    
    def set_waypoints(self, waypoints):
        """
        Set waypoints for trajectory generation.
        
        Args:
            waypoints: List of carla.Waypoint
        """
        # Convert to (x, y) tuples
        self._waypoints = [(wp.transform.location.x, wp.transform.location.y) for wp in waypoints] if waypoints else []
        
        if self.controller_type == 'mpc' and self._waypoints:
            success = self.mpc.set_waypoints(self._waypoints)
            if not success:
                print("[VehicleController] Failed to set waypoints for MPC")
        elif self.controller_type == 'pid' and self._waypoints:
            # Pass waypoints to Pure Pursuit controller
            self.lateral.set_waypoints(self._waypoints)
    
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
        
        # Solve MPC (reference trajectory generated internally)
        optimal_control = self.mpc.solve(current_state, target_speed_ms)
        
        if optimal_control is None:
            print("[VehicleController] MPC solve failed")
            return self.emergency_stop()
        
        # Convert MPC output to CARLA control
        steering_rad, accel = optimal_control
        
        # Convert steering angle to normalized steering [-1, 1]
        max_steer_rad = self.mpc.MAX_STEER
        steering = np.clip(steering_rad / max_steer_rad, -1.0, 1.0)
        
        # Convert acceleration to throttle/brake
        if accel >= 0:
            throttle = np.clip(accel / self.mpc.MAX_ACCEL, 0.0, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = np.clip(-accel / abs(self.mpc.MAX_DECEL), 0.0, 1.0)
        
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
            'predicted_trajectory': self.mpc.get_predicted_trajectory(),
            'planned_controls': self.mpc.get_planned_controls(),
            'reference_trajectory': self.mpc.get_reference_trajectory(),
            'solve_time_ms': self.mpc.get_last_solve_time_ms()
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
