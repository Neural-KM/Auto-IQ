"""
Vehicle Controller
==================

Combines lateral and longitudinal control into unified vehicle control.
"""

import carla
import numpy as np

from control.lateral_controller import LateralController
from control.longitudinal_controller import LongitudinalController
from config import Settings


class VehicleController:
    """
    Combined vehicle controller.
    Manages both steering and speed control.
    """
    
    def __init__(self, vehicle):
        """
        Initialize vehicle controller.
        
        Args:
            vehicle: CARLA vehicle actor
        """
        self.vehicle = vehicle
        
        self.lateral = LateralController(vehicle)
        self.longitudinal = LongitudinalController(vehicle)
        
        # Steering smoothing
        self.previous_steering = 0.0
        self.max_steering_diff = Settings.MAX_STEERING_DIFF
    
    def smooth_steering(self, steering):
        """
        Apply steering smoothing to prevent jerky movements.
        
        Args:
            steering: Raw steering value
            
        Returns:
            Smoothed steering value
        """
        if steering > self.previous_steering + self.max_steering_diff:
            steering = self.previous_steering + self.max_steering_diff
        elif steering < self.previous_steering - self.max_steering_diff:
            steering = self.previous_steering - self.max_steering_diff
        
        self.previous_steering = steering
        return steering
    
    def run_step(self, target_waypoint, target_speed):
        """
        Compute vehicle control for waypoint following.
        
        Args:
            target_waypoint: Target carla.Waypoint
            target_speed: Target speed in km/h
            
        Returns:
            carla.VehicleControl object
        """
        # Compute steering
        raw_steering = self.lateral.run_step(target_waypoint)
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
    
    def run_step_from_error(self, cross_track_error, target_speed):
        """
        Compute vehicle control from perception error.
        
        Args:
            cross_track_error: Lateral offset from lane center
            target_speed: Target speed in km/h
            
        Returns:
            carla.VehicleControl object
        """
        # Compute steering from error
        raw_steering = self.lateral.run_step_from_error(cross_track_error)
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
    
    def emergency_stop(self):
        """Return control for emergency stop."""
        throttle, brake = self.longitudinal.emergency_stop()
        
        control = carla.VehicleControl()
        control.steer = float(self.previous_steering)  # Keep current steering
        control.throttle = float(throttle)
        control.brake = float(brake)
        control.hand_brake = False
        control.manual_gear_shift = False
        
        return control
    
    def get_speed(self):
        """Get current vehicle speed in km/h."""
        return self.longitudinal.get_speed()
    
    def reset(self):
        """Reset controller state."""
        self.lateral.reset()
        self.longitudinal.reset()
        self.previous_steering = 0.0
