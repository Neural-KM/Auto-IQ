"""
Longitudinal Controller
=======================

Speed control using PID for throttle and brake.
"""

import math
import numpy as np

from control.pid_controller import PIDController
from config import Settings


class LongitudinalController:
    """
    Longitudinal controller for speed control using PID.
    Controls throttle and brake to maintain target speed.
    """
    
    def __init__(self, vehicle, k_p=None, k_i=None, k_d=None, dt=None):
        """
        Initialize longitudinal controller.
        
        Args:
            vehicle: CARLA vehicle actor
            k_p, k_i, k_d: PID gains (uses config defaults if None)
            dt: Time step
        """
        self.vehicle = vehicle
        
        # Use config defaults if not specified
        k_p = k_p if k_p is not None else Settings.LONGITUDINAL_KP
        k_i = k_i if k_i is not None else Settings.LONGITUDINAL_KI
        k_d = k_d if k_d is not None else Settings.LONGITUDINAL_KD
        dt = dt if dt is not None else Settings.FIXED_DELTA_SECONDS
        
        self.pid = PIDController(k_p, k_i, k_d, dt)
        self.max_throttle = Settings.MAX_THROTTLE
        self.max_brake = Settings.MAX_BRAKE
    
    def get_speed(self):
        """Get current vehicle speed in km/h."""
        velocity = self.vehicle.get_velocity()
        return 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    
    def run_step(self, target_speed):
        """
        Compute throttle/brake to reach target speed.
        
        Args:
            target_speed: Target speed in km/h
            
        Returns:
            Tuple of (throttle, brake) values in range [0, 1]
        """
        current_speed = self.get_speed()
        error = target_speed - current_speed
        
        acceleration = self.pid.step(error)
        
        if acceleration >= 0.0:
            throttle = min(acceleration, self.max_throttle)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(abs(acceleration), self.max_brake)
        
        return throttle, brake
    
    def emergency_stop(self):
        """Return control for emergency stop."""
        self.pid.reset()
        return 0.0, self.max_brake
    
    def reset(self):
        """Reset controller state."""
        self.pid.reset()
