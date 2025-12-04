"""
Lateral Controller
==================

Steering control using PID to follow waypoints.
"""

import math
import numpy as np

from control.pid_controller import PIDController
from config import Settings


class LateralController:
    """
    Lateral controller for steering using PID.
    Steers the vehicle towards target waypoints.
    """
    
    def __init__(self, vehicle, k_p=None, k_i=None, k_d=None, dt=None):
        """
        Initialize lateral controller.
        
        Args:
            vehicle: CARLA vehicle actor
            k_p, k_i, k_d: PID gains (uses config defaults if None)
            dt: Time step
        """
        self.vehicle = vehicle
        
        # Use config defaults if not specified
        k_p = k_p if k_p is not None else Settings.LATERAL_KP
        k_i = k_i if k_i is not None else Settings.LATERAL_KI
        k_d = k_d if k_d is not None else Settings.LATERAL_KD
        dt = dt if dt is not None else Settings.FIXED_DELTA_SECONDS
        
        self.pid = PIDController(k_p, k_i, k_d, dt)
        self.max_steering = Settings.MAX_STEERING
    
    def run_step(self, waypoint):
        """
        Compute steering control to reach target waypoint.
        
        Args:
            waypoint: Target carla.Waypoint
            
        Returns:
            Steering value in range [-1, 1]
        """
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        
        # Get vehicle forward vector
        forward_vec = vehicle_transform.get_forward_vector()
        forward_vec = np.array([forward_vec.x, forward_vec.y, 0.0])
        
        # Get vector from vehicle to waypoint
        wp_location = waypoint.transform.location
        target_vec = np.array([
            wp_location.x - vehicle_location.x,
            wp_location.y - vehicle_location.y,
            0.0
        ])
        
        # Compute angle between vectors
        norm_product = np.linalg.norm(forward_vec) * np.linalg.norm(target_vec)
        if norm_product < 0.001:
            return 0.0
        
        dot_product = np.dot(forward_vec, target_vec)
        angle = math.acos(np.clip(dot_product / norm_product, -1.0, 1.0))
        
        # Determine direction using cross product
        cross_product = np.cross(forward_vec, target_vec)
        if cross_product[2] < 0:
            angle *= -1.0
        
        # Compute steering using PID
        steering = self.pid.step(angle)
        
        return np.clip(steering, -self.max_steering, self.max_steering)
    
    def run_step_from_error(self, cross_track_error):
        """
        Compute steering from cross-track error (for perception-based control).
        
        Args:
            cross_track_error: Lateral offset from lane center (positive = right)
            
        Returns:
            Steering value in range [-1, 1]
        """
        steering = self.pid.step(cross_track_error)
        return np.clip(steering, -self.max_steering, self.max_steering)
    
    def reset(self):
        """Reset controller state."""
        self.pid.reset()
