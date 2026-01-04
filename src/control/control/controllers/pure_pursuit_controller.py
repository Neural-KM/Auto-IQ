"""
Pure Pursuit Controller
=======================

Lateral steering control using the Pure Pursuit algorithm.
Computes steering to follow a lookahead point on the path.

Reference: Implementation of the Pure Pursuit Path Tracking Algorithm
"""

import math
import numpy as np

# Default value (was from config.config CONTROL)
DEFAULT_MAX_STEERING = 1.0


class PurePursuitController:
    """
    Pure Pursuit lateral controller for path tracking.
    
    The Pure Pursuit algorithm computes the steering angle required to
    follow a circular arc that passes through the vehicle's rear axle
    and a lookahead point on the desired path.
    
    Steering angle: delta = atan(2 * L * sin(alpha) / Ld)
    
    Where:
        L   = Vehicle wheelbase
        Ld  = Lookahead distance
        alpha = Angle between vehicle heading and lookahead point
    """
    
    def __init__(self, vehicle, wheelbase=2.875, min_lookahead=4.0, max_lookahead=25.0,
                 lookahead_ratio=0.5, max_steering=DEFAULT_MAX_STEERING):
        """
        Initialize Pure Pursuit controller.
        
        Args:
            vehicle: CARLA vehicle actor
            wheelbase: Distance between front and rear axles (Tesla Model 3: 2.875m)
            min_lookahead: Minimum lookahead distance in meters
            max_lookahead: Maximum lookahead distance in meters
            lookahead_ratio: Ratio of speed to lookahead distance (Ld = ratio * v + min)
            max_steering: Maximum steering angle [-1, 1]
        """
        self.vehicle = vehicle
        self.wheelbase = wheelbase
        self.min_lookahead = min_lookahead
        self.max_lookahead = max_lookahead
        self.lookahead_ratio = lookahead_ratio
        
        # Max steering angle
        self.max_steering = max_steering
        
        # Waypoints for path tracking
        self._waypoints = None
        
        # Store current lookahead point for visualization
        self._current_lookahead_point = None
        self._current_lookahead_distance = 0.0
        
    def set_waypoints(self, waypoints):
        """
        Set waypoints for path tracking.
        
        Args:
            waypoints: List of carla.Waypoint objects
        """
        self._waypoints = waypoints
    
    def _compute_lookahead_distance(self, speed):
        """
        Compute adaptive lookahead distance based on speed.
        
        Args:
            speed: Current vehicle speed in m/s
            
        Returns:
            Lookahead distance in meters
        """
        # Adaptive lookahead: increases with speed for stability
        lookahead = self.min_lookahead + self.lookahead_ratio * speed
        return np.clip(lookahead, self.min_lookahead, self.max_lookahead)
    
    def _find_lookahead_point(self, vehicle_location, lookahead_distance):
        """
        Find the lookahead point on the path.
        
        Args:
            vehicle_location: Current vehicle position
            lookahead_distance: Target lookahead distance
            
        Returns:
            Lookahead point (x, y) or None if not found
        """
        if not self._waypoints or len(self._waypoints) == 0:
            return None
        
        # Find the first waypoint beyond lookahead distance
        vx, vy = vehicle_location.x, vehicle_location.y
        
        for wp in self._waypoints:
            wx, wy = wp[0], wp[1]
            dist = math.sqrt((wx - vx)**2 + (wy - vy)**2)
            
            if dist >= lookahead_distance:
                return (wx, wy)
        
        # If no waypoint is far enough, use the last waypoint
        if self._waypoints:
            last_wp = self._waypoints[-1]
            return (last_wp[0], last_wp[1])
        
        return None
    
    def _find_lookahead_interpolated(self, vehicle_location, lookahead_distance):
        """
        Find interpolated lookahead point on the path for smoother tracking.
        
        Args:
            vehicle_location: Current vehicle position
            lookahead_distance: Target lookahead distance
            
        Returns:
            Lookahead point (x, y) or None if not found
        """
        if not self._waypoints or len(self._waypoints) < 2:
            return self._find_lookahead_point(vehicle_location, lookahead_distance)
        
        vx, vy = vehicle_location.x, vehicle_location.y
        
        # Find the two waypoints that straddle the lookahead distance
        for i in range(len(self._waypoints) - 1):
            wp1_x, wp1_y = self._waypoints[i][0], self._waypoints[i][1]
            wp2_x, wp2_y = self._waypoints[i+1][0], self._waypoints[i+1][1]
            
            d1 = math.sqrt((wp1_x - vx)**2 + (wp1_y - vy)**2)
            d2 = math.sqrt((wp2_x - vx)**2 + (wp2_y - vy)**2)
            
            # Check if lookahead distance is between these two waypoints
            if d1 < lookahead_distance <= d2:
                # Interpolate between wp1 and wp2
                segment_length = math.sqrt((wp2_x - wp1_x)**2 + (wp2_y - wp1_y)**2)
                if segment_length < 0.001:
                    return (wp2_x, wp2_y)
                
                # How far along the segment to reach lookahead distance
                # Using linear approximation
                ratio = (lookahead_distance - d1) / (d2 - d1 + 0.001)
                ratio = np.clip(ratio, 0.0, 1.0)
                
                lx = wp1_x + ratio * (wp2_x - wp1_x)
                ly = wp1_y + ratio * (wp2_y - wp1_y)
                
                return (lx, ly)
        
        # Fallback to simple method
        return self._find_lookahead_point(vehicle_location, lookahead_distance)
    
    def run_step(self):
        """
        Compute steering using Pure Pursuit algorithm.
        
        Uses internal waypoints set via set_waypoints().

        Returns:
            Steering value in range [-1, 1]
        """
        if not self._waypoints or len(self._waypoints) == 0:
            return 0.0
            
        # Get vehicle state
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        velocity = self.vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2)  # m/s
        
        # Vehicle heading in radians
        yaw = math.radians(vehicle_transform.rotation.yaw)
        
        # Compute adaptive lookahead distance
        lookahead_distance = self._compute_lookahead_distance(speed)
        self._current_lookahead_distance = lookahead_distance
        
        # Find lookahead point from waypoints
        if len(self._waypoints) > 1:
            self._current_lookahead_point = self._find_lookahead_interpolated(vehicle_location, lookahead_distance)
        else:
            # Single waypoint - use it directly
            wp = self._waypoints[0]
            self._current_lookahead_point = (wp.transform.location.x, wp.transform.location.y)
        
        if self._current_lookahead_point is None:
            return 0.0
        
        # Compute vector from rear axle to lookahead point
        # Rear axle is approximately at vehicle location (CARLA uses rear axle as reference)
        dx = self._current_lookahead_point[0] - vehicle_location.x
        dy = self._current_lookahead_point[1] - vehicle_location.y
        
        # Actual distance to lookahead point
        ld = math.sqrt(dx**2 + dy**2)
        
        # CRITICAL: If lookahead point is too close, don't steer aggressively
        # This prevents jerky behavior when waypoints run out
        if ld < self.min_lookahead * 0.5:
            # Point is too close - use gentle steering based on heading error only
            return 0.0
        
        # Enforce minimum effective lookahead distance for stability
        ld = max(ld, self.min_lookahead)
        
        # Transform to vehicle coordinate frame
        # Rotate by -yaw to get coordinates in vehicle frame
        cos_yaw = math.cos(-yaw)
        sin_yaw = math.sin(-yaw)
        
        # Lookahead point in vehicle frame
        local_x = dx * cos_yaw - dy * sin_yaw  # Forward
        local_y = dx * sin_yaw + dy * cos_yaw  # Left (positive)
        
        # Pure pursuit formula:
        # curvature = 2 * y / Ld^2
        # steering_angle = atan(curvature * L) = atan(2 * L * y / Ld^2)
        curvature = 2.0 * local_y / (ld * ld)
        steering_angle = math.atan(self.wheelbase * curvature)
        
        # Normalize steering to [-1, 1]
        # Assuming max steering angle is ~35 degrees (0.61 rad) for Tesla Model 3
        max_steering_angle = math.radians(35.0)
        steering = steering_angle / max_steering_angle
        
        return np.clip(steering, -self.max_steering, self.max_steering)
    
    def get_lookahead_point(self):
        """
        Get the current lookahead point for visualization.
        
        Returns:
            Tuple (x, y) of lookahead point in world coordinates, or None
        """
        return self._current_lookahead_point
    
    def get_current_lookahead_distance(self):
        """
        Get the current lookahead distance being used.
        
        Returns:
            Lookahead distance in meters
        """
        return self._current_lookahead_distance
    
    def get_lookahead_info(self):
        """
        Get debug information about current lookahead state.
        
        Returns:
            Dictionary with lookahead info
        """
        velocity = self.vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2)
        ld = self._compute_lookahead_distance(speed)
        
        return {
            'speed_ms': speed,
            'lookahead_distance': ld,
            'num_waypoints': len(self._waypoints) if self._waypoints else 0
        }
    
    def reset(self):
        """Reset controller state."""
        self._waypoints = None