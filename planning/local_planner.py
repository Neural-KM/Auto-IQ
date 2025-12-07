"""
Local Planner
=============

Waypoint-based local path planning.
"""

from collections import deque
from config import Settings


class LocalPlanner:
    """
    Local planner for waypoint following.
    
    Manages a queue of waypoints and provides target waypoints
    for the vehicle controller.
    """
    
    def __init__(self, vehicle, world_map):
        """
        Initialize local planner.
        
        Args:
            vehicle: CARLA vehicle actor
            world_map: CARLA map instance
        """
        self.vehicle = vehicle
        self.map = world_map
        
        self.waypoint_queue = deque(maxlen=100)
        self.lookahead_distance = Settings.LOOKAHEAD_DISTANCE
        self.waypoint_spacing = Settings.WAYPOINT_SPACING
    
    def get_current_waypoint(self):
        """Get waypoint at vehicle's current location."""
        location = self.vehicle.get_location()
        return self.map.get_waypoint(location)
    
    def get_next_waypoints(self, current_wp, count=20):
        """
        Get waypoints ahead of current position.
        
        Args:
            current_wp: Current waypoint
            count: Number of waypoints to retrieve
            
        Returns:
            List of waypoints
        """
        waypoints = [current_wp]
        
        for _ in range(count):
            next_wps = waypoints[-1].next(self.waypoint_spacing)
            if next_wps:
                waypoints.append(next_wps[0])
            else:
                break
        
        return waypoints
    
    def get_target_waypoint(self):
        """
        Get target waypoint at lookahead distance.
        
        Returns:
            Target waypoint for steering
        """
        current_wp = self.get_current_waypoint()
        vehicle_location = self.vehicle.get_location()
        
        waypoints = self.get_next_waypoints(current_wp)
        
        # Find waypoint at lookahead distance
        for wp in waypoints:
            if vehicle_location.distance(wp.transform.location) >= self.lookahead_distance:
                return wp
        
        return waypoints[-1] if waypoints else current_wp
    
    def get_left_lane_waypoint(self):
        """Get waypoint in left lane."""
        current_wp = self.get_current_waypoint()
        left_wp = current_wp.get_left_lane()
        
        if left_wp and left_wp.lane_type.name == 'Driving':
            return left_wp
        return None
    
    def get_right_lane_waypoint(self):
        """Get waypoint in right lane."""
        current_wp = self.get_current_waypoint()
        right_wp = current_wp.get_right_lane()
        
        if right_wp and right_wp.lane_type.name == 'Driving':
            return right_wp
        return None
    
    def can_change_lane(self, direction):
        """
        Check if lane change is possible.
        
        Args:
            direction: 'left' or 'right'
            
        Returns:
            Boolean indicating if lane change is safe
        """
        if direction == 'left':
            target_wp = self.get_left_lane_waypoint()
        else:
            target_wp = self.get_right_lane_waypoint()
        
        return target_wp is not None
    
    def generate_lane_change_waypoints(self, direction, distance=20.0):
        """
        Generate waypoints for lane change maneuver.
        
        Args:
            direction: 'left' or 'right'
            distance: Distance over which to complete lane change
            
        Returns:
            List of waypoints for lane change, or None if not possible
        """
        current_wp = self.get_current_waypoint()
        
        # Get target lane
        if direction == 'left':
            target_lane_wp = current_wp.get_left_lane()
        else:
            target_lane_wp = current_wp.get_right_lane()
        
        if target_lane_wp is None:
            return None
        
        # Generate waypoints transitioning to new lane
        waypoints = []
        
        # Start in current lane - approach phase (3 waypoints)
        wp = current_wp
        for _ in range(3):
            next_wps = wp.next(self.waypoint_spacing)
            if next_wps:
                waypoints.append(next_wps[0])
                wp = next_wps[0]
        
        # Transition point - get adjacent lane waypoint
        if direction == 'left':
            transition_wp = wp.get_left_lane()
        else:
            transition_wp = wp.get_right_lane()
        
        if transition_wp:
            waypoints.append(transition_wp)
            
            # Continue in new lane - EXTENDED to clear obstacle
            # Generate enough waypoints to travel ~30-40m past the obstacle
            wp = transition_wp
            for _ in range(25):  # Was 10, now 25 waypoints (~50m at 2m spacing)
                next_wps = wp.next(self.waypoint_spacing)
                if next_wps:
                    waypoints.append(next_wps[0])
                    wp = next_wps[0]
        
        return waypoints
