"""
Local Planner
=============

Waypoint-based local path planning.
"""

import math
import numpy as np
import carla
from collections import deque
from typing import Optional
from config.config import CONTROL
from state.lane import LaneDetection


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
        self.lookahead_distance = CONTROL['planning']['lookahead_distance']
        self.waypoint_spacing = CONTROL['planning']['waypoint_spacing']
    
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
    
    def generate_waypoints_from_lane_fits(self, lane_detection: Optional[LaneDetection]):
        """
        Generate waypoints from lane detection polynomial fits.
        
        Args:
            lane_detection: LaneDetection dataclass from state module
            
        Returns:
            List of carla.Waypoint-like objects representing the lane center path
        """
        if not lane_detection or not lane_detection.valid:
            return []
            
        left_fit = lane_detection.left_fit
        right_fit = lane_detection.right_fit
        bev_params = lane_detection.bev_params
        lane_width_bev = lane_detection.lane_width_bev
        
        if not bev_params:
            return []
            
        width = bev_params['width']
        height = bev_params['height']
        x_min, x_max = bev_params['x_range']
        y_min, y_max = bev_params['y_range']
        
        # Generate points
        waypoints = []
        
        # Sample y from bottom (near) to top (far)
        # Skip the very bottom to avoid noise
        num_points = 10
        y_pixels = np.linspace(height - 10, 0, num_points)
        
        vehicle_transform = self.vehicle.get_transform()
        
        # Helper class for waypoints
        class PerceptionWaypoint:
            def __init__(self, location):
                self.transform = carla.Transform(location)
        
        for y_pix in y_pixels:
            # Calculate x center in pixels
            x_left = np.polyval(left_fit, y_pix) if left_fit is not None else None
            x_right = np.polyval(right_fit, y_pix) if right_fit is not None else None
            
            if x_left is not None and x_right is not None:
                x_center_pix = (x_left + x_right) / 2.0
            elif x_left is not None:
                # Use detected lane width or default
                if lane_width_bev is not None:
                    lw = lane_width_bev
                else:
                    # Default 3.5m converted to pixels
                    meters_per_pixel_x = (x_max - x_min) / width
                    lw = 3.5 / meters_per_pixel_x
                x_center_pix = x_left + lw / 2.0
            elif x_right is not None:
                if lane_width_bev is not None:
                    lw = lane_width_bev
                else:
                    meters_per_pixel_x = (x_max - x_min) / width
                    lw = 3.5 / meters_per_pixel_x
                x_center_pix = x_right - lw / 2.0
            else:
                continue
                
            # Convert to vehicle coordinates (LaneDetector frame: X=Right, Y=Forward)
            # x_pix 0 -> x_min, width -> x_max
            x_ld = x_min + (x_center_pix / width) * (x_max - x_min)
            
            # y_pix 0 -> y_max, height -> y_min
            # Note: y_pix=0 is top (Far/Max), y_pix=height is bottom (Near/Min)
            y_ld = y_max - (y_pix / height) * (y_max - y_min)
            
            # Convert to CARLA vehicle frame (X=Forward, Y=Right)
            x_carla = y_ld
            y_carla = x_ld
            
            # Transform to world
            loc_local = carla.Location(x=x_carla, y=y_carla, z=0.0)
            loc_world = vehicle_transform.transform(loc_local)
            
            waypoints.append(PerceptionWaypoint(loc_world))
            
        return waypoints

    def get_lane_context(self):
        """
        Get information about all lanes on the current road section.
        
        Returns:
            dict: {
                'current_lane_id': int,
                'left_lanes': int,  # Number of lanes to the left
                'right_lanes': int, # Number of lanes to the right
                'total_lanes': int,
                'lane_ids': list    # List of all lane IDs from left to right
            }
        """
        current_wp = self.get_current_waypoint()
        current_lane_id = current_wp.lane_id
        
        # Check left
        left_lanes = 0
        left_ids = []
        wp = current_wp
        while True:
            left_wp = wp.get_left_lane()
            # Check if left lane exists, is driving type, and is in same direction (same sign)
            if (left_wp and 
                left_wp.lane_type == carla.LaneType.Driving and 
                left_wp.lane_id * current_lane_id > 0):
                
                left_lanes += 1
                left_ids.append(left_wp.lane_id)
                wp = left_wp
            else:
                break
                
        # Check right
        right_lanes = 0
        right_ids = []
        wp = current_wp
        while True:
            right_wp = wp.get_right_lane()
            # Check if right lane exists, is driving type, and is in same direction (same sign)
            if (right_wp and 
                right_wp.lane_type == carla.LaneType.Driving and 
                right_wp.lane_id * current_lane_id > 0):
                
                right_lanes += 1
                right_ids.append(right_wp.lane_id)
                wp = right_wp
            else:
                break
                
        return {
            'current_lane_id': current_lane_id,
            'left_lanes': left_lanes,
            'right_lanes': right_lanes,
            'total_lanes': 1 + left_lanes + right_lanes,
            'left_lane_ids': left_ids,
            'right_lane_ids': right_ids
        }

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
