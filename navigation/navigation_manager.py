"""
Navigation Manager
==================

Orchestrates autonomous navigation from point A to point B.
"""

import carla
from planning.global_planner import GlobalPlanner, RoadOption
from control import VehicleController
from config import Settings


class NavigationManager:
    """
    Manages autonomous navigation from origin to destination.
    """
    
    def __init__(self, vehicle, world):
        self.vehicle = vehicle
        self.world = world
        self.map = world.get_map()
        
        # Initialize global planner
        self.global_planner = GlobalPlanner(self.map, sampling_resolution=2.0)
        
        # Initialize controller
        self.controller = VehicleController(vehicle)
        
        # Navigation state
        self.target_speed = Settings.TARGET_SPEED
        self.destination = None
        self.is_navigating = False
        self._route_drawn = False
    
    def set_destination(self, destination, draw_route=True):
        """Set navigation destination and compute route."""
        if isinstance(destination, carla.Transform):
            destination = destination.location
        elif isinstance(destination, (tuple, list)):
            destination = carla.Location(x=destination[0], y=destination[1], 
                                         z=destination[2] if len(destination) > 2 else 0)
        
        self.destination = destination
        origin = self.vehicle.get_location()
        
        success = self.global_planner.set_destination(origin, destination)
        
        if success:
            self.is_navigating = True
            if draw_route:
                self._draw_route()
        else:
            self.is_navigating = False
        
        return success
    
    def _draw_route(self):
        """Draw route with short lifetime - needs to be refreshed each frame."""
        self.global_planner.draw_route(self.world, life_time=0.5)
        self._route_drawn = True
    
    def _draw_current_waypoint(self, waypoint):
        """Draw the current waypoint being followed (larger, red)."""
        if waypoint is None:
            return
        self.world.debug.draw_point(
            waypoint.transform.location + carla.Location(z=0.5),
            size=0.2,
            color=carla.Color(255, 0, 0),  # Red
            life_time=1
        )
    
    def _clear_route(self):
        """Clear the drawn route by not refreshing it."""
        self._route_drawn = False
    
    def set_destination_by_index(self, spawn_index, draw_route=True):
        """Set destination using spawn point index."""
        spawn_points = self.map.get_spawn_points()
        if spawn_index < 0 or spawn_index >= len(spawn_points):
            print(f"[Navigation] Invalid index. Valid: 0-{len(spawn_points)-1}")
            return False
        return self.set_destination(spawn_points[spawn_index].location, draw_route)
    
    def run_step(self):
        """Execute one navigation step. Returns control or None if complete."""
        if not self.is_navigating:
            return None
        
        vehicle_location = self.vehicle.get_location()
        
        # Refresh route visualization while navigating
        if self._route_drawn:
            self._draw_route()
        
        if self.global_planner.is_destination_reached(vehicle_location, threshold=5.0):
            print("[Navigation] Destination reached!")
            self.is_navigating = False
            self._clear_route()
            return self.controller.emergency_stop()
        
        self.global_planner.advance(vehicle_location)
        target_wp, road_option = self.global_planner.get_next_waypoint()
        
        if target_wp is None:
            self.is_navigating = False
            self._clear_route()
            return self.controller.emergency_stop()
        
        # Draw the current waypoint being followed
        self._draw_current_waypoint(target_wp)
        
        return self.controller.run_step(target_wp, self.target_speed)
    
    def get_status(self):
        """Get navigation status."""
        return {
            'is_navigating': self.is_navigating,
            'progress': self.global_planner.get_progress(),
            'remaining_distance': self.global_planner.get_remaining_distance(),
            'target_speed': self.target_speed,
            'road_option': self.global_planner.get_current_road_option().name
        }
    
    def set_target_speed(self, speed_kmh):
        self.target_speed = speed_kmh
    
    def stop(self):
        self.is_navigating = False
        self._clear_route()
        return self.controller.emergency_stop()
    
    def resume(self):
        if self.global_planner.get_route():
            self.is_navigating = True
            self._draw_route()
