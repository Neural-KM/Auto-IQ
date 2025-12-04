"""
Helper Utilities
================

Common helper functions.
"""

import math
import carla

from config import Settings


def get_speed(vehicle):
    """
    Get vehicle speed in km/h.
    
    Args:
        vehicle: CARLA vehicle actor
        
    Returns:
        Speed in km/h
    """
    velocity = vehicle.get_velocity()
    return 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)


def spawn_vehicle(world, spawn_point=None, vehicle_filter=None):
    """
    Spawn a vehicle in the world.
    
    Args:
        world: CARLA world
        spawn_point: Optional spawn transform
        vehicle_filter: Blueprint filter string
        
    Returns:
        Spawned vehicle actor
    """
    vehicle_filter = vehicle_filter or Settings.VEHICLE_FILTER
    
    blueprint_library = world.get_blueprint_library()
    blueprints = blueprint_library.filter(vehicle_filter)
    
    if not blueprints:
        # Fallback to any vehicle
        blueprints = blueprint_library.filter('vehicle.*')
    
    vehicle_bp = blueprints[0]
    
    if spawn_point is None:
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points available in map")
        spawn_point = spawn_points[0]
    
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    
    return vehicle


def get_vehicle_telemetry(vehicle, world_map):
    """
    Get comprehensive telemetry from vehicle.
    
    Args:
        vehicle: CARLA vehicle actor
        world_map: CARLA map
        
    Returns:
        Dictionary with telemetry data
    """
    velocity = vehicle.get_velocity()
    transform = vehicle.get_transform()
    control = vehicle.get_control()
    
    # Get current waypoint info
    current_wp = world_map.get_waypoint(transform.location)
    
    return {
        'speed_kmh': 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2),
        'location': {
            'x': transform.location.x,
            'y': transform.location.y,
            'z': transform.location.z
        },
        'rotation': {
            'pitch': transform.rotation.pitch,
            'yaw': transform.rotation.yaw,
            'roll': transform.rotation.roll
        },
        'lane_id': current_wp.lane_id,
        'road_id': current_wp.road_id,
        'is_junction': current_wp.is_junction,
        'control': control
    }
