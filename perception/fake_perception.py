"""
Fake Perception Module
======================

Mimics a perception system by retrieving ground truth data from CARLA
and filtering for obstacles within a certain radius of the ego vehicle.
"""

import carla
import math
from visualization.bounding_box_visualizer import BoundingBoxVisualizer


class FakePerception(BoundingBoxVisualizer):
    """
    Simulates a perfect perception system using ground truth data.
    Inherits from BoundingBoxVisualizer for debug visualization capabilities.
    """
    
    def __init__(self, vehicle, world, debug=False):
        """
        Initialize fake perception.
        
        Args:
            vehicle: The ego vehicle actor.
            world: The CARLA world instance.
            debug: Whether to draw debug visualizations.
        """
        super().__init__(world, ego_vehicle=vehicle)
        self.vehicle = vehicle
        self.debug_mode = debug
        
    def detect_obstacles(self, radius=50.0):
        """
        Detect obstacles within a given radius.
        
        Args:
            radius: Detection radius in meters.
            
        Returns:
            List of dictionaries containing obstacle details:
            {
                'id': actor_id,
                'type': 'vehicle' or 'pedestrian',
                'type_id': blueprint_id,
                'location': carla.Location,
                'distance': float (meters),
                'velocity': carla.Vector3D,
                'speed': float (m/s),
                'bounding_box': {
                    'relative_location': carla.Location, # Center of BB relative to ego
                    'extent': carla.Vector3D,            # Half-size (x, y, z)
                    'dimensions': carla.Vector3D         # Full size (length, width, height)
                }
            }
        """
        if not self.vehicle or not self.vehicle.is_alive:
            return []
            
        ego_transform = self.vehicle.get_transform()
        ego_location = ego_transform.location
        obstacles = []
        
        # Get all relevant actors
        actor_list = self.world.get_actors()
        vehicles = actor_list.filter('vehicle.*')
        pedestrians = actor_list.filter('walker.pedestrian.*')
        
        all_actors = list(vehicles) + list(pedestrians)
        
        for actor in all_actors:
            # Skip ego vehicle
            if actor.id == self.vehicle.id:
                continue
                
            actor_location = actor.get_location()
            distance = ego_location.distance(actor_location)
            
            if distance <= radius:
                velocity = actor.get_velocity()
                speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                
                obstacle_type = 'vehicle' if 'vehicle' in actor.type_id else 'pedestrian'
                
                # Bounding box calculations
                bbox = actor.bounding_box
                
                # 1. Get bbox center in world coordinates
                # bbox.location is relative to actor center. 
                # We need to transform it to world space using actor's transform
                actor_transform = actor.get_transform()
                bbox_center_world = actor_transform.transform(bbox.location)
                
                # 2. Transform world coordinate to ego vehicle local coordinate
                bbox_center_relative = ego_transform.inverse_transform(bbox_center_world)
                
                # Dimensions
                dimensions = carla.Vector3D(
                    x=bbox.extent.x * 2,
                    y=bbox.extent.y * 2,
                    z=bbox.extent.z * 2
                )
                
                obstacle_info = {
                    'id': actor.id,
                    'type': obstacle_type,
                    'type_id': actor.type_id,
                    'location': actor_location,
                    'distance': distance,
                    'velocity': velocity,
                    'speed': speed,
                    'bounding_box': {
                        'relative_location': bbox_center_relative,
                        'extent': bbox.extent,
                        'dimensions': dimensions
                    }
                }
                obstacles.append(obstacle_info)
                
                # Draw debug bounding box if enabled
                if self.debug_mode:
                    self.draw_bounding_box(
                        relative_location=bbox_center_relative,
                        dimensions=dimensions,
                        yaw=actor_transform.rotation.yaw - ego_transform.rotation.yaw,
                        color=carla.Color(255, 0, 0) if obstacle_type == 'vehicle' else carla.Color(0, 255, 0),
                        life_time=0.5,
                        thickness=0.1
                    )
                
        # Sort by distance
        obstacles.sort(key=lambda x: x['distance'])
        
        return obstacles
