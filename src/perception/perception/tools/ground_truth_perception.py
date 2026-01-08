"""Fake Perception - Ground truth obstacle detection from CARLA."""

import math
from typing import List

from perception.data_classes.obstacle import Obstacle


class GroundTruthPerception:
    """Ground truth perception using CARLA world state."""
    
    def __init__(self, vehicle, world):
        self.vehicle = vehicle
        self.world = world
        
    def detect_obstacles(self, radius: float = 50.0) -> List[Obstacle]:
        """
        Detect obstacles within a given radius.
        
        Args:
            radius: Detection radius in meters.
            
        Returns:
            List of Obstacle dataclass instances, sorted by distance.
        """
        if not self.vehicle or not self.vehicle.is_alive:
            return []
            
        ego_transform = self.vehicle.get_transform()
        ego_location = ego_transform.location
        obstacles: List[Obstacle] = []
        
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
                
                # Transform velocity to ego-relative frame
                ego_forward = ego_transform.get_forward_vector()
                ego_right = ego_transform.get_right_vector()
                ego_up = ego_transform.get_up_vector()
                
                vx_rel = velocity.x * ego_forward.x + velocity.y * ego_forward.y + velocity.z * ego_forward.z
                vy_rel = velocity.x * ego_right.x + velocity.y * ego_right.y + velocity.z * ego_right.z
                vz_rel = velocity.x * ego_up.x + velocity.y * ego_up.y + velocity.z * ego_up.z
                
                # Dimensions (full extent)
                dims_x = bbox.extent.x * 2
                dims_y = bbox.extent.y * 2
                dims_z = bbox.extent.z * 2
                
                # Compute relative yaw (obstacle heading relative to ego heading)
                obstacle_yaw = math.radians(actor_transform.rotation.yaw)
                ego_yaw = math.radians(ego_transform.rotation.yaw)
                relative_yaw = obstacle_yaw - ego_yaw
                
                obstacle = Obstacle(
                    id=actor.id,
                    obstacle_type=obstacle_type,
                    x=bbox_center_relative.x,
                    y=-bbox_center_relative.y,  # Flip Y: CARLA left-handed -> ROS right-handed
                    z=bbox_center_relative.z,
                    vx=vx_rel,
                    vy=-vy_rel,  # Flip Y velocity as well
                    vz=vz_rel,
                    length=dims_x,
                    width=dims_y,
                    height=dims_z,
                    yaw=-relative_yaw,  # Flip yaw for right-handed convention
                )
                obstacles.append(obstacle)
                
        # Sort by distance (computed from position)
        obstacles.sort(key=lambda o: (o.x**2 + o.y**2 + o.z**2)**0.5)
        
        return obstacles
