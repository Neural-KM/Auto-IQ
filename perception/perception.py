"""Real Perception - Obstacle detection from sensor data."""

from typing import List

from navigation.obstacle_state import Obstacle


class Perception:
    """Real perception using sensor data (LiDAR, camera, etc.)."""
    
    def __init__(self, vehicle, world, sensor_suite):
        self.vehicle = vehicle
        self.world = world
        self.sensor_suite = sensor_suite
    
    def detect_obstacles(self, radius: float = 50.0) -> List[Obstacle]:
        """Detect obstacles from sensor data."""
        # TODO: Implement real perception pipeline
        # - Process LiDAR point cloud
        # - Run object detection on camera images
        # - Fuse sensor data
        # - Return list of Obstacle instances
        raise NotImplementedError("Real perception not yet implemented")
