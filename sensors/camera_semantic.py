"""
Semantic Segmentation Camera
============================

Camera that provides semantic labels for each pixel.
"""

import carla
import numpy as np

from sensors.base_sensor import BaseSensor
from config import Settings


class CameraSemantic(BaseSensor):
    """Semantic Segmentation Camera sensor."""
    
    def __init__(self, vehicle, world, position,
                 width=1280, height=720, fov=90, sensor_tick=0.0):
        """
        Initialize semantic camera.
        
        Args:
            vehicle: CARLA vehicle actor
            world: CARLA world
            position: Dict with x, y, z, pitch, yaw, roll
            width: Image width in pixels
            height: Image height in pixels
            fov: Field of view in degrees
            sensor_tick: Sensor capture rate (0.0 = every frame)
        """
        super().__init__(vehicle, world)
        
        self.width = width
        self.height = height
        self.fov = fov
        self.sensor_tick = sensor_tick
        self.position = position
        
        # Store raw labels separately
        self.labels = None
    
    def _get_blueprint(self):
        """Get semantic segmentation camera blueprint."""
        bp_library = self.world.get_blueprint_library()
        bp = bp_library.find('sensor.camera.semantic_segmentation')
        bp.set_attribute('image_size_x', str(self.width))
        bp.set_attribute('image_size_y', str(self.height))
        bp.set_attribute('fov', str(self.fov))
        bp.set_attribute('sensor_tick', str(self.sensor_tick))
        return bp
    
    def _get_transform(self):
        """Get camera transform."""
        return carla.Transform(
            carla.Location(
                x=self.position['x'],
                y=self.position['y'],
                z=self.position['z']
            ),
            carla.Rotation(
                pitch=self.position['pitch'],
                yaw=self.position['yaw'],
                roll=self.position['roll']
            )
        )
    
    def _process_data(self, image):
        """Convert CARLA semantic image to label array."""
        # Raw data contains BGRA, but R channel has the semantic label
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        
        # Red channel contains semantic labels
        self.labels = array[:, :, 2].copy()
        
        return self.labels
    
    def get_labels(self):
        """Get raw semantic labels array."""
        return self.labels
    
    def get_road_mask(self):
        """Get binary mask of road pixels."""
        if self.labels is None:
            return None
        return self.labels == Settings.SemanticLabels.ROAD
    
    def get_lane_line_mask(self):
        """Get binary mask of lane line pixels."""
        if self.labels is None:
            return None
        return self.labels == Settings.SemanticLabels.ROAD_LINE
    
    def get_obstacle_mask(self):
        """
        Get binary mask of potential obstacles (non-road, non-sky).
        """
        if self.labels is None:
            return None
        
        road_mask = self.labels == Settings.SemanticLabels.ROAD
        road_line_mask = self.labels == Settings.SemanticLabels.ROAD_LINE
        sidewalk_mask = self.labels == Settings.SemanticLabels.SIDEWALK
        
        # Obstacles are anything on road that's not road/lines
        drivable = road_mask | road_line_mask | sidewalk_mask
        
        # We care about obstacles in our forward view (lower portion of image)
        return ~drivable
    
    def get_colored_image(self):
        """Convert labels to colored image for visualization."""
        if self.labels is None:
            return None
        
        # Create RGB image
        colored = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Apply colors for each label
        for label, color in Settings.SEMANTIC_COLORS.items():
            mask = self.labels == label
            colored[mask] = color
        
        return colored
