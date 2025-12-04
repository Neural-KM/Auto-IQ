"""
Depth Camera Sensor
===================

Camera that provides depth information for each pixel.
"""

import carla
import numpy as np

from sensors.base_sensor import BaseSensor
from config import Settings


class CameraDepth(BaseSensor):
    """Depth Camera sensor."""
    
    def __init__(self, vehicle, world, position='front',
                 width=None, height=None, fov=None):
        """
        Initialize depth camera.
        
        Args:
            vehicle: CARLA vehicle actor
            world: CARLA world
            position: 'front' or custom dict
            width: Image width
            height: Image height
            fov: Field of view
        """
        super().__init__(vehicle, world)
        
        self.width = width or Settings.CAMERA_WIDTH
        self.height = height or Settings.CAMERA_HEIGHT
        self.fov = fov or Settings.CAMERA_FOV
        
        if position == 'front':
            self.position = Settings.CAMERA_FRONT
        elif isinstance(position, dict):
            self.position = position
        else:
            self.position = Settings.CAMERA_FRONT
        
        # Depth in meters
        self.depth_meters = None
    
    def _get_blueprint(self):
        """Get depth camera blueprint."""
        bp_library = self.world.get_blueprint_library()
        bp = bp_library.find('sensor.camera.depth')
        bp.set_attribute('image_size_x', str(self.width))
        bp.set_attribute('image_size_y', str(self.height))
        bp.set_attribute('fov', str(self.fov))
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
        """Convert CARLA depth image to meters."""
        # Depth is encoded in RGB channels
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array.astype(np.float32)
        
        # Decode depth: (R + G*256 + B*256*256) / (256^3 - 1) * 1000
        # This gives depth in meters (max 1000m)
        normalized = (array[:, :, 2] + 
                     array[:, :, 1] * 256 + 
                     array[:, :, 0] * 256 * 256)
        normalized = normalized / (256 * 256 * 256 - 1)
        self.depth_meters = normalized * 1000.0
        
        return self.depth_meters
    
    def get_depth_meters(self):
        """Get depth array in meters."""
        return self.depth_meters
    
    def get_distance_at_point(self, x, y):
        """
        Get depth at specific pixel.
        
        Args:
            x: Pixel x coordinate
            y: Pixel y coordinate
            
        Returns:
            Depth in meters, or None if out of bounds
        """
        if self.depth_meters is None:
            return None
        
        if 0 <= y < self.height and 0 <= x < self.width:
            return self.depth_meters[int(y), int(x)]
        return None
    
    def get_min_distance_in_region(self, x1, y1, x2, y2):
        """
        Get minimum depth in a rectangular region.
        
        Args:
            x1, y1: Top-left corner
            x2, y2: Bottom-right corner
            
        Returns:
            Minimum depth in meters
        """
        if self.depth_meters is None:
            return None
        
        # Clamp to valid range
        x1 = max(0, min(x1, self.width - 1))
        x2 = max(0, min(x2, self.width - 1))
        y1 = max(0, min(y1, self.height - 1))
        y2 = max(0, min(y2, self.height - 1))
        
        region = self.depth_meters[int(y1):int(y2), int(x1):int(x2)]
        if region.size == 0:
            return None
        
        return np.min(region)
    
    def get_normalized_image(self):
        """Get depth as normalized grayscale for visualization."""
        if self.depth_meters is None:
            return None
        
        # Normalize to 0-255, clip at 100m for better visualization
        normalized = np.clip(self.depth_meters / 100.0, 0, 1)
        grayscale = (normalized * 255).astype(np.uint8)
        
        # Convert to RGB
        return np.stack([grayscale, grayscale, grayscale], axis=-1)
