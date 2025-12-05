"""
RGB Camera Sensor
=================

Standard RGB camera for visualization and perception.
"""

import carla
import numpy as np

from sensors.base_sensor import BaseSensor
from config import Settings


class CameraRGB(BaseSensor):
    """RGB Camera sensor."""
    
    def __init__(self, vehicle, world, position='third_person', 
                 width=None, height=None, fov=None):
        """
        Initialize RGB camera.
        
        Args:
            vehicle: CARLA vehicle actor
            world: CARLA world
            position: 'third_person' or 'front' or custom dict
            width: Image width (uses config default if None)
            height: Image height (uses config default if None)
            fov: Field of view (uses config default if None)
        """
        super().__init__(vehicle, world)
        
        self.width = width or Settings.CAMERA_WIDTH
        self.height = height or Settings.CAMERA_HEIGHT
        self.fov = fov or Settings.CAMERA_FOV
        
        # Set camera position
        if position == 'third_person':
            self.position = Settings.CAMERA_THIRD_PERSON
        elif position == 'front':
            self.position = Settings.CAMERA_FRONT
        elif isinstance(position, dict):
            self.position = position
        else:
            self.position = Settings.CAMERA_THIRD_PERSON
        
        self.surface = None  # For pygame rendering
    
    def _get_blueprint(self):
        """Get RGB camera blueprint."""
        bp_library = self.world.get_blueprint_library()
        bp = bp_library.find('sensor.camera.rgb')
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
        """Convert CARLA image to numpy array (RGB format)."""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        # CARLA returns BGRA, convert to RGB
        array = array[:, :, :3][:, :, ::-1]  # Remove alpha and BGR->RGB
        return array
    
    def get_pygame_surface(self):
        """Get pygame surface for display."""
        if self.data is None:
            return None
        
        try:
            import pygame
            # Data is already RGB
            self.surface = pygame.surfarray.make_surface(self.data.swapaxes(0, 1))
            return self.surface
        except ImportError:
            return None
