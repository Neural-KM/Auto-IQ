"""
Sensor Suite
============

Manages all sensors attached to the vehicle.
"""

from sensors.camera_rgb import CameraRGB
from sensors.camera_semantic import CameraSemantic
from sensors.camera_depth import CameraDepth


class SensorSuite:
    """
    Manages all sensors for the autonomous vehicle.
    Provides unified interface to sensor data.
    """
    
    def __init__(self, vehicle, world):
        """
        Initialize sensor suite.
        
        Args:
            vehicle: CARLA vehicle actor
            world: CARLA world
        """
        self.vehicle = vehicle
        self.world = world
        self.sensors = {}
    
    def add_rgb_camera(self, name='rgb_main', position='third_person', **kwargs):
        """Add an RGB camera."""
        camera = CameraRGB(self.vehicle, self.world, position, **kwargs)
        camera.spawn()
        self.sensors[name] = camera
        return camera
    
    def add_semantic_camera(self, name='semantic', position='front', **kwargs):
        """Add a semantic segmentation camera."""
        camera = CameraSemantic(self.vehicle, self.world, position, **kwargs)
        camera.spawn()
        self.sensors[name] = camera
        return camera
    
    def add_depth_camera(self, name='depth', position='front', **kwargs):
        """Add a depth camera."""
        camera = CameraDepth(self.vehicle, self.world, position, **kwargs)
        camera.spawn()
        self.sensors[name] = camera
        return camera
    
    def get_sensor(self, name):
        """Get a sensor by name."""
        return self.sensors.get(name)
    
    def get_data(self, name):
        """
        Get data from a specific sensor by name.
        
        Args:
            name: Sensor name
            
        Returns:
            Sensor data (numpy array) or None if sensor not found/ready
        """
        sensor = self.sensors.get(name)
        if sensor is not None:
            return sensor.get_data()
        return None
    
    def get_all_data(self):
        """Get data from all sensors."""
        return {name: sensor.get_data() for name, sensor in self.sensors.items()}
    
    def all_ready(self):
        """Check if all sensors have received data."""
        return all(sensor.is_ready() for sensor in self.sensors.values())
    
    def destroy_all(self):
        """Destroy all sensors."""
        for sensor in self.sensors.values():
            sensor.destroy()
        self.sensors.clear()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.destroy_all()
