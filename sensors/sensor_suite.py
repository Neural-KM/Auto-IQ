"""
Sensor Suite
============

Manages all sensors attached to the vehicle.
"""

import carla
from sensors.camera_rgb import CameraRGB
from sensors.camera_semantic import CameraSemantic
from sensors.camera_depth import CameraDepth
from config.sensors import SENSORS


# Mapping of sensor type to class
SENSOR_CLASSES = {
    'rgb': CameraRGB,
    'semantic': CameraSemantic,
    'depth': CameraDepth,
}

# Colors for different sensor types (R, G, B)
SENSOR_COLORS = {
    'rgb': carla.Color(0, 255, 0),       # Green
    'semantic': carla.Color(255, 0, 255), # Magenta
    'depth': carla.Color(0, 255, 255),    # Cyan
}


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
    
    def add_sensor(self, sensor_type, name, position, **kwargs):
        """
        Add a sensor of specified type.
        
        Args:
            sensor_type: 'rgb', 'semantic', or 'depth'
            name: Unique sensor name
            position: Dict with x, y, z, pitch, yaw, roll
            **kwargs: width, height, fov, sensor_tick
            
        Returns:
            The created sensor instance
        """
        sensor_class = SENSOR_CLASSES.get(sensor_type)
        if sensor_class is None:
            raise ValueError(f"Unknown sensor type: {sensor_type}")
        
        sensor = sensor_class(self.vehicle, self.world, position, **kwargs)
        sensor.spawn()
        self.sensors[name] = sensor
        return sensor
    
    def add_sensors_from_config(self):
        """
        Add all sensors defined in config/sensors.py.
        
        Returns:
            List of sensor names that were added
        """
        added_sensors = []
        
        for config in SENSORS:
            sensor_type = config['type']
            name = config['name']
            
            # Build position dict combining position and rotation
            position = {
                'x': config['position']['x'],
                'y': config['position']['y'],
                'z': config['position']['z'],
                'pitch': config['rotation']['pitch'],
                'yaw': config['rotation']['yaw'],
                'roll': config['rotation']['roll'],
            }
            
            # Build kwargs
            kwargs = {
                'width': config.get('width', 1280),
                'height': config.get('height', 720),
                'fov': config.get('fov', 90),
                'sensor_tick': config.get('sensor_tick', 0.0),
            }
            
            try:
                self.add_sensor(sensor_type, name, position, **kwargs)
                added_sensors.append(name)
            except Exception as e:
                print(f"[SensorSuite] Failed to add sensor '{name}': {e}")
        
        return added_sensors
    
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
    
    def draw_sensor_markers(self, life_time=0.1):
        """
        Draw debug markers in the simulator showing sensor positions.
        Call this every frame to keep markers visible.
        
        Args:
            life_time: How long markers persist (seconds)
        """
        vehicle_transform = self.vehicle.get_transform()
        
        for config in SENSORS:
            sensor_type = config['type']
            name = config['name']
            pos = config['position']
            rot = config['rotation']
            
            # Calculate world position of sensor
            # Sensor position is relative to vehicle
            local_location = carla.Location(x=pos['x'], y=pos['y'], z=pos['z'])
            world_location = vehicle_transform.transform(local_location)
            
            # Get color for this sensor type
            color = SENSOR_COLORS.get(sensor_type, carla.Color(255, 255, 255))
            
            # Draw a simple point to represent the camera
            self.world.debug.draw_point(
                world_location,
                size=0.1,
                color=color,
                life_time=life_time
            )
    
    def destroy_all(self):
        """Destroy all sensors."""
        for sensor in self.sensors.values():
            sensor.destroy()
        self.sensors.clear()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.destroy_all()
