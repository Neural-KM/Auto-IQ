"""
Base Sensor Class
=================

Abstract base class for all sensors.
"""

import weakref
from abc import ABC, abstractmethod


class BaseSensor(ABC):
    """Abstract base class for sensors."""
    
    def __init__(self, vehicle, world):
        """
        Initialize base sensor.
        
        Args:
            vehicle: CARLA vehicle actor to attach to
            world: CARLA world
        """
        self.vehicle = vehicle
        self.world = world
        self.sensor = None
        self.data = None
        self._data_ready = False
    
    @abstractmethod
    def _get_blueprint(self):
        """Get the sensor blueprint. Must be implemented by subclass."""
        pass
    
    @abstractmethod
    def _get_transform(self):
        """Get the sensor transform. Must be implemented by subclass."""
        pass
    
    @abstractmethod
    def _process_data(self, data):
        """Process incoming sensor data. Must be implemented by subclass."""
        pass
    
    def spawn(self):
        """Spawn the sensor and attach to vehicle."""
        bp = self._get_blueprint()
        transform = self._get_transform()
        
        self.sensor = self.world.spawn_actor(
            bp,
            transform,
            attach_to=self.vehicle
        )
        
        # Set up callback with weak reference
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda data: BaseSensor._on_data(weak_self, data))
    
    @staticmethod
    def _on_data(weak_self, data):
        """Static callback to handle incoming data."""
        self = weak_self()
        if self is not None:
            self.data = self._process_data(data)
            self._data_ready = True
    
    def get_data(self):
        """Get the latest sensor data."""
        return self.data
    
    def is_ready(self):
        """Check if sensor has received data."""
        return self._data_ready
    
    def destroy(self):
        """Clean up sensor."""
        if self.sensor is not None:
            self.sensor.stop()
            self.sensor.destroy()
            self.sensor = None
