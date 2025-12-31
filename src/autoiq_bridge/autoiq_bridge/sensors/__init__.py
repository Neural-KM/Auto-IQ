"""
AutoIQ Bridge Sensors Package
=============================

Sensor classes for CARLA integration.
These are kept for reference but the bridge node handles sensors directly.
"""

from autoiq_bridge.sensors.base_sensor import BaseSensor
from autoiq_bridge.sensors.camera_rgb import CameraRGB
from autoiq_bridge.sensors.camera_depth import CameraDepth
from autoiq_bridge.sensors.camera_semantic import CameraSemantic
from autoiq_bridge.sensors.lidar import Lidar
from autoiq_bridge.sensors.sensor_suite import SensorSuite

__all__ = [
    'BaseSensor',
    'CameraRGB',
    'CameraDepth',
    'CameraSemantic',
    'Lidar',
    'SensorSuite',
]
