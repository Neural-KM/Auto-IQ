"""
AutoIQ - Autonomous Truck Simulation
====================================

A modular framework for Level 4 autonomous truck simulation using CARLA.

Modules:
    config      - Configuration and settings
    sensors     - Camera, LiDAR, GPS sensors
    perception  - Lane and obstacle detection
    planning    - Global and local path planning
    control     - Vehicle controllers (PID)
    visualization - HUD and debug displays
    utils       - Helper utilities
"""

__version__ = '0.2.0'
__author__ = 'AutoIQ Team'

# Make key classes available at package level
from .config import Settings
from .control import VehicleController
from .sensors import SensorSuite
from .visualization import HUD
