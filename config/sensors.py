"""
Sensor Configuration
====================

Centralized sensor configuration for the autonomous vehicle.
Add, modify, or remove sensors by editing this file.

Each sensor entry defines:
    - type: Sensor class type ('rgb', 'semantic', 'depth')
    - name: Unique identifier for the sensor
    - position: Transform position (x, y, z) relative to vehicle
    - rotation: Transform rotation (pitch, yaw, roll) in degrees
    - fov: Field of view in degrees
    - width: Image width in pixels
    - height: Image height in pixels
    - sensor_tick: Data capture rate (0.0 = every frame, 0.05 = 20Hz, etc.)
"""


# =============================================================================
# Sensor Configurations
# =============================================================================

SENSORS = [
    # -------------------------------------------------------------------------
    # RGB Cameras
    # -------------------------------------------------------------------------
    {
        'type': 'rgb',
        'name': 'main',
        'position': {'x': -6.0, 'y': 0.0, 'z': 2.5},
        'rotation': {'pitch': -8.0, 'yaw': 0.0, 'roll': 0.0},
        'fov': 90,
        'width': 1280,
        'height': 720,
        'sensor_tick': 0.0,  # Every frame
    },
    {
        'type': 'rgb',
        'name': 'front_rgb',
        'position': {'x': 2.0, 'y': 0.0, 'z': 1.5},
        'rotation': {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0},
        'fov': 90,
        'width': 1280,
        'height': 720,
        'sensor_tick': 0.0,
    },
    
    # -------------------------------------------------------------------------
    # Semantic Segmentation Cameras
    # -------------------------------------------------------------------------
    {
        'type': 'semantic',
        'name': 'semantic',
        'position': {'x': 1.0, 'y': 0.0, 'z': 1.5},
        'rotation': {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0},
        'fov': 90,
        'width': 1280,
        'height': 720,
        'sensor_tick': 0.0,
    },
    
    # -------------------------------------------------------------------------
    # Depth Cameras
    # -------------------------------------------------------------------------
    {
        'type': 'depth',
        'name': 'depth',
        'position': {'x': 2.0, 'y': 0.0, 'z': 1.5},
        'rotation': {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0},
        'fov': 90,
        'width': 1280,
        'height': 720,
        'sensor_tick': 0.0,
    },
]
