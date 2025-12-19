"""
Visualization Configuration
===========================

Defines which sensors appear in each layout.
"""

from config.config import SENSORS

# Display dimensions
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 860

# Minimum panel heights for tiled layout
BOTTOM_BAR_HEIGHT = 175
MIN_SIDE_PANEL_HEIGHT = 150

# Define which sensors to show in which layout
# These reference sensor names from config/sensors.py
LAYOUT_SENSORS = {
    'default': {
        'main': '3rd_person',
        'side': ['front_rgb', 'semantic', 'depth'],
    },
    'surround': {
        'grid': ['3rd_person', 'front_rgb', 'left_rgb', 'right_rgb', 'back_rgb', 'semantic', 'depth'],
    },
    'fullscreen': {
        'sensor': '3rd_person',  # Default sensor, can be changed via dropdown
    },
}


def get_all_sensor_names():
    """Get list of all sensor names from config."""
    return [s['name'] for s in SENSORS]


def get_layout_config(layout_name):
    """
    Get sensor configuration for a layout.
    Filters out sensors that don't exist in config.
    """
    layout = LAYOUT_SENSORS.get(layout_name, LAYOUT_SENSORS['default'])
    sensor_names = {s['name'] for s in SENSORS}
    
    # Filter to only include sensors that exist
    filtered = {}
    for key, value in layout.items():
        if isinstance(value, list):
            filtered[key] = [s for s in value if s in sensor_names]
        elif value in sensor_names:
            filtered[key] = value
        else:
            filtered[key] = None
    
    return filtered
