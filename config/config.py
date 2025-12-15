"""
AutoIQ Configuration
====================

Unified configuration loader.
All settings are loaded from YAML files in the config/ directory.

Usage:
    from config.config import CARLA, CONTROL, SENSORS
    
    host = CARLA['connection']['host']
    speed = CONTROL['planning']['target_speed']
    for sensor in SENSORS:
        print(sensor['name'])
"""

import yaml
from pathlib import Path

_CONFIG_DIR = Path(__file__).parent


def _load_yaml(filename):
    """Load a YAML config file from the config directory."""
    yaml_path = _CONFIG_DIR / filename
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


# Expose all config sections as module-level variables
CARLA = _load_yaml('carla.yaml')
CONTROL = _load_yaml('control.yaml')
SENSORS = _load_yaml('sensors.yaml')['sensors']
