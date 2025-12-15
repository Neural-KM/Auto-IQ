# """
# AutoIQ Configuration Settings
# ==============================

# Unified configuration loader.
# All settings are loaded from YAML files in the config/ directory:
#   - carla.yaml: CARLA connection, simulation, vehicle settings
#   - control.yaml: Control parameters (lateral, longitudinal, planning, obstacle)
#   - sensors.yaml: Sensor definitions (loaded separately via sensors.py)

# Usage:
#     from config.settings import CARLA, CONTROL
    
#     host = CARLA['connection']['host']
#     speed = CONTROL['planning']['target_speed']
# """

# import yaml
# from pathlib import Path

# _CONFIG_DIR = Path(__file__).parent


# def _load_yaml(filename):
#     """Load a YAML config file from the config directory."""
#     yaml_path = _CONFIG_DIR / filename
#     with open(yaml_path, 'r') as f:
#         return yaml.safe_load(f)


# # Expose logical config sections as module-level variables
# CARLA = _load_yaml('carla.yaml')
# CONTROL = _load_yaml('control.yaml')
