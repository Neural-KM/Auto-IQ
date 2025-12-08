"""
AutoIQ Configuration Settings
==============================

All tunable parameters in one place.
Control parameters are loaded from control_params.yaml for easy tuning.
"""

import os
import yaml


def _load_control_params():
    """Load control parameters from YAML file."""
    yaml_path = os.path.join(os.path.dirname(__file__), 'control_params.yaml')
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


# Load control params once at import time
_control_params = _load_control_params()


class Settings:
    """Central configuration for AutoIQ."""
    
    # ==========================================================================
    # CARLA Connection
    # ==========================================================================
    CARLA_HOST = '127.0.0.1'
    CARLA_PORT = 2000
    CARLA_TIMEOUT = 10.0
    
    # ==========================================================================
    # Simulation
    # ==========================================================================
    SYNC_MODE = True
    FIXED_DELTA_SECONDS = 0.05  # 20 FPS simulation
    
    # ==========================================================================
    # Vehicle
    # ==========================================================================
    VEHICLE_FILTER = 'vehicle.tesla.model3'
    
    # ==========================================================================
    # Display
    # ==========================================================================
    WINDOW_WIDTH = 1280
    WINDOW_HEIGHT = 720
    
    # ==========================================================================
    # Camera Settings
    # ==========================================================================
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    CAMERA_FOV = 90
    
    # ==========================================================================
    # Control - Lateral (Steering) - from YAML
    # ==========================================================================
    MAX_STEERING = _control_params['lateral']['max_steering']
    MAX_STEERING_DIFF = _control_params['lateral']['max_steering_diff']
    
    # ==========================================================================
    # Control - Longitudinal (Speed) - from YAML
    # ==========================================================================
    LONGITUDINAL_KP = _control_params['longitudinal']['kp']
    LONGITUDINAL_KI = _control_params['longitudinal']['ki']
    LONGITUDINAL_KD = _control_params['longitudinal']['kd']
    MAX_THROTTLE = _control_params['longitudinal']['max_throttle']
    MAX_BRAKE = _control_params['longitudinal']['max_brake']
    
    # ==========================================================================
    # Planning - from YAML
    # ==========================================================================
    TARGET_SPEED = _control_params['planning']['target_speed']
    LOOKAHEAD_DISTANCE = _control_params['planning']['lookahead_distance']
    WAYPOINT_SPACING = _control_params['planning']['waypoint_spacing']
    
    # ==========================================================================
    # Obstacle Avoidance - from YAML
    # ==========================================================================
    OBSTACLE_DETECTION_DISTANCE = _control_params['obstacle']['detection_distance']
    LANE_CHANGE_TRIGGER_DISTANCE = _control_params['obstacle']['lane_change_trigger_distance']
    SAFE_LANE_CHANGE_GAP = _control_params['obstacle']['safe_lane_change_gap']
