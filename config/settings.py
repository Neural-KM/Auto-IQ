"""
AutoIQ Configuration Settings
==============================

All tunable parameters in one place.
"""


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
    # Control - Lateral (Steering)
    # ==========================================================================
    LATERAL_KP = 1.5
    LATERAL_KI = 0.0
    LATERAL_KD = 0.3
    MAX_STEERING = 1.0
    MAX_STEERING_DIFF = 0.1  # Max change per step (smoothing)
    
    # ==========================================================================
    # Control - Longitudinal (Speed)
    # ==========================================================================
    LONGITUDINAL_KP = 1.0
    LONGITUDINAL_KI = 0.05
    LONGITUDINAL_KD = 0.0
    MAX_THROTTLE = 0.75
    MAX_BRAKE = 0.5
    
    # ==========================================================================
    # Planning
    # ==========================================================================
    TARGET_SPEED = 30.0  # km/h
    LOOKAHEAD_DISTANCE = 5.0  # meters
    WAYPOINT_SPACING = 2.0  # meters between waypoints
    
    # ==========================================================================
    # Obstacle Avoidance
    # ==========================================================================
    OBSTACLE_DETECTION_DISTANCE = 30.0  # meters
    LANE_CHANGE_TRIGGER_DISTANCE = 20.0  # meters
    SAFE_LANE_CHANGE_GAP = 10.0  # meters
    
    # ==========================================================================
    # Behavior States
    # ==========================================================================
    class BehaviorState:
        LANE_FOLLOWING = 'LANE_FOLLOWING'
        PREPARING_LANE_CHANGE = 'PREPARING_LANE_CHANGE'
        LANE_CHANGING = 'LANE_CHANGING'
        RETURNING_TO_LANE = 'RETURNING_TO_LANE'
