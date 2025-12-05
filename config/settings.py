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
    
    # Camera positions (relative to vehicle)
    CAMERA_THIRD_PERSON = {
        'x': -6.0,
        'y': 0.0,
        'z': 2.5,
        'pitch': -8.0,
        'yaw': 0.0,
        'roll': 0.0
    }
    
    CAMERA_FRONT = {
        'x': 2.0,
        'y': 0.0,
        'z': 1.5,
        'pitch': 0.0,
        'yaw': 0.0,
        'roll': 0.0
    }
    
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
    
    # ==========================================================================
    # Semantic Segmentation Colors (CARLA default palette)
    # ==========================================================================
    class SemanticLabels:
        ROAD = 1           # Roads (128, 64, 128)
        SIDEWALK = 2       # Sidewalks (244, 35, 232)
        BUILDING = 3       # Buildings (70, 70, 70)
        WALL = 4           # Walls (102, 102, 156)
        FENCE = 5          # Fences (190, 153, 153)
        POLE = 6           # Poles (153, 153, 153)
        TRAFFIC_LIGHT = 7  # Traffic lights (250, 170, 30)
        TRAFFIC_SIGN = 8   # Traffic signs (220, 220, 0)
        VEGETATION = 9     # Vegetation (107, 142, 35)
        TERRAIN = 10       # Terrain (152, 251, 152)
        SKY = 11           # Sky (70, 130, 180)
        PEDESTRIAN = 12    # Pedestrians (220, 20, 60)
        RIDER = 13         # Riders (255, 0, 0)
        VEHICLE = 14       # Cars/Vans (0, 0, 142)
        TRUCK = 15         # Trucks (0, 0, 70)
        BUS = 16           # Buses (0, 60, 100)
        TRAIN = 17         # Trains (0, 80, 100)
        MOTORCYCLE = 18    # Motorcycles (0, 0, 230)
        BICYCLE = 19       # Bicycles (119, 11, 32)
        STATIC = 20        # Static objects (110, 190, 160)
        DYNAMIC = 21       # Dynamic objects (170, 120, 50)
        OTHER = 22         # Other (55, 90, 80)
        WATER = 23         # Water (45, 60, 150)
        ROAD_LINE = 24     # Road markings (157, 234, 50)
        GROUND = 25        # Ground (81, 0, 81)
        BRIDGE = 26        # Bridge (150, 100, 100)
        RAIL_TRACK = 27    # Rail tracks (230, 150, 140)
        GUARD_RAIL = 28    # Guard rails (180, 165, 180)
        
    # Colors in RGB format for visualization
    SEMANTIC_COLORS = {
        1: (128, 64, 128),   # Road - purple
        2: (244, 35, 232),   # Sidewalk - pink
        6: (153, 153, 153),  # Pole - gray
        12: (220, 20, 60),   # Pedestrian - crimson
        14: (0, 0, 142),     # Vehicle - dark blue
        24: (157, 234, 50),  # Road line - green/yellow
    }
