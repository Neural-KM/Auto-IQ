"""
Semantic Segmentation Camera
============================

Camera that provides semantic labels for each pixel.
CARLA semantic segmentation label definitions and colors.
Reference: https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera
"""

import carla
import numpy as np

from sensors.base_sensor import BaseSensor


# =============================================================================
# Semantic Labels (CARLA default)
# =============================================================================

class SemanticLabels:
    """CARLA semantic segmentation label IDs."""
    
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
    SemanticLabels.ROAD: (128, 64, 128),        # Road - purple
    SemanticLabels.SIDEWALK: (244, 35, 232),    # Sidewalk - pink
    SemanticLabels.BUILDING: (70, 70, 70),      # Building - dark gray
    SemanticLabels.WALL: (102, 102, 156),       # Wall - grayish blue
    SemanticLabels.FENCE: (190, 153, 153),      # Fence - pinkish gray
    SemanticLabels.POLE: (153, 153, 153),       # Pole - gray
    SemanticLabels.TRAFFIC_LIGHT: (250, 170, 30),  # Traffic light - orange
    SemanticLabels.TRAFFIC_SIGN: (220, 220, 0), # Traffic sign - yellow
    SemanticLabels.VEGETATION: (107, 142, 35),  # Vegetation - olive
    SemanticLabels.TERRAIN: (152, 251, 152),    # Terrain - light green
    SemanticLabels.SKY: (70, 130, 180),         # Sky - steel blue
    SemanticLabels.PEDESTRIAN: (220, 20, 60),   # Pedestrian - crimson
    SemanticLabels.RIDER: (255, 0, 0),          # Rider - red
    SemanticLabels.VEHICLE: (0, 0, 142),        # Vehicle - dark blue
    SemanticLabels.TRUCK: (0, 0, 70),           # Truck - navy
    SemanticLabels.BUS: (0, 60, 100),           # Bus - dark teal
    SemanticLabels.TRAIN: (0, 80, 100),         # Train - teal
    SemanticLabels.MOTORCYCLE: (0, 0, 230),     # Motorcycle - blue
    SemanticLabels.BICYCLE: (119, 11, 32),      # Bicycle - dark red
    SemanticLabels.STATIC: (110, 190, 160),     # Static - seafoam
    SemanticLabels.DYNAMIC: (170, 120, 50),     # Dynamic - brown
    SemanticLabels.OTHER: (55, 90, 80),         # Other - dark teal
    SemanticLabels.WATER: (45, 60, 150),        # Water - blue
    SemanticLabels.ROAD_LINE: (157, 234, 50),   # Road line - green/yellow
    SemanticLabels.GROUND: (81, 0, 81),         # Ground - purple
    SemanticLabels.BRIDGE: (150, 100, 100),     # Bridge - brownish
    SemanticLabels.RAIL_TRACK: (230, 150, 140), # Rail track - salmon
    SemanticLabels.GUARD_RAIL: (180, 165, 180), # Guard rail - lavender
}


class CameraSemantic(BaseSensor):
    """Semantic Segmentation Camera sensor."""
    
    def __init__(self, vehicle, world, position,
                 width=1280, height=720, fov=90, sensor_tick=0.0):
        """
        Initialize semantic camera.
        
        Args:
            vehicle: CARLA vehicle actor
            world: CARLA world
            position: Dict with x, y, z, pitch, yaw, roll
            width: Image width in pixels
            height: Image height in pixels
            fov: Field of view in degrees
            sensor_tick: Sensor capture rate (0.0 = every frame)
        """
        super().__init__(vehicle, world)
        
        self.width = width
        self.height = height
        self.fov = fov
        self.sensor_tick = sensor_tick
        self.position = position
        
        # Store raw labels separately
        self.labels = None
    
    def _get_blueprint(self):
        """Get semantic segmentation camera blueprint."""
        bp_library = self.world.get_blueprint_library()
        bp = bp_library.find('sensor.camera.semantic_segmentation')
        bp.set_attribute('image_size_x', str(self.width))
        bp.set_attribute('image_size_y', str(self.height))
        bp.set_attribute('fov', str(self.fov))
        bp.set_attribute('sensor_tick', str(self.sensor_tick))
        return bp
    
    def _get_transform(self):
        """Get camera transform."""
        return carla.Transform(
            carla.Location(
                x=self.position['x'],
                y=self.position['y'],
                z=self.position['z']
            ),
            carla.Rotation(
                pitch=self.position['pitch'],
                yaw=self.position['yaw'],
                roll=self.position['roll']
            )
        )
    
    def _process_data(self, image):
        """Convert CARLA semantic image to label array."""
        # Raw data contains BGRA, but R channel has the semantic label
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        
        # Red channel contains semantic labels
        self.labels = array[:, :, 2].copy()
        
        return self.labels
    
    def get_labels(self):
        """Get raw semantic labels array."""
        return self.labels
    
    def get_road_mask(self):
        """Get binary mask of road pixels."""
        if self.labels is None:
            return None
        return self.labels == SemanticLabels.ROAD
    
    def get_lane_line_mask(self):
        """Get binary mask of lane line pixels."""
        if self.labels is None:
            return None
        return self.labels == SemanticLabels.ROAD_LINE
    
    def get_obstacle_mask(self):
        """
        Get binary mask of potential obstacles (non-road, non-sky).
        """
        if self.labels is None:
            return None
        
        road_mask = self.labels == SemanticLabels.ROAD
        road_line_mask = self.labels == SemanticLabels.ROAD_LINE
        sidewalk_mask = self.labels == SemanticLabels.SIDEWALK
        
        # Obstacles are anything on road that's not road/lines
        drivable = road_mask | road_line_mask | sidewalk_mask
        
        # We care about obstacles in our forward view (lower portion of image)
        return ~drivable
    
    def get_colored_image(self):
        """Convert labels to colored image for visualization."""
        if self.labels is None:
            return None
        
        # Create RGB image
        colored = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Apply colors for each label
        for label, color in SEMANTIC_COLORS.items():
            mask = self.labels == label
            colored[mask] = color
        
        return colored
