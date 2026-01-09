"""
Bounding Box Visualizer
=======================

Provides debug visualization for bounding boxes in the CARLA simulator.
"""

import carla


class BoundingBoxVisualizer:
    """
    Base class for drawing bounding boxes in the CARLA simulator.
    Works with relative coordinates to support both fake and real perception.
    """
    
    def __init__(self, world, ego_vehicle=None):
        """
        Initialize bounding box visualizer.
        
        Args:
            world: CARLA world instance
            ego_vehicle: The ego vehicle actor (needed for coordinate transforms)
        """
        self.world = world
        self._debug = world.debug
        self.debug_mode = False
        self.ego_vehicle = ego_vehicle
    
    def set_ego_vehicle(self, vehicle):
        """Set the ego vehicle for coordinate transforms."""
        self.ego_vehicle = vehicle
    
    def draw_bounding_box(self, relative_location, dimensions, yaw=0.0, color=None, life_time=0.5, thickness=0.1):
        """
        Draw a bounding box in the world using relative coordinates.
        
        Args:
            relative_location: Center of bbox relative to ego vehicle (x=forward, y=right, z=up)
                               Can be carla.Location or dict with 'x', 'y', 'z' keys
            dimensions: Full size of the box (length, width, height)
                        Can be carla.Vector3D or dict with 'x', 'y', 'z' keys
            yaw: Rotation of the box in degrees (relative to ego heading)
            color: Color of the box (carla.Color), defaults to red
            life_time: Duration to show the box in seconds
            thickness: Thickness of the lines
        """
        if self.ego_vehicle is None:
            return
            
        if color is None:
            color = carla.Color(255, 0, 0)  # Red
        
        # Convert relative_location to carla.Location
        rel_loc = carla.Location(
            x=float(relative_location['x']),
            y=float(relative_location['y']),
            z=float(relative_location['z'])
        )
        
        # Convert dimensions to extent (half-size)
        extent = carla.Vector3D(
            x=dimensions.x / 2,
            y=dimensions.y / 2,
            z=dimensions.z / 2
        )
        
        # Transform relative location to world coordinates
        ego_transform = self.ego_vehicle.get_transform()
        world_location = ego_transform.transform(rel_loc)
        
        # Calculate world rotation (ego yaw + relative yaw)
        world_yaw = ego_transform.rotation.yaw + yaw
        world_rotation = carla.Rotation(pitch=0, yaw=world_yaw, roll=0)
        
        # Create bounding box at world location
        bbox = carla.BoundingBox(world_location, extent)
        
        # Draw using CARLA's debug helper
        self._debug.draw_box(
            bbox,
            world_rotation,
            thickness=thickness,
            color=color,
            life_time=life_time
        )
        
        # Also draw edges manually for better visibility
        self._draw_box_edges(world_location, extent, world_rotation, color, life_time, thickness)
    
    def _draw_box_edges(self, center, extent, rotation, color, life_time, thickness):
        """
        Draw the 12 edges of a bounding box.
        
        Args:
            center: World location of box center
            extent: Half-size of the box
            rotation: World rotation of the box
            color: Line color
            life_time: Duration in seconds
            thickness: Line thickness
        """
        import math
        
        # Get yaw in radians
        yaw_rad = math.radians(rotation.yaw)
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)
        
        # Calculate 8 corners in local coordinates, then rotate and translate
        corners_local = [
            (-extent.x, -extent.y, -extent.z),
            ( extent.x, -extent.y, -extent.z),
            ( extent.x,  extent.y, -extent.z),
            (-extent.x,  extent.y, -extent.z),
            (-extent.x, -extent.y,  extent.z),
            ( extent.x, -extent.y,  extent.z),
            ( extent.x,  extent.y,  extent.z),
            (-extent.x,  extent.y,  extent.z),
        ]
        
        # Transform to world coordinates
        vertices = []
        for lx, ly, lz in corners_local:
            # Rotate around Z axis
            wx = center.x + lx * cos_yaw - ly * sin_yaw
            wy = center.y + lx * sin_yaw + ly * cos_yaw
            wz = center.z + lz
            vertices.append(carla.Location(x=wx, y=wy, z=wz))
        
        # Draw the 12 edges
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
        ]
        
        for i, j in edges:
            self._debug.draw_line(
                vertices[i],
                vertices[j],
                thickness=thickness,
                color=color,
                life_time=life_time
            )
    
    def draw_point(self, location, size=0.1, color=None, life_time=0.5):
        """
        Draw a point in the world.
        
        Args:
            location: carla.Location
            size: Size of the point
            color: carla.Color
            life_time: Duration in seconds
        """
        if color is None:
            color = carla.Color(255, 255, 0)  # Yellow
        
        self._debug.draw_point(
            location,
            size=size,
            color=color,
            life_time=life_time
        )
    
    def draw_arrow(self, start, end, color=None, life_time=0.5, thickness=0.1, arrow_size=0.3):
        """
        Draw an arrow in the world.
        
        Args:
            start: Start location (carla.Location)
            end: End location (carla.Location)
            color: carla.Color
            life_time: Duration in seconds
            thickness: Line thickness
            arrow_size: Size of the arrow head
        """
        if color is None:
            color = carla.Color(0, 255, 0)  # Green
        
        self._debug.draw_arrow(
            start,
            end,
            thickness=thickness,
            arrow_size=arrow_size,
            color=color,
            life_time=life_time
        )
    
    def draw_string(self, location, text, color=None, life_time=0.5):
        """
        Draw text in the world.
        
        Args:
            location: carla.Location
            text: String to display
            color: carla.Color
            life_time: Duration in seconds
        """
        if color is None:
            color = carla.Color(255, 255, 255)  # White
        
        self._debug.draw_string(
            location,
            text,
            draw_shadow=True,
            color=color,
            life_time=life_time
        )
