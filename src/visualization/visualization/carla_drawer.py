"""
CARLA Debug Drawer
==================

Utility class for drawing debug shapes in CARLA simulator.
Handles routes, trajectories, waypoints, and obstacles.
"""

import carla
import math
from typing import List, Tuple, Optional


class CarlaDrawer:
    """
    Utility for drawing debug visualizations in CARLA.
    
    All drawings are temporary and need to be refreshed each frame
    (or use lifetime parameter for persistent drawings).
    """
    
    # Color presets (RGBA)
    COLOR_GLOBAL_ROUTE = carla.Color(0, 255, 0, 255)      # Green
    COLOR_TRAJECTORY = carla.Color(0, 255, 255, 255)      # Cyan
    COLOR_PREDICTED = carla.Color(255, 165, 0, 255)       # Orange
    COLOR_WAYPOINTS = carla.Color(0, 0, 255, 255)         # Blue
    COLOR_VEHICLE = carla.Color(255, 0, 0, 255)           # Red
    COLOR_OBSTACLE_VEHICLE = carla.Color(255, 0, 0, 255)  # Red
    COLOR_OBSTACLE_PEDESTRIAN = carla.Color(255, 165, 0, 255)  # Orange
    COLOR_LOOKAHEAD = carla.Color(255, 255, 0, 255)       # Yellow
    
    def __init__(self, carla_world: carla.World):
        """
        Initialize CARLA drawer.
        
        Args:
            carla_world: CARLA world object
        """
        self.world = carla_world
        self.debug = carla_world.debug
    
    def draw_global_route(self, waypoints: List[Tuple[float, float, float]], 
                          lifetime: float = 0.1, thickness: float = 0.1):
        """
        Draw global route as a series of connected lines.
        
        Args:
            waypoints: List of (x, y, z) tuples
            lifetime: How long the drawing persists (seconds)
            thickness: Line thickness
        """
        if not waypoints or len(waypoints) < 2:
            return
        
        for i in range(len(waypoints) - 1):
            start = carla.Location(x=waypoints[i][0], y=waypoints[i][1], 
                                   z=waypoints[i][2] if len(waypoints[i]) > 2 else 0.5)
            end = carla.Location(x=waypoints[i+1][0], y=waypoints[i+1][1],
                                 z=waypoints[i+1][2] if len(waypoints[i+1]) > 2 else 0.5)
            
            self.debug.draw_line(start, end, thickness=thickness,
                                color=self.COLOR_GLOBAL_ROUTE, life_time=lifetime)
    
    def draw_trajectory(self, waypoints: List[Tuple[float, float]], 
                        lifetime: float = 0.1, thickness: float = 0.15,
                        color: carla.Color = None):
        """
        Draw local trajectory as connected lines.
        
        Args:
            waypoints: List of (x, y) tuples
            lifetime: How long the drawing persists (seconds)
            thickness: Line thickness
            color: Override color (default: cyan)
        """
        if not waypoints or len(waypoints) < 2:
            return
        
        draw_color = color or self.COLOR_TRAJECTORY
        z = 0.5  # Draw slightly above ground
        
        for i in range(len(waypoints) - 1):
            start = carla.Location(x=waypoints[i][0], y=waypoints[i][1], z=z)
            end = carla.Location(x=waypoints[i+1][0], y=waypoints[i+1][1], z=z)
            
            self.debug.draw_line(start, end, thickness=thickness,
                                color=draw_color, life_time=lifetime)
    
    def draw_predicted_trajectory(self, states: List[Tuple[float, float, float, float]],
                                   lifetime: float = 0.1, thickness: float = 0.12):
        """
        Draw MPC predicted trajectory.
        
        Args:
            states: List of (x, y, yaw, speed) tuples
            lifetime: How long the drawing persists
            thickness: Line thickness
        """
        if not states or len(states) < 2:
            return
        
        z = 0.6  # Slightly higher than trajectory
        
        for i in range(len(states) - 1):
            start = carla.Location(x=states[i][0], y=states[i][1], z=z)
            end = carla.Location(x=states[i+1][0], y=states[i+1][1], z=z)
            
            self.debug.draw_line(start, end, thickness=thickness,
                                color=self.COLOR_PREDICTED, life_time=lifetime)
        
        # Draw arrow at end to show direction
        if len(states) >= 2:
            last = states[-1]
            self.debug.draw_arrow(
                carla.Location(x=last[0], y=last[1], z=z),
                carla.Location(x=last[0] + math.cos(last[2]), 
                              y=last[1] + math.sin(last[2]), z=z),
                thickness=thickness * 0.5,
                arrow_size=0.3,
                color=self.COLOR_PREDICTED,
                life_time=lifetime
            )
    
    def draw_waypoints(self, waypoints: List[Tuple[float, float]], 
                       lifetime: float = 0.1, size: float = 0.2):
        """
        Draw waypoints as individual points.
        
        Args:
            waypoints: List of (x, y) tuples
            lifetime: How long the drawing persists
            size: Point size
        """
        if not waypoints:
            return
        
        z = 0.5
        for wp in waypoints:
            location = carla.Location(x=wp[0], y=wp[1], z=z)
            self.debug.draw_point(location, size=size,
                                 color=self.COLOR_WAYPOINTS, life_time=lifetime)
    
    def draw_lookahead_point(self, point: Tuple[float, float], 
                             vehicle_pos: Tuple[float, float],
                             lifetime: float = 0.1):
        """
        Draw lookahead point and line from vehicle.
        
        Args:
            point: Lookahead point (x, y)
            vehicle_pos: Vehicle position (x, y)
            lifetime: How long the drawing persists
        """
        z = 0.5
        
        # Draw point
        self.debug.draw_point(
            carla.Location(x=point[0], y=point[1], z=z),
            size=0.3,
            color=self.COLOR_LOOKAHEAD,
            life_time=lifetime
        )
        
        # Draw line from vehicle to lookahead
        self.debug.draw_line(
            carla.Location(x=vehicle_pos[0], y=vehicle_pos[1], z=z),
            carla.Location(x=point[0], y=point[1], z=z),
            thickness=0.05,
            color=self.COLOR_LOOKAHEAD,
            life_time=lifetime
        )
    
    def draw_vehicle_heading(self, x: float, y: float, yaw: float,
                             lifetime: float = 0.1, length: float = 3.0):
        """
        Draw vehicle heading arrow.
        
        Args:
            x, y: Vehicle position
            yaw: Vehicle heading in radians
            lifetime: How long the drawing persists
            length: Arrow length
        """
        z = 0.7
        
        start = carla.Location(x=x, y=y, z=z)
        end = carla.Location(
            x=x + length * math.cos(yaw),
            y=y + length * math.sin(yaw),
            z=z
        )
        
        self.debug.draw_arrow(start, end, thickness=0.1, arrow_size=0.3,
                             color=self.COLOR_VEHICLE, life_time=lifetime)
    
    def draw_obstacle_box(self, x: float, y: float, yaw: float,
                          length: float, width: float, height: float = 1.8,
                          is_vehicle: bool = True, lifetime: float = 0.1,
                          color: Tuple[int, int, int] = None):
        """
        Draw obstacle as a 3D bounding box.
        
        Args:
            x, y: Obstacle center position (world frame)
            yaw: Obstacle heading in radians
            length, width, height: Box dimensions
            is_vehicle: True for vehicle (red), False for pedestrian (orange)
            lifetime: How long the drawing persists
            color: Optional RGB tuple to override default color
        """
        if color:
            draw_color = carla.Color(color[0], color[1], color[2], 255)
        else:
            draw_color = self.COLOR_OBSTACLE_VEHICLE if is_vehicle else self.COLOR_OBSTACLE_PEDESTRIAN
        
        # Create bounding box
        extent = carla.Vector3D(x=length/2, y=width/2, z=height/2)
        box = carla.BoundingBox(carla.Location(x=x, y=y, z=height/2), extent)
        rotation = carla.Rotation(yaw=math.degrees(yaw))
        
        self.debug.draw_box(box, rotation, thickness=0.1,
                           color=draw_color, life_time=lifetime)
    
    def draw_text(self, text: str, x: float, y: float, z: float = 2.0,
                  lifetime: float = 0.1):
        """
        Draw text at a location.
        
        Args:
            text: Text to display
            x, y, z: Position
            lifetime: How long the text persists
        """
        self.debug.draw_string(
            carla.Location(x=x, y=y, z=z),
            text,
            draw_shadow=True,
            color=carla.Color(255, 255, 255),
            life_time=lifetime
        )
    
    def draw_speed_indicator(self, x: float, y: float, speed_kmh: float,
                             target_speed_kmh: float, lifetime: float = 0.1):
        """
        Draw speed indicator above vehicle.
        
        Args:
            x, y: Vehicle position
            speed_kmh: Current speed in km/h
            target_speed_kmh: Target speed in km/h
            lifetime: How long the text persists
        """
        text = f"{speed_kmh:.1f}/{target_speed_kmh:.1f} km/h"
        self.draw_text(text, x, y, z=3.0, lifetime=lifetime)
