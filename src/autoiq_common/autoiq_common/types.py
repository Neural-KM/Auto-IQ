"""
Pure Python Types for Auto-IQ
=============================

These dataclasses are used for internal logic and do NOT import any ROS types.
They provide a clean separation between ROS messaging and core algorithms.

Use converters.py to convert between these types and ROS messages.
"""

from dataclasses import dataclass, field
from typing import List

@dataclass
class VehicleState:
    """
    Vehicle state in world coordinates.
    
    Attributes:
        x: X position in world frame (meters)
        y: Y position in world frame (meters)
        z: Z position in world frame (meters)
        yaw: Heading angle in radians (0 = +X axis, counter-clockwise positive)
        speed: Scalar speed magnitude (m/s)
        vx: Linear velocity in X direction (m/s)
        vy: Linear velocity in Y direction (m/s)
        vz: Linear velocity in Z direction (m/s)
        angular_x: Angular velocity around X axis (rad/s, roll rate)
        angular_y: Angular velocity around Y axis (rad/s, pitch rate)
        angular_z: Angular velocity around Z axis (rad/s, yaw rate)
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    speed: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    angular_x: float = 0.0
    angular_y: float = 0.0
    angular_z: float = 0.0


@dataclass
class TrajectoryPoint:
    """
    Single point in a trajectory with full kinematic state.
    
    Attributes:
        x: X position in world frame (meters)
        y: Y position in world frame (meters)
        yaw: Heading angle in radians
        s: Longitudinal position along reference path (Frenet s-coordinate)
        d: Lateral offset from reference path (Frenet d-coordinate)
        speed: Target speed at this point (m/s)
        curvature: Path curvature at this point (1/m, positive = left turn)
        time_from_start: Time from trajectory start (seconds)
    """
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    s: float = 0.0
    d: float = 0.0
    speed: float = 0.0
    curvature: float = 0.0
    time_from_start: float = 0.0


@dataclass
class Obstacle:
    """
    Detected or ground-truth obstacle in the environment.
    
    Attributes:
        id: Unique identifier for tracking
        type: Classification ("vehicle", "pedestrian", "cyclist", "static")
        x: X position in world frame (meters)
        y: Y position in world frame (meters)
        yaw: Heading angle in radians
        vx: Velocity in X direction (m/s)
        vy: Velocity in Y direction (m/s)
        length: Bounding box length (meters, front-to-back)
        width: Bounding box width (meters, side-to-side)
        height: Bounding box height (meters)
        confidence: Detection confidence [0.0, 1.0]
    """
    id: int = 0
    type: str = "unknown"
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    length: float = 4.5
    width: float = 2.0
    height: float = 1.5
    confidence: float = 1.0


@dataclass
class Trajectory:
    """
    A sequence of trajectory points representing a planned path.
    
    Attributes:
        points: List of TrajectoryPoint objects
        header_stamp_sec: Timestamp seconds (for synchronization)
        header_stamp_nanosec: Timestamp nanoseconds
        frame_id: Coordinate frame (default: "map")
    """
    points: List[TrajectoryPoint] = field(default_factory=list)
    header_stamp_sec: int = 0
    header_stamp_nanosec: int = 0
    frame_id: str = "map"


@dataclass
class ObstacleArray:
    """
    Collection of obstacles.
    
    Attributes:
        obstacles: List of Obstacle objects
        header_stamp_sec: Timestamp seconds
        header_stamp_nanosec: Timestamp nanoseconds
        frame_id: Coordinate frame
    """
    obstacles: List[Obstacle] = field(default_factory=list)
    header_stamp_sec: int = 0
    header_stamp_nanosec: int = 0
    frame_id: str = "map"
