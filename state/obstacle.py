"""Obstacle data types."""

from dataclasses import dataclass

@dataclass(frozen=True)
class Obstacle:
    """Generic obstacle representation in ego-relative coordinates."""
    id: int
    obstacle_type: str          # 'vehicle' or 'pedestrian'
    # Position in ego-relative frame (meters)
    x: float                    # forward positive
    y: float                    # left positive
    z: float                    # up positive
    # Velocity in ego-relative frame (m/s)
    vx: float
    vy: float
    vz: float
    # Bounding box extents (meters, full size)
    length: float               # extent in x
    width: float                # extent in y
    height: float               # extent in z
