"""Shared obstacle state container.

Stores latest normalized obstacles so components can read without direct coupling
or threading concerns.
"""

from dataclasses import dataclass
from typing import List, Optional


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


class ObstacleState:
    """Holds the latest normalized obstacles for consumers."""

    def __init__(self) -> None:
        self._obstacles: List[Obstacle] = []

    def update(self, obstacles: Optional[List[Obstacle]]) -> None:
        """Replace the stored obstacles with the latest snapshot."""
        self._obstacles = obstacles or []

    def get_all(self) -> List[Obstacle]:
        """Return all currently stored obstacles."""
        return self._obstacles

    def get_closest(self) -> Optional[Obstacle]:
        """Return the closest obstacle if any are present."""
        if not self._obstacles:
            return None
        return self._obstacles[0]
