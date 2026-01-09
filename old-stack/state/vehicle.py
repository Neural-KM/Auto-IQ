"""Vehicle state data types."""

from dataclasses import dataclass

@dataclass
class VehicleState:
    """Current state of the ego vehicle."""
    x: float
    y: float
    z: float
    pitch: float
    yaw: float
    roll: float
    vx: float
    vy: float
    vz: float
    speed: float  # m/s
