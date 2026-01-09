"""Lane detection data types."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class LaneDetection:
    """Result of a lane detection cycle."""
    valid: bool = False
    confidence: float = 0.0
    
    # Image coordinates (pixels)
    lane_center: Optional[int] = None
    left_lane: Optional[int] = None
    right_lane: Optional[int] = None
    lane_width: Optional[int] = None
    
    # Polynomial fits (coefficients)
    left_fit: Optional[Any] = None  # numpy array or list
    right_fit: Optional[Any] = None # numpy array or list
    
    # BEV (Bird's Eye View) metrics (meters)
    lane_width_bev: Optional[float] = None
    
    # Metadata
    bev_params: Dict[str, Any] = field(default_factory=dict)
