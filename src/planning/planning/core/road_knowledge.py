"""Road Knowledge — a priori constants about the road.

Loaded once from ``config/road_knowledge.yaml``.
Right-hand drive only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields

import yaml


@dataclass
class RoadKnowledge:
    """Static road constants (all distances in metres)."""

    total_road_width: float = None
    lane_count: int = None
    clearance_from_wall: float = None
    clearance_from_divider: float = None
    preferred_lane: int = None
    min_road_width: float = None
    max_road_width: float = None
    local_follow_speed: float = None        # km/h
    curvature_speed_factor: float = None    # v_max = sqrt(factor/|κ|)

    # ── Factory ─────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str) -> "RoadKnowledge":
        """Load from a YAML file.  Missing keys keep their defaults."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
        valid_keys = {fld.name for fld in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def default_path(cls) -> str:
        """Return the default config path relative to the package source."""
        here = os.path.dirname(os.path.abspath(__file__))          # planning/core/
        return os.path.join(here, os.pardir, os.pardir,            # planning/
                            'config', 'road_knowledge.yaml')

    def __repr__(self) -> str:
        return (
            f"RoadKnowledge(road_w={self.total_road_width}m, "
            f"lanes={self.lane_count}, pref={self.preferred_lane}, "
            f"clear_wall={self.clearance_from_wall}m, "
            f"clear_div={self.clearance_from_divider}m)"
        )
