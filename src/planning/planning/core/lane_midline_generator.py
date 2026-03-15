"""
Lane Midline Generator  (Right-Hand Drive only)
================================================

Takes detected road boundaries (from mapping, in base_link frame) and
a priori road knowledge to produce the midline of the preferred lane.

Frame convention (base_link):
  x = forward,  y = lateral  (+left, −right)
  Left  boundary → higher y  (divider / oncoming side)
  Right boundary → lower  y  (wall / kerb side)

Right-hand drive lane numbering (preferred_lane index):
  0 = rightmost (closest to wall/kerb)
  1, 2, … = toward centre

Algorithm (both boundaries available):
  1. Resample both boundaries onto a common forward-distance grid.
  2. Measure road width,  reject implausible widths.
  3. Inset right edge by clearance_from_wall,  left edge by clearance_from_divider.
  4. Divide remaining width equally into lane_count lanes.
  5. Pick the midline of preferred_lane.

Single-boundary fallback:
  Use nominal lane width from road knowledge to offset from the visible edge.

Author : Auto-IQ Planning Stack
"""

from __future__ import annotations

import numpy as np
from typing import Optional
from scipy.ndimage import uniform_filter1d

from planning.core.road_knowledge import RoadKnowledge


class LaneMidlineGenerator:
    """Generates lane midlines from road boundaries + road knowledge."""

    # Flow control codes
    SUCCESS = 0
    FALLBACK_NO_BOUNDARIES = 1
    FALLBACK_INVALID = 2
    FALLBACK_SHORT = 3
    FAIL_NO_BOUNDARIES = -1
    FAIL_INVALID = -2
    FAIL_SHORT = -3

    STATUS_MSGS = {
        SUCCESS: "Midline generated successfully.",
        FALLBACK_NO_BOUNDARIES: "No boundaries available. Using previous midline.",
        FALLBACK_INVALID: "Generated midline invalid or insufficient points. Using previous midline.",
        FALLBACK_SHORT: "Generated midline is too short. Using previous midline.",
        FAIL_NO_BOUNDARIES: "No boundaries available and no previous midline.",
        FAIL_INVALID: "Generated midline invalid and no previous midline.",
        FAIL_SHORT: "Generated midline is too short and no previous midline.",
    }

    def __init__(self, road_knowledge: RoadKnowledge):
        self.road = road_knowledge

        # Processing tunables
        self.smoothing_window: int = 7       # moving-avg window (samples)
        self.resample_spacing: float = 0.5   # m between resampled points
        self.min_forward_range: float = 3.0  # min useful forward span (m)
        self.near_field_skip: float = 1.0    # skip very close pts (m)

        # Temporal persistence
        self._last_midline: Optional[np.ndarray] = None

    # ─────────────────── public API ─────────────────────────────

    def generate(
        self,
        left_boundary: Optional[np.ndarray],
        right_boundary: Optional[np.ndarray],
    ) -> tuple[Optional[np.ndarray], int]:
        """Return ((K,2) midline in base_link, status_code)."""

        if left_boundary is not None:
            left_boundary = np.asarray(left_boundary)
        if right_boundary is not None:
            right_boundary = np.asarray(right_boundary)

        if left_boundary is None and right_boundary is None:
            if self._last_midline is not None:
                return self._last_midline, self.FALLBACK_NO_BOUNDARIES
            return None, self.FAIL_NO_BOUNDARIES

        if left_boundary is not None and right_boundary is not None:
            midline = self._from_both(left_boundary, right_boundary)
            if midline is None:
                # Fallback to right first, then left if both failed
                midline = self._from_single(right_boundary, is_right=True)
                if midline is None:
                    midline = self._from_single(left_boundary, is_right=False)
        elif right_boundary is not None:
            midline = self._from_single(right_boundary, is_right=True)
        else:
            midline = self._from_single(left_boundary, is_right=False)

        if midline is None or len(midline) < 3:
            if self._last_midline is not None:
                return self._last_midline, self.FALLBACK_INVALID
            return None, self.FAIL_INVALID

        midline = self._smooth(midline)

        if midline[-1, 0] - midline[0, 0] < self.min_forward_range:
            if self._last_midline is not None:
                return self._last_midline, self.FALLBACK_SHORT
            return None, self.FAIL_SHORT

        self._last_midline = midline
        return midline, self.SUCCESS

    # ────────────── both-boundary path ──────────────────────────

    def _from_both(self, left_pts, right_pts) -> Optional[np.ndarray]:
        left_s = left_pts[np.argsort(left_pts[:, 0])]
        right_s = right_pts[np.argsort(right_pts[:, 0])]

        x_lo = max(left_s[0, 0], right_s[0, 0], self.near_field_skip)
        x_hi = min(left_s[-1, 0], right_s[-1, 0])
        if x_hi - x_lo < self.min_forward_range:
            return None

        n = max(10, int((x_hi - x_lo) / self.resample_spacing))
        x_grid = np.linspace(x_lo, x_hi, n)

        left_y = np.interp(x_grid, left_s[:, 0], left_s[:, 1])
        right_y = np.interp(x_grid, right_s[:, 0], right_s[:, 1])

        width = left_y - right_y
        ok = (width >= self.road.min_road_width) & (width <= self.road.max_road_width)
        if np.sum(ok) < 5:
            return None

        x_grid, left_y, right_y = x_grid[ok], left_y[ok], right_y[ok]

        # Inset edges by clearances
        inner_right = right_y + self.road.clearance_from_wall
        inner_left = left_y - self.road.clearance_from_divider

        # Divide usable width into equal lanes
        lane_w = (inner_left - inner_right) / self.road.lane_count
        idx = self.road.preferred_lane
        midline_y = inner_right + lane_w * (idx + 0.5)

        return np.column_stack([x_grid, midline_y])

    # ────────────── single-boundary fallback ────────────────────

    def _from_single(self, pts, is_right: bool) -> Optional[np.ndarray]:
        s = pts[np.argsort(pts[:, 0])]
        x_lo = max(s[0, 0], self.near_field_skip)
        x_hi = s[-1, 0]
        if x_hi - x_lo < self.min_forward_range:
            return None

        n = max(10, int((x_hi - x_lo) / self.resample_spacing))
        x_grid = np.linspace(x_lo, x_hi, n)
        b_y = np.interp(x_grid, s[:, 0], s[:, 1])

        nominal_lane_w = self.road.total_road_width / self.road.lane_count
        idx = self.road.preferred_lane

        if is_right:
            # Right edge = wall side → midline is inward (higher y)
            offset = self.road.clearance_from_wall + nominal_lane_w * (idx + 0.5)
            midline_y = b_y + offset
        else:
            # Left edge = divider side → midline is inward (lower y)
            lanes_from_left = self.road.lane_count - idx - 1
            offset = self.road.clearance_from_divider + nominal_lane_w * (lanes_from_left + 0.5)
            midline_y = b_y - offset

        return np.column_stack([x_grid, midline_y])

    # ────────────── helpers ─────────────────────────────────────

    def _smooth(self, midline: np.ndarray) -> np.ndarray:
        if len(midline) <= self.smoothing_window:
            return midline
        y_sm = uniform_filter1d(midline[:, 1], size=self.smoothing_window, mode='nearest')
        return np.column_stack([midline[:, 0], y_sm])
