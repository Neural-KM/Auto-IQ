"""
Drivable Area Generator
=======================

Generates structured drivable area with lane information from detected
road boundaries. This module was migrated from planning.core.lane_midline_generator
to centralize lane computation in the mapping package.

Frame convention (base_link):
  x = forward,  y = lateral  (+left, -right)
  Left  boundary -> higher y  (divider / oncoming side)
  Right boundary -> lower  y  (wall / kerb side)

Right-hand drive lane numbering (lane_index):
  0 = rightmost (closest to wall/kerb)
  1, 2, ... = toward centre

Algorithm (both boundaries available):
  1. Resample both boundaries onto a common forward-distance grid.
  2. Measure road width, reject implausible widths.
  3. Inset right edge by clearance_from_wall, left edge by clearance_from_divider.
  4. Divide remaining width equally into lane_count lanes.
  5. Compute centerline for EACH lane.

Single-boundary fallback:
  Use nominal lane width from configuration to compute all lane positions.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from scipy.ndimage import uniform_filter1d


# Status codes
STATUS_OK = 0
STATUS_NO_BOUNDARIES = 1
STATUS_INVALID_WIDTH = 2
STATUS_INSUFFICIENT_POINTS = 3


@dataclass
class LaneResult:
    """Single lane within the drivable area."""
    lane_index: int
    left_boundary: np.ndarray    # (N, 2) x, y in base_link
    right_boundary: np.ndarray   # (N, 2) x, y in base_link
    centerline: np.ndarray       # (N, 2) x, y in base_link
    width_avg: float
    is_valid: bool
    is_partial: bool = False     # True if computed from single boundary


@dataclass
class DrivableAreaResult:
    """Complete drivable area with all lanes."""
    valid: bool
    status_code: int
    left_boundary: np.ndarray    # (N, 2) raw left road edge
    right_boundary: np.ndarray   # (N, 2) raw right road edge
    lanes: List[LaneResult] = field(default_factory=list)
    road_width_avg: float = 0.0
    lane_count: int = 0


class DrivableAreaGenerator:
    """
    Generates structured drivable area from road boundaries.

    Computes lane structure including centerlines for all lanes,
    allowing downstream consumers to select which lane to follow.
    """

    def __init__(
        self,
        lane_count: int,
        total_road_width: float,
        clearance_from_wall: float,
        clearance_from_divider: float,
        min_road_width: float,
        max_road_width: float,
        resample_spacing: float = 0.5,
        min_forward_range: float = 3.0,
        near_field_skip: float = 1.0,
        smoothing_window: int = 7,
    ):
        """
        Initialize the drivable area generator.

        Args:
            lane_count: Number of lanes to divide road into
            total_road_width: Expected total road width (meters)
            clearance_from_wall: Safety margin from right edge/kerb (meters)
            clearance_from_divider: Safety margin from left edge/divider (meters)
            min_road_width: Minimum valid road width (meters)
            max_road_width: Maximum valid road width (meters)
            resample_spacing: Distance between resampled points (meters)
            min_forward_range: Minimum forward span required (meters)
            near_field_skip: Skip points closer than this (meters)
            smoothing_window: Moving average window size (samples)
        """
        self.lane_count = lane_count
        self.total_road_width = total_road_width
        self.clearance_from_wall = clearance_from_wall
        self.clearance_from_divider = clearance_from_divider
        self.min_road_width = min_road_width
        self.max_road_width = max_road_width
        self.resample_spacing = resample_spacing
        self.min_forward_range = min_forward_range
        self.near_field_skip = near_field_skip
        self.smoothing_window = smoothing_window

    def generate(
        self,
        left_boundary: Optional[np.ndarray],
        right_boundary: Optional[np.ndarray],
    ) -> DrivableAreaResult:
        """
        Generate structured drivable area from road boundaries.

        Args:
            left_boundary: (N, 2) array of left boundary points (x, y) in base_link
            right_boundary: (N, 2) array of right boundary points (x, y) in base_link

        Returns:
            DrivableAreaResult with lane structure and status
        """
        # Convert to numpy arrays
        if left_boundary is not None:
            left_boundary = np.asarray(left_boundary)
        if right_boundary is not None:
            right_boundary = np.asarray(right_boundary)

        # No boundaries at all
        if left_boundary is None and right_boundary is None:
            return DrivableAreaResult(
                valid=False,
                status_code=STATUS_NO_BOUNDARIES,
                left_boundary=np.empty((0, 2)),
                right_boundary=np.empty((0, 2)),
            )

        # Try to compute from both boundaries first
        if left_boundary is not None and right_boundary is not None:
            result = self._from_both(left_boundary, right_boundary)
            if result is not None:
                return result
            # Fall through to single-boundary fallback

        # Single boundary fallback
        if right_boundary is not None:
            result = self._from_single(right_boundary, is_right=True)
        else:
            result = self._from_single(left_boundary, is_right=False)

        if result is None:
            return DrivableAreaResult(
                valid=False,
                status_code=STATUS_INSUFFICIENT_POINTS,
                left_boundary=left_boundary if left_boundary is not None else np.empty((0, 2)),
                right_boundary=right_boundary if right_boundary is not None else np.empty((0, 2)),
            )

        return result

    def _from_both(
        self,
        left_pts: np.ndarray,
        right_pts: np.ndarray,
    ) -> Optional[DrivableAreaResult]:
        """
        Compute lane structure when both boundaries are available.

        Resamples boundaries onto common grid, validates width,
        and computes centerline for each lane.
        """
        # Sort by x (forward distance)
        left_s = left_pts[np.argsort(left_pts[:, 0])]
        right_s = right_pts[np.argsort(right_pts[:, 0])]

        # Find common x range
        x_lo = max(left_s[0, 0], right_s[0, 0], self.near_field_skip)
        x_hi = min(left_s[-1, 0], right_s[-1, 0])

        if x_hi - x_lo < self.min_forward_range:
            return None

        # Resample onto common grid
        n = max(10, int((x_hi - x_lo) / self.resample_spacing))
        x_grid = np.linspace(x_lo, x_hi, n)

        left_y = np.interp(x_grid, left_s[:, 0], left_s[:, 1])
        right_y = np.interp(x_grid, right_s[:, 0], right_s[:, 1])

        # Validate road width
        width = left_y - right_y
        ok = (width >= self.min_road_width) & (width <= self.max_road_width)
        if np.sum(ok) < 5:
            return None

        # Filter to valid points
        x_grid = x_grid[ok]
        left_y = left_y[ok]
        right_y = right_y[ok]
        width = width[ok]

        # Smooth boundaries
        if len(x_grid) > self.smoothing_window:
            left_y = uniform_filter1d(left_y, size=self.smoothing_window, mode='nearest')
            right_y = uniform_filter1d(right_y, size=self.smoothing_window, mode='nearest')

        # Inset edges by clearances to get usable lane area
        inner_right = right_y + self.clearance_from_wall
        inner_left = left_y - self.clearance_from_divider

        # Compute actual usable width from detected boundaries (after clearances)
        usable_width = inner_left - inner_right

        # Compute lane width from actual detected road width
        lane_width = usable_width / self.lane_count

        lanes = []
        for idx in range(self.lane_count):
            # Lane boundaries positioned from the inner right edge (wall side)
            lane_right_y = inner_right + lane_width * idx
            lane_left_y = inner_right + lane_width * (idx + 1)

            # Lane centerline
            centerline_y = inner_right + lane_width * (idx + 0.5)

            lane_result = LaneResult(
                lane_index=idx,
                left_boundary=np.column_stack([x_grid, lane_left_y]),
                right_boundary=np.column_stack([x_grid, lane_right_y]),
                centerline=np.column_stack([x_grid, centerline_y]),
                width_avg=float(np.mean(lane_width)),
                is_valid=True,
                is_partial=False,
            )
            lanes.append(lane_result)

        return DrivableAreaResult(
            valid=True,
            status_code=STATUS_OK,
            left_boundary=np.column_stack([x_grid, left_y]),
            right_boundary=np.column_stack([x_grid, right_y]),
            lanes=lanes,
            road_width_avg=float(np.mean(width)),
            lane_count=self.lane_count,
        )

    def _from_single(
        self,
        pts: np.ndarray,
        is_right: bool,
    ) -> Optional[DrivableAreaResult]:
        """
        Compute lane structure from a single boundary using nominal lane width.

        Args:
            pts: Boundary points (N, 2) in base_link
            is_right: True if this is the right boundary (wall/kerb side)
        """
        # Sort by x
        s = pts[np.argsort(pts[:, 0])]
        x_lo = max(s[0, 0], self.near_field_skip)
        x_hi = s[-1, 0]

        if x_hi - x_lo < self.min_forward_range:
            return None

        # Resample
        n = max(10, int((x_hi - x_lo) / self.resample_spacing))
        x_grid = np.linspace(x_lo, x_hi, n)
        b_y = np.interp(x_grid, s[:, 0], s[:, 1])

        # Smooth
        if len(x_grid) > self.smoothing_window:
            b_y = uniform_filter1d(b_y, size=self.smoothing_window, mode='nearest')

        # Nominal lane width
        nominal_lane_w = self.total_road_width / self.lane_count

        lanes = []
        for idx in range(self.lane_count):
            if is_right:
                # Right boundary visible: lanes extend inward (higher y)
                offset_right = self.clearance_from_wall + nominal_lane_w * idx
                offset_left = self.clearance_from_wall + nominal_lane_w * (idx + 1)
                offset_center = self.clearance_from_wall + nominal_lane_w * (idx + 0.5)

                lane_right_y = b_y + offset_right
                lane_left_y = b_y + offset_left
                centerline_y = b_y + offset_center

                # Estimated boundaries
                right_boundary = np.column_stack([x_grid, b_y])
                left_boundary = np.column_stack([x_grid, b_y + self.total_road_width])
            else:
                # Left boundary visible: lanes extend inward (lower y)
                lanes_from_left = self.lane_count - idx - 1
                offset_right = self.clearance_from_divider + nominal_lane_w * (lanes_from_left + 1)
                offset_left = self.clearance_from_divider + nominal_lane_w * lanes_from_left
                offset_center = self.clearance_from_divider + nominal_lane_w * (lanes_from_left + 0.5)

                lane_right_y = b_y - offset_right
                lane_left_y = b_y - offset_left
                centerline_y = b_y - offset_center

                # Estimated boundaries
                left_boundary = np.column_stack([x_grid, b_y])
                right_boundary = np.column_stack([x_grid, b_y - self.total_road_width])

            lane_result = LaneResult(
                lane_index=idx,
                left_boundary=np.column_stack([x_grid, lane_left_y]),
                right_boundary=np.column_stack([x_grid, lane_right_y]),
                centerline=np.column_stack([x_grid, centerline_y]),
                width_avg=nominal_lane_w,
                is_valid=True,
                is_partial=True,
            )
            lanes.append(lane_result)

        return DrivableAreaResult(
            valid=True,
            status_code=STATUS_OK,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
            lanes=lanes,
            road_width_avg=self.total_road_width,
            lane_count=self.lane_count,
        )
