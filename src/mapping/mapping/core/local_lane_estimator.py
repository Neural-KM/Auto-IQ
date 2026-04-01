"""
Local Lane Estimator
====================

Estimates lane structure from road boundaries using a priori knowledge.

Uses detected road boundaries (left/right edges) and divides the road into
lanes based on configured lane_count and total_road_width parameters.

Frame convention (base_link):
  x = forward,  y = lateral  (+left, -right)
  Left  boundary -> higher y  (divider / oncoming side)
  Right boundary -> lower  y  (wall / kerb side)

Right-hand drive lane numbering (lane_index):
  0 = rightmost (closest to wall/kerb)
  1, 2, ... = toward centre

NOTE: The "local_lane_data" lanes are NOT detected - they are estimated/guessed
by dividing the detected road width using configured lane_count and road_width
parameters. This uses a priori knowledge about expected road structure.

Changelog:
---------
2026-04-01: Refactored to receive boundaries directly (boundary extraction moved
    to BoundaryExtractor in lane_fitting_node). Removed estimate_from_semantic().
2026-03-25: Improved curve handling via updated boundary_extractor constraints.
    Boundary extraction now uses wider search windows and allows more lateral
    movement per step for better tracking on turns.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
from scipy.ndimage import uniform_filter1d


# Status codes for lane estimation
STATUS_OK = 0
STATUS_NO_BOUNDARIES = 1
STATUS_INVALID_WIDTH = 2
STATUS_INSUFFICIENT_POINTS = 3


# ============================================================================
# Result Dataclasses
# ============================================================================


@dataclass
class LaneResult:
    """Single lane within the local lane data."""
    lane_index: int
    left_boundary: np.ndarray    # (N, 2) x, y in base_link
    right_boundary: np.ndarray   # (N, 2) x, y in base_link
    centerline: np.ndarray       # (N, 2) x, y in base_link
    width_avg: float
    is_valid: bool
    is_partial: bool = False     # True if computed from single boundary


@dataclass
class LocalLaneDataResult:
    """Complete local lane data with all lanes."""
    valid: bool
    status_code: int
    left_boundary: np.ndarray    # (N, 2) raw left road edge
    right_boundary: np.ndarray   # (N, 2) raw right road edge
    lanes: List[LaneResult] = field(default_factory=list)
    road_width_avg: float = 0.0
    lane_count: int = 0


# ============================================================================
# Local Lane Estimator
# ============================================================================


class LocalLaneEstimator:
    """
    Estimates lane structure from road boundaries.

    Takes pre-extracted left and right road boundaries and estimates
    lane positions using configured lane_count and total_road_width parameters.

    This is NOT actual lane detection - it's geometric estimation using
    a priori knowledge about expected road structure.

    Usage:
        estimator = LocalLaneEstimator(lane_count=4, total_road_width=14.0)
        result = estimator.estimate_from_boundaries(left_pts, right_pts)
    """

    def __init__(
        self,
        # Lane structure parameters (a priori knowledge for estimation)
        lane_count: int = 4,
        total_road_width: float = 14.0,
        clearance_from_wall: float = 1.0,
        clearance_from_divider: float = 1.0,
        min_road_width: float = 10.0,
        max_road_width: float = 20.0,
        # Lane generation parameters
        resample_spacing: float = 0.5,
        min_forward_range: float = 3.0,
        near_field_skip: float = 1.0,
        smoothing_window: int = 7,
    ):
        """
        Initialize the local lane estimator.

        Args:
            lane_count: Expected number of lanes (a priori knowledge)
            total_road_width: Expected total road width in meters (a priori).
                Used when only one boundary is detected to estimate the other.
            clearance_from_wall: Buffer from right drivable edge to rightmost lane (meters).
                The extracted boundary is the OUTER edge of the drivable area, not the
                lane marking. This accounts for the shoulder/buffer area between the
                drivable edge and the actual rightmost lane boundary.
            clearance_from_divider: Buffer from left drivable edge to leftmost lane (meters).
                Same concept as clearance_from_wall but for the left side (center divider).
            min_road_width: Minimum valid road width (meters)
            max_road_width: Maximum valid road width (meters)
            resample_spacing: Distance between resampled points (meters)
            min_forward_range: Minimum forward span required (meters)
            near_field_skip: Skip points closer than this (meters)
            smoothing_window: Moving average window size (samples)
        """
        # Lane structure parameters
        self.lane_count = lane_count
        self.total_road_width = total_road_width
        self.clearance_from_wall = clearance_from_wall
        self.clearance_from_divider = clearance_from_divider
        self.min_road_width = min_road_width
        self.max_road_width = max_road_width

        # Lane generation parameters
        self.resample_spacing = resample_spacing
        self.min_forward_range = min_forward_range
        self.near_field_skip = near_field_skip
        self.smoothing_window = smoothing_window

        # Validation thresholds
        self.min_path_points = 8
        self.min_forward_span_m = 3.0

    def estimate_from_boundaries(
        self,
        left_boundary: Optional[np.ndarray],
        right_boundary: Optional[np.ndarray],
    ) -> LocalLaneDataResult:
        """
        Estimate lane structure from road boundaries.

        Args:
            left_boundary: (N, 2) array of left boundary points (x, y) in base_link
            right_boundary: (N, 2) array of right boundary points (x, y) in base_link

        Returns:
            LocalLaneDataResult with lane structure and status
        """
        # Convert to numpy arrays
        if left_boundary is not None:
            left_boundary = np.asarray(left_boundary)
            if left_boundary.size == 0:
                left_boundary = None
        if right_boundary is not None:
            right_boundary = np.asarray(right_boundary)
            if right_boundary.size == 0:
                right_boundary = None

        # No boundaries at all
        if left_boundary is None and right_boundary is None:
            return LocalLaneDataResult(
                valid=False,
                status_code=STATUS_NO_BOUNDARIES,
                left_boundary=np.empty((0, 2)),
                right_boundary=np.empty((0, 2)),
            )

        # Try to compute from both boundaries first
        if left_boundary is not None and right_boundary is not None:
            result = self._lanes_from_both_boundaries(left_boundary, right_boundary)
            if result is not None:
                return result
            # Fall through to single-boundary fallback

        # Single boundary fallback
        if right_boundary is not None:
            result = self._lanes_from_single_boundary(right_boundary, is_right=True)
        else:
            result = self._lanes_from_single_boundary(left_boundary, is_right=False)

        if result is None:
            return LocalLaneDataResult(
                valid=False,
                status_code=STATUS_INSUFFICIENT_POINTS,
                left_boundary=left_boundary if left_boundary is not None else np.empty((0, 2)),
                right_boundary=right_boundary if right_boundary is not None else np.empty((0, 2)),
            )

        return result

    # ========================================================================
    # Lane Generation (from boundaries)
    # ========================================================================

    def _lanes_from_both_boundaries(
        self,
        left_pts: np.ndarray,
        right_pts: np.ndarray,
    ) -> Optional[LocalLaneDataResult]:
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

        # Inset edges by buffer distances to get actual lane area
        # The detected boundaries are the OUTER drivable area edge, not lane markings.
        # There's a buffer (shoulder, etc.) between drivable edge and lane boundary.
        inner_right = right_y + self.clearance_from_wall      # Right lane boundary
        inner_left = left_y - self.clearance_from_divider     # Left lane boundary

        # Compute actual usable width for lanes (drivable width minus both buffers)
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

        return LocalLaneDataResult(
            valid=True,
            status_code=STATUS_OK,
            left_boundary=np.column_stack([x_grid, left_y]),
            right_boundary=np.column_stack([x_grid, right_y]),
            lanes=lanes,
            road_width_avg=float(np.mean(width)),
            lane_count=self.lane_count,
        )

    def _lanes_from_single_boundary(
        self,
        pts: np.ndarray,
        is_right: bool,
    ) -> Optional[LocalLaneDataResult]:
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

        return LocalLaneDataResult(
            valid=True,
            status_code=STATUS_OK,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
            lanes=lanes,
            road_width_avg=self.total_road_width,
            lane_count=self.lane_count,
        )
