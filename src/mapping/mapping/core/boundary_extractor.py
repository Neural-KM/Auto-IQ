"""
Road Boundary Extractor
=======================

Extracts left and right road boundaries from semantic segmentation images
using Bird's Eye View (BEV) transform and sliding window tracking.

Architecture:
    semantic_img → IPM → BEV masks → edge detection → sliding window → polynomial fit → base_link coords

Key concept - three area types in BEV:
    1. Drivable (white): road + road_line pixels - can drive here
    2. Obstruction (black): observed non-drivable - buildings, barriers, etc.
    3. Unmapped (gray): outside camera FOV - unknown, no boundary here

Boundaries only form between drivable and obstruction areas, NOT between
drivable and unmapped areas. This prevents false boundaries at FOV edges.

BEV coordinate convention (direct from IPM):
    - rows = X / forward:  row 0 = far (X_max), row max = near (X_min)
    - cols = Y / lateral:  col 0 = left (Y_max), col max = right (Y_min)

Author: Auto-IQ Mapping Stack
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


@dataclass
class BoundaryResult:
    """Single road boundary detection result in BEV pixel coordinates."""
    detected: bool = False
    poly_coeffs: Optional[np.ndarray] = None
    points_col: np.ndarray = field(default_factory=lambda: np.array([]))  # lateral (Y)
    points_row: np.ndarray = field(default_factory=lambda: np.array([]))  # forward (X)
    fitted_col: Optional[np.ndarray] = None
    fitted_row: Optional[np.ndarray] = None
    row_min: Optional[float] = None
    row_max: Optional[float] = None


@dataclass
class BoundaryExtractionResult:
    """Complete boundary extraction result."""
    left_boundary: Optional[np.ndarray] = None   # (N, 2) x, y in base_link
    right_boundary: Optional[np.ndarray] = None  # (N, 2) x, y in base_link
    debug_image: Optional[np.ndarray] = None     # BGR debug visualization
    valid: bool = False


class BoundaryExtractor:
    """
    Extracts road boundaries from semantic segmentation images.

    Pipeline:
        1. Create drivable and observed masks from semantic image
        2. Transform both to BEV using IPM
        3. Extract edges only where drivable meets observed obstruction
        4. Track boundaries using sliding window algorithm
        5. Fit polynomials to detected boundary points
        6. Convert to base_link metric coordinates

    Important: Boundaries only form between drivable area and observed
    obstructions. Unmapped areas (outside camera FOV) do not create boundaries.

    Usage:
        extractor = BoundaryExtractor(
            bev_x_range=(3.0, 60.0),
            bev_y_range=(-50.0, 50.0),
            pixels_per_meter=8.0,
        )
        extractor.set_camera_params(fov=90, pitch=5, camera_height=1.5)
        result = extractor.extract(semantic_img)
    """

    # Semantic class IDs
    ROAD_ID = 1
    ROAD_LINE_ID = 24

    def __init__(
        self,
        bev_x_range: Tuple[float, float],
        bev_y_range: Tuple[float, float],
        pixels_per_meter: float,
        # Sliding window parameters (in meters for resolution independence)
        num_windows: int = 15,
        window_margin_m: float = 7.5,
        min_pixels: int = 10,
        max_lateral_shift_m: float = 5.0,
        poly_order: int = 2,
    ):
        """
        Initialize boundary extractor.

        Args:
            bev_x_range: Forward range (x_min, x_max) in meters
            bev_y_range: Lateral range (y_min, y_max) in meters
            pixels_per_meter: BEV resolution
            num_windows: Number of sliding windows
            window_margin_m: Half-width of search window in meters
            min_pixels: Minimum edge pixels to update window center
            max_lateral_shift_m: Maximum lateral shift per window in meters
            poly_order: Polynomial order for curve fitting
        """
        self.bev_x_range = bev_x_range
        self.bev_y_range = bev_y_range
        self.pixels_per_meter = pixels_per_meter

        # Compute BEV dimensions
        x_span = bev_x_range[1] - bev_x_range[0]
        y_span = bev_y_range[1] - bev_y_range[0]
        self.bev_height = int(x_span * pixels_per_meter)
        self.bev_width = int(y_span * pixels_per_meter)
        self.resolution = 1.0 / pixels_per_meter

        # Sliding window parameters (convert meters to pixels)
        self.num_windows = num_windows
        self.window_margin = int(window_margin_m * pixels_per_meter)
        self.min_pixels = min_pixels
        self.max_lateral_shift = int(max_lateral_shift_m * pixels_per_meter)
        self.poly_order = poly_order

        # Camera parameters
        self.camera_fov: Optional[float] = None
        self.camera_pitch: Optional[float] = None
        self.camera_height: Optional[float] = None

    def set_camera_params(self, fov: float, pitch: float, camera_height: float):
        """Set camera parameters for IPM transform."""
        self.camera_fov = fov
        self.camera_pitch = pitch
        self.camera_height = camera_height

    def extract(self, semantic_img: np.ndarray) -> BoundaryExtractionResult:
        """
        Extract road boundaries from semantic segmentation image.

        Args:
            semantic_img: Mono8 semantic segmentation image

        Returns:
            BoundaryExtractionResult with boundaries in base_link coordinates
        """
        from mapping.utils.ipm_transform import convert_to_ipm

        result = BoundaryExtractionResult()

        if self.camera_fov is None:
            return result

        # 1. Create masks from semantic image
        # Drivable = road + road_line
        drivable_mask = (
            (semantic_img == self.ROAD_ID) | (semantic_img == self.ROAD_LINE_ID)
        ).astype(np.uint8) * 255

        # Observed = any labeled pixel (semantic_id != 0 means camera saw something)
        observed_mask = (semantic_img != 0).astype(np.uint8) * 255

        # 2. Transform both masks to BEV
        bev_drivable, _, _ = convert_to_ipm(
            image=drivable_mask,
            fov=self.camera_fov,
            pitch=self.camera_pitch,
            camera_height=self.camera_height,
            bev_x_range=self.bev_x_range,
            bev_y_range=self.bev_y_range,
            pixels_per_meter=self.pixels_per_meter,
            interpolation=cv2.INTER_NEAREST,
        )

        bev_observed, _, _ = convert_to_ipm(
            image=observed_mask,
            fov=self.camera_fov,
            pitch=self.camera_pitch,
            camera_height=self.camera_height,
            bev_x_range=self.bev_x_range,
            bev_y_range=self.bev_y_range,
            pixels_per_meter=self.pixels_per_meter,
            interpolation=cv2.INTER_NEAREST,
        )

        # Binary masks
        drivable = (bev_drivable > 127).astype(np.uint8)
        observed = (bev_observed > 127).astype(np.uint8)

        # Obstruction = observed but not drivable
        obstruction = observed & ~drivable

        # 3. Extract edges only at drivable-obstruction boundaries
        edge_img = self._extract_edges(drivable, obstruction)

        # 4. Find boundaries using sliding window
        left, right, diags = self._find_boundaries(edge_img)

        # 5. Convert to base_link coordinates
        result.left_boundary = self._to_base_link(left)
        result.right_boundary = self._to_base_link(right)
        result.valid = result.left_boundary is not None or result.right_boundary is not None

        # 6. Generate debug visualization (show unmapped in gray)
        result.debug_image = self._draw_debug(drivable, observed, left, right, diags, edge_img)

        return result

    def _extract_edges(self, drivable: np.ndarray, obstruction: np.ndarray) -> np.ndarray:
        """
        Extract boundary edges between drivable area and obstructions.

        Uses Sobel gradient for precise edge detection, then filters to only
        keep edges adjacent to actual obstructions. Transitions from drivable
        to unmapped (unobserved) areas are ignored.

        Args:
            drivable: Binary mask of drivable area (1=drivable)
            obstruction: Binary mask of observed obstructions (1=obstruction)

        Returns:
            Binary edge image (255 at boundary pixels, 0 elsewhere)
        """
        # Sobel gradient for precise edge detection on drivable mask
        drivable_u8 = drivable * 255
        sobel_x = cv2.Sobel(drivable_u8, cv2.CV_64F, 1, 0, ksize=3)
        edges = np.uint8(np.abs(sobel_x) > 30)

        # Only keep edges that are adjacent to obstruction (not unmapped)
        kernel = np.ones((5, 5), dtype=np.uint8)
        obstruction_nearby = cv2.dilate(obstruction, kernel, iterations=1)

        return (edges & obstruction_nearby) * 255

    def _find_boundaries(
        self, edge_img: np.ndarray
    ) -> Tuple[BoundaryResult, BoundaryResult, List[tuple]]:
        """
        Find left and right boundaries using sliding window algorithm.

        Returns:
            left_result, right_result, diagnostics_list
        """
        h, w = edge_img.shape

        # Initialize from histogram peaks
        histogram = np.sum(edge_img, axis=0).astype(np.float64)
        midpoint = w // 2

        # Find peaks in left and right halves
        left_hist = histogram[:midpoint]
        right_hist = histogram[midpoint:]

        # Use peak if there's significant signal, else use default positions
        left_base = int(np.argmax(left_hist)) if np.max(left_hist) > 0 else midpoint // 2
        right_base = int(np.argmax(right_hist)) + midpoint if np.max(right_hist) > 0 else midpoint + midpoint // 2

        # Sliding window setup
        sweep_margin = int(h * 0.05)
        sweep_start = max(10, sweep_margin)
        sweep_end = h - max(10, sweep_margin)
        window_h = max(1, (sweep_end - sweep_start) // self.num_windows)

        left_current, right_current = left_base, right_base
        left_pts_col, left_pts_row = [], []
        right_pts_col, right_pts_row = [], []
        diags = []

        # Get all edge pixel locations
        nz_row, nz_col = edge_img.nonzero()

        # Sweep from near (high row) to far (low row)
        for win_idx in range(self.num_windows):
            r_hi = sweep_end - win_idx * window_h
            r_lo = sweep_end - (win_idx + 1) * window_h

            # Find edge pixels in each search window
            left_mask = (
                (nz_row >= r_lo) & (nz_row < r_hi) &
                (nz_col >= left_current - self.window_margin) &
                (nz_col < left_current + self.window_margin)
            )
            right_mask = (
                (nz_row >= r_lo) & (nz_row < r_hi) &
                (nz_col >= right_current - self.window_margin) &
                (nz_col < right_current + self.window_margin)
            )

            left_cols = nz_col[left_mask]
            left_rows = nz_row[left_mask]
            right_cols = nz_col[right_mask]
            right_rows = nz_row[right_mask]

            # Update window centers with constraints
            left_frozen = len(left_cols) < self.min_pixels
            right_frozen = len(right_cols) < self.min_pixels

            if not left_frozen:
                new_left = int(np.mean(left_cols))
                if abs(new_left - left_current) <= self.max_lateral_shift:
                    left_current = new_left
                else:
                    left_frozen = True

            if not right_frozen:
                new_right = int(np.mean(right_cols))
                if abs(new_right - right_current) <= self.max_lateral_shift:
                    right_current = new_right
                else:
                    right_frozen = True

            # Collect points
            if len(left_cols) > 0:
                left_pts_col.extend(left_cols.tolist())
                left_pts_row.extend(left_rows.tolist())
            if len(right_cols) > 0:
                right_pts_col.extend(right_cols.tolist())
                right_pts_row.extend(right_rows.tolist())

            diags.append((r_lo, r_hi, left_current, right_current, self.window_margin, left_frozen, right_frozen))

        # Fit polynomials
        left_result = self._fit_polynomial(left_pts_col, left_pts_row, h)
        right_result = self._fit_polynomial(right_pts_col, right_pts_row, h)

        return left_result, right_result, diags

    def _fit_polynomial(
        self, cols: List[int], rows: List[int], row_extent: int
    ) -> BoundaryResult:
        """Fit polynomial col = f(row) to boundary points."""
        result = BoundaryResult()

        if len(cols) < self.poly_order + 1:
            return result

        cols_arr = np.array(cols, dtype=np.float64)
        rows_arr = np.array(rows, dtype=np.float64)

        try:
            coeffs = np.polyfit(rows_arr, cols_arr, self.poly_order)
        except (np.linalg.LinAlgError, ValueError):
            return result

        result.detected = True
        result.poly_coeffs = coeffs
        result.points_col = cols_arr
        result.points_row = rows_arr
        result.row_min = float(np.min(rows_arr))
        result.row_max = float(np.max(rows_arr))

        # Evaluate polynomial
        r_start = int(max(0, np.floor(result.row_min)))
        r_end = int(min(row_extent - 1, np.ceil(result.row_max)))
        if r_end >= r_start:
            result.fitted_row = np.arange(r_start, r_end + 1, dtype=np.float64)
            result.fitted_col = np.polyval(coeffs, result.fitted_row)

        return result

    def _to_base_link(self, result: BoundaryResult, sample_step: int = 3) -> Optional[np.ndarray]:
        """Convert BEV pixel coordinates to base_link metric coordinates."""
        if not result.detected or result.fitted_col is None:
            return None

        rows = result.fitted_row[::sample_step]
        cols = result.fitted_col[::sample_step]

        # Clip to valid range
        valid = (rows >= 0) & (rows < self.bev_height) & (cols >= 0) & (cols < self.bev_width)
        rows, cols = rows[valid], cols[valid]

        if len(rows) == 0:
            return None

        # BEV to base_link: x = X_max - row*res, y = Y_max - col*res
        x_base = self.bev_x_range[1] - rows * self.resolution
        y_base = self.bev_y_range[1] - cols * self.resolution

        # Sort by x ascending (near to far)
        idx = np.argsort(x_base)
        return np.column_stack([x_base[idx], y_base[idx]])

    def _draw_debug(
        self,
        drivable: np.ndarray,
        observed: np.ndarray,
        left: BoundaryResult,
        right: BoundaryResult,
        diags: List[tuple],
        edge_img: np.ndarray,
    ) -> np.ndarray:
        """
        Generate debug visualization.

        Color coding:
            - White: Drivable area (road)
            - Black: Obstruction (observed non-drivable)
            - Gray: Unmapped (outside camera FOV)
            - Red edges: Detected boundaries
            - Green/Red boxes: Sliding windows (green=active, red=frozen)
            - Yellow/Cyan lines: Fitted boundary polynomials
        """
        h, w = drivable.shape

        # Create base visualization with three area types
        # Start with gray (unmapped)
        vis = np.full((h, w, 3), 128, dtype=np.uint8)

        # Observed non-drivable = black (obstruction)
        obstruction = observed & ~drivable
        vis[obstruction == 1] = [0, 0, 0]

        # Drivable = white
        vis[drivable == 1] = [255, 255, 255]

        # Overlay edges in red
        edge_mask = edge_img > 0
        vis[edge_mask] = [0, 0, 255]

        GREEN, RED, YELLOW, CYAN = (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 0)

        # Draw windows
        overlay = vis.copy()
        for r_lo, r_hi, left_c, right_c, margin, left_frozen, right_frozen in diags:
            # Left window
            color = RED if left_frozen else GREEN
            pt1 = (max(0, left_c - margin), r_lo)
            pt2 = (min(w - 1, left_c + margin), r_hi)
            cv2.rectangle(overlay, pt1, pt2, color, -1)
            cv2.rectangle(vis, pt1, pt2, color, 2)
            # Right window
            color = RED if right_frozen else GREEN
            pt1 = (max(0, right_c - margin), r_lo)
            pt2 = (min(w - 1, right_c + margin), r_hi)
            cv2.rectangle(overlay, pt1, pt2, color, -1)
            cv2.rectangle(vis, pt1, pt2, color, 2)

        vis = cv2.addWeighted(overlay, 0.3, vis, 0.7, 0)

        # Draw fitted curves
        for boundary, color in [(left, YELLOW), (right, CYAN)]:
            if boundary.detected and boundary.fitted_col is not None:
                pts = np.column_stack([
                    np.clip(boundary.fitted_col.astype(np.int32), 0, w - 1),
                    np.clip(boundary.fitted_row.astype(np.int32), 0, h - 1),
                ])
                cv2.polylines(vis, [pts], False, color, 3)

        return vis
