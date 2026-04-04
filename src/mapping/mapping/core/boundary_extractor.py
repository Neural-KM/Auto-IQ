"""
Road Boundary Extractor — Sliding Window on BEV Drivable-Area Mask
==================================================================

Extracts left and right continuous road boundaries from a binary Bird's-Eye
View (BEV) local lane data mask produced by Inverse Perspective Mapping.

The algorithm uses a highly constrained Sliding Window approach designed to
explicitly reject sudden road cutouts such as U-turns, intersections, and
missing mask data.

The grid convention (from IPM) is:
    grid[r, c]  —  rows = Y / lateral,  cols = X / forward
    x_base = origin_x + c * resolution
    y_base = origin_y + r * resolution
    Row 0  = most-negative Y (right side of vehicle)
    Col 0  = most-negative X (behind vehicle)

The sliding window sweeps along the **column** (forward) axis and tracks
boundary **row** (lateral) positions.  No transposition or flipping is
performed at any stage.

Designed for Level 4 autonomous driving on heavy-duty trucks where lane
boundary stability is safety-critical.

Changelog:
---------
2026-04-01: Fixed sweep range bug:
    - Changed sweep_start from w//2 to 5% of width (was missing near-field boundaries)
    - BEV col 0 = X_min (near edge), not vehicle center, so full sweep is needed
2026-03-25: Improved curve handling:
    - Increased MAX_LATERAL_SHIFT from 25 to 45 pixels for better turn tracking
    - Increased MARGIN from 40 to 60 for wider search windows
    - Enhanced visualization with thicker window borders and alpha blending
    - Windows now more visible with semi-transparent fill

Author : Auto-IQ Mapping Stack
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


# ───────────────────────── data classes ──────────────────────────

@dataclass
class BoundaryResult:
    """Container for a single road-boundary extraction result.

    In this module's convention:
        xs  = collected **row** positions   (lateral / Y — the dependent var)
        ys  = collected **column** positions (forward / X — the independent var)
        fitted_x = polynomial-evaluated row values
        fitted_y = column values used for evaluation
    """

    poly_coeffs: Optional[np.ndarray] = None  # 2nd-order [a, b, c]
    xs: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    ys: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    fitted_x: Optional[np.ndarray] = None     # row values  (lateral)
    fitted_y: Optional[np.ndarray] = None     # col values  (forward)
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    detected: bool = False


# ───────────────────── helper: edge extraction ───────────────────

def _extract_boundary_edges(binary_mask: np.ndarray) -> np.ndarray:
    """
    Extract road-boundary edges from a binary drivable-area mask.

    Road boundaries run roughly along the column (forward) axis.
    Transitions happen along the row (lateral) axis, so we use a
    **vertical** Sobel gradient (dy = 1).

    Returns a binary edge image (uint8, 0 / 255).
    """
    mask_u8 = binary_mask.astype(np.uint8) * 255

    # Vertical gradient — detects edges where intensity changes across rows
    sobel_y = cv2.Sobel(mask_u8, cv2.CV_64F, 0, 1, ksize=3)
    edge = np.uint8(np.absolute(sobel_y) > 30) * 255
    return edge


# ───────────────────── core algorithm ────────────────────────────

def extract_boundaries(
    binary_warped: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[BoundaryResult, BoundaryResult, np.ndarray, list, list]:
    """
    Extract left and right road boundaries from a BEV binary mask.

    Parameters
    ----------
    binary_warped : np.ndarray
        2-D binary image, shape (rows, cols).  1 = drivable, 0 = background.
        Rows = Y / lateral.  Cols = X / forward.
    valid_mask : np.ndarray, optional
        Same shape.  1 = observed by camera, 0 = unseen / UNKNOWN.

    Returns
    -------
    left_result  : BoundaryResult   (high-row boundary = left of vehicle)
    right_result : BoundaryResult   (low-row boundary = right of vehicle)
    edge_img     : np.ndarray       (intermediate edge image for debug)
    left_diags   : list[tuple]      (c_lo, c_hi, r_center, margin, frozen)
    right_diags  : list[tuple]      (c_lo, c_hi, r_center, margin, frozen)
    """
    # ── Hardcoded parameters ────────────────────────────────────
    NWINDOWS = 15
    MARGIN = 60                    # half-height of search window (rows) - increased from 40
    MINPIX = 10                    # min edge pixels to recenter
    MAX_LATERAL_SHIFT = 45         # max row-shift between consecutive windows - increased from 25
    LANE_WIDTH_TOLERANCE = 60      # allowed row-width deviation - increased from 40
    POLY_ORDER = 2
    VALID_MASK_ERODE = 5

    h, w = binary_warped.shape[:2]          # h = rows (lateral), w = cols (forward)

    # ── Valid mask ──────────────────────────────────────────────
    if valid_mask is None:
        valid_mask = np.ones((h, w), dtype=np.uint8)
    else:
        valid_mask = valid_mask.astype(np.uint8)

    if VALID_MASK_ERODE > 0:
        kern = np.ones((VALID_MASK_ERODE, VALID_MASK_ERODE), dtype=np.uint8)
        valid_mask = cv2.erode(valid_mask, kern, iterations=1)

    # ── 1. Edge extraction ──────────────────────────────────────
    edge_img = _extract_boundary_edges(binary_warped)
    edge_img = (edge_img * valid_mask).astype(np.uint8)

    # ── 2. Histogram initialisation ─────────────────────────────
    # Sum edge pixels across ALL columns → one value per row.
    # Peaks indicate the two road boundary rows.
    histogram = np.sum(edge_img, axis=1).astype(np.float64)

    midpoint = h // 2
    # Row 0 = right side of vehicle → right boundary has row < midpoint
    # Row h = left side of vehicle  → left  boundary has row > midpoint
    right_base = int(np.argmax(histogram[:midpoint]))
    left_base  = int(np.argmax(histogram[midpoint:])) + midpoint

    # Expected lane width in rows
    expected_width = left_base - right_base
    if expected_width <= 0:
        expected_width = h // 2

    # ── 3. Sliding window setup ─────────────────────────────────
    # Sweep along columns from near edge to far edge.
    # NOTE: The BEV covers X_min to X_max (e.g., 5m to 35m), with col 0 = X_min.
    # We start from a small offset to avoid noisy data at the very near edge.
    sweep_start = max(10, int(w * 0.05))  # Start 5% into the BEV (avoids near-field noise)
    sweep_range = w - sweep_start
    window_w = max(1, sweep_range // NWINDOWS)
    margin = MARGIN
    minpix = MINPIX

    right_current = right_base
    left_current  = left_base

    # Collectors  (xs = row positions, ys = col positions)
    right_xs: List[int] = []
    right_ys: List[int] = []
    left_xs:  List[int] = []
    left_ys:  List[int] = []

    right_diags: list = []
    left_diags:  list = []

    nonzero = edge_img.nonzero()
    nz_row  = np.array(nonzero[0])
    nz_col  = np.array(nonzero[1])

    # ── 4. Core loop — near to far along columns ───────────────
    for win_idx in range(NWINDOWS):
        c_lo = sweep_start + win_idx * window_w
        c_hi = sweep_start + (win_idx + 1) * window_w

        # Right boundary search window (low rows)
        r_right_lo = right_current - margin
        r_right_hi = right_current + margin

        # Left boundary search window (high rows)
        r_left_lo = left_current - margin
        r_left_hi = left_current + margin

        # Edge pixels inside each window
        right_inds = (
            (nz_col >= c_lo) & (nz_col < c_hi) &
            (nz_row >= r_right_lo) & (nz_row < r_right_hi)
        )
        left_inds = (
            (nz_col >= c_lo) & (nz_col < c_hi) &
            (nz_row >= r_left_lo) & (nz_row < r_left_hi)
        )

        right_pix_r = nz_row[right_inds]
        right_pix_c = nz_col[right_inds]
        left_pix_r  = nz_row[left_inds]
        left_pix_c  = nz_col[left_inds]

        # Candidate new row-centres
        right_new = int(np.mean(right_pix_r)) if len(right_pix_r) >= minpix else right_current
        left_new  = int(np.mean(left_pix_r))  if len(left_pix_r)  >= minpix else left_current

        # ── Constraint checks ──────────────────────────────────
        right_frozen = len(right_pix_r) < minpix
        left_frozen  = len(left_pix_r)  < minpix

        # (b) Momentum / max lateral shift
        if not right_frozen and abs(right_new - right_current) > MAX_LATERAL_SHIFT:
            right_new = right_current
            right_frozen = True
        if not left_frozen and abs(left_new - left_current) > MAX_LATERAL_SHIFT:
            left_new = left_current
            left_frozen = True

        # (c) Constant lane width enforcement
        proposed_width = left_new - right_new
        if proposed_width > expected_width + LANE_WIDTH_TOLERANCE:
            right_delta = abs(right_new - right_current)
            left_delta  = abs(left_new  - left_current)
            if right_delta > left_delta:
                right_new = left_new - expected_width
                right_frozen = True
            else:
                left_new = right_new + expected_width
                left_frozen = True

        # ── Accept updates ──────────────────────────────────────
        right_current = right_new
        left_current  = left_new

        if len(right_pix_r) > 0:
            right_xs.extend(right_pix_r.tolist())
            right_ys.extend(right_pix_c.tolist())
        if len(left_pix_r) > 0:
            left_xs.extend(left_pix_r.tolist())
            left_ys.extend(left_pix_c.tolist())

        # Diagnostics: (c_lo, c_hi, r_center, margin, frozen)
        right_diags.append((c_lo, c_hi, right_current, margin, right_frozen))
        left_diags.append((c_lo, c_hi, left_current, margin, left_frozen))

    # ── 5. Polynomial fitting  row = f(col) ────────────────────
    right_result = _fit_boundary(right_xs, right_ys, w, POLY_ORDER)
    left_result  = _fit_boundary(left_xs,  left_ys,  w, POLY_ORDER)

    return left_result, right_result, edge_img, left_diags, right_diags


# ───────────────────── curve fitting helper ──────────────────────

def _fit_boundary(
    xs: List[int],
    ys: List[int],
    col_extent: int,
    order: int = 2,
) -> BoundaryResult:
    """Fit a polynomial  row = f(col)  to collected boundary pixels.

    xs = row positions (dependent), ys = col positions (independent).
    """
    result = BoundaryResult()
    if len(xs) < order + 1:
        return result

    xs_arr = np.array(xs, dtype=np.float64)   # rows
    ys_arr = np.array(ys, dtype=np.float64)   # cols

    try:
        coeffs = np.polyfit(ys_arr, xs_arr, order)
    except (np.linalg.LinAlgError, ValueError):
        return result

    result.poly_coeffs = coeffs
    result.xs = xs_arr
    result.ys = ys_arr
    result.detected = True
    result.y_min = float(np.min(ys_arr))
    result.y_max = float(np.max(ys_arr))

    # Evaluate only within supported column range
    c_start = int(max(0, np.floor(result.y_min)))
    c_end   = int(min(col_extent - 1, np.ceil(result.y_max)))
    if c_end < c_start:
        return result

    plot_col = np.arange(c_start, c_end + 1, dtype=np.float64)
    plot_row = np.polyval(coeffs, plot_col)
    result.fitted_x = plot_row    # row values  (lateral)
    result.fitted_y = plot_col    # col values  (forward)

    return result


# ───────────────────── visualisation ─────────────────────────────

def draw_boundary_overlay(
    bev_img: np.ndarray,
    left_result: BoundaryResult,
    right_result: BoundaryResult,
    left_diags: list,
    right_diags: list,
    edge_img: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Draw sliding windows and fitted polynomials on a copy of the BEV image.

    Everything is drawn in **native grid space** (rows = lateral, cols = forward).
    At the end the image is rotated for human-friendly display:
        top = ahead,  bottom = behind,  left = left,  right = right.

    Windows: green = valid (active tracking),  red = frozen / constrained.
             Windows now drawn with semi-transparent fill for better visibility.
    Curves:  yellow = left boundary,  cyan = right boundary.

    Each diag entry is a tuple: (c_lo, c_hi, r_center, margin, is_frozen).
    """
    # Build 3-channel canvas in native grid space
    if bev_img.ndim == 2:
        vis = cv2.cvtColor(bev_img, cv2.COLOR_GRAY2BGR)
    else:
        vis = bev_img.copy()

    # Blend edge image (faintly)
    if edge_img is not None:
        edge_bgr = cv2.cvtColor(edge_img, cv2.COLOR_GRAY2BGR)
        vis = cv2.addWeighted(vis, 0.8, edge_bgr, 0.3, 0)

    GREEN  = (0, 255, 0)
    RED    = (0, 0, 255)
    YELLOW = (0, 255, 255)
    CYAN   = (255, 255, 0)

    # Draw sliding windows with semi-transparent fill for better visibility
    overlay = vis.copy()
    for c_lo, c_hi, r_center, margin, is_frozen in left_diags + right_diags:
        colour = RED if is_frozen else GREEN
        pt1 = (int(c_lo), int(r_center - margin))
        pt2 = (int(c_hi), int(r_center + margin))
        # Draw filled rectangle on overlay
        cv2.rectangle(overlay, pt1, pt2, colour, -1)
        # Draw border on main image with thicker line
        cv2.rectangle(vis, pt1, pt2, colour, 3)

    # Blend overlay with main image for semi-transparent windows
    vis = cv2.addWeighted(overlay, 0.3, vis, 0.7, 0)

    # Draw fitted curves
    for result, colour in [(left_result, YELLOW), (right_result, CYAN)]:
        if result.detected and result.fitted_x is not None:
            pts = np.column_stack([
                result.fitted_y.astype(np.int32),   # col → x
                result.fitted_x.astype(np.int32),   # row → y
            ])
            pts[:, 0] = np.clip(pts[:, 0], 0, vis.shape[1] - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, vis.shape[0] - 1)
            cv2.polylines(vis, [pts], isClosed=False, color=colour, thickness=3)

    # ── Display rotation ────────────────────────────────────────
    # Native grid: rows top→bot = right→left,  cols left→right = behind→ahead.
    # Desired:     top = ahead,  right = right.
    # 90° CCW then horizontal flip achieves this.
    vis = cv2.flip(cv2.rotate(vis, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)

    return vis


# ───────────────────── BEV-pixel → base_link conversion ─────────

def boundary_to_base_link(
    result: BoundaryResult,
    grid_height: int,
    grid_width: int,
    origin_x: float,
    origin_y: float,
    resolution: float,
    sample_step: int = 5,
) -> Optional[np.ndarray]:
    """
    Convert a fitted boundary from grid-pixel coordinates to ``base_link``
    metric coordinates suitable for publishing as a ``nav_msgs/Path``.

    Grid mapping (from IPM):
        x_base = origin_x + col * resolution
        y_base = origin_y + row * resolution

    In BoundaryResult (no transpose, no flip):
        fitted_y = column values  (forward / X)
        fitted_x = row values     (lateral / Y)

    Returns
    -------
    points : np.ndarray, shape (N, 2)
        (x_base, y_base) in the vehicle frame, ordered near → far.
        *None* if the boundary was not detected.
    """
    if not result.detected or result.fitted_x is None or result.fitted_y is None:
        return None

    cols = result.fitted_y[::sample_step]    # forward
    rows = result.fitted_x[::sample_step]    # lateral

    # Clip to valid grid range
    valid = (
        (cols >= 0) & (cols < grid_width) &
        (rows >= 0) & (rows < grid_height)
    )
    cols = cols[valid]
    rows = rows[valid]

    if len(cols) == 0:
        return None

    # Direct mapping — no transpose to undo
    x_base = origin_x + cols * resolution
    y_base = origin_y + rows * resolution

    return np.column_stack([x_base, y_base])


# ───────────────────── BoundaryExtractor class ─────────────────────


@dataclass
class BoundaryExtractionResult:
    """Result from boundary extraction pipeline."""
    left_boundary: Optional[np.ndarray] = None   # (N, 2) x, y in base_link
    right_boundary: Optional[np.ndarray] = None  # (N, 2) x, y in base_link
    debug_image: Optional[np.ndarray] = None     # BGR debug visualization
    valid: bool = False


class BoundaryExtractor:
    """
    Extracts road boundaries from semantic segmentation images.

    Creates a Bird's Eye View (BEV) using IPM transform, then uses sliding
    window algorithm to extract left and right road boundaries.

    The extracted boundaries represent the OUTER edge of the drivable area
    (road + road_line), NOT the lane markings themselves. The lane estimator
    must account for the buffer area between drivable edge and actual lanes.

    Usage:
        extractor = BoundaryExtractor(
            bev_x_range=(5.0, 35.0),
            bev_y_range=(-15.0, 15.0),
            pixels_per_meter=10.0,
        )
        extractor.set_camera_params(fov=90, pitch=5, height=1.5)
        result = extractor.extract(semantic_img)
    """

    # Semantic class IDs (must match perception/data_classes/semantic_labels.py)
    ROAD_ID = 1
    ROAD_LINE_ID = 24

    def __init__(
        self,
        bev_x_range: Tuple[float, float],
        bev_y_range: Tuple[float, float],
        pixels_per_meter: float,
    ):
        """
        Initialize the boundary extractor.

        Args:
            bev_x_range: Forward range for BEV transform (meters)
            bev_y_range: Lateral range for BEV transform (meters)
            pixels_per_meter: BEV resolution
        """
        self.bev_x_range = bev_x_range
        self.bev_y_range = bev_y_range
        self.pixels_per_meter = pixels_per_meter

        # Compute BEV dimensions
        x_span = bev_x_range[1] - bev_x_range[0]
        y_span = bev_y_range[1] - bev_y_range[0]
        self.bev_height = int(x_span * pixels_per_meter)
        self.bev_width = int(y_span * pixels_per_meter)

        # Camera parameters (set via set_camera_params)
        self.camera_fov: Optional[float] = None
        self.camera_pitch: Optional[float] = None
        self.camera_height: Optional[float] = None

    def set_camera_params(
        self,
        fov: float,
        pitch: float,
        camera_height: float,
    ):
        """Set camera extrinsics for IPM transform."""
        self.camera_fov = fov
        self.camera_pitch = pitch
        self.camera_height = camera_height

    def _is_boundary_at_fov_edge(
        self,
        boundary: BoundaryResult,
        observed_mask: np.ndarray,
        is_left: bool,
        margin: int = 15,
    ) -> bool:
        """
        Check if a detected boundary is at the edge of the observed (FOV) region.

        A boundary at the FOV edge is not a real road boundary - it's just where
        our camera view ends. Such boundaries should be invalidated.

        Args:
            boundary: BoundaryResult from extract_boundaries
            observed_mask: Binary mask of observed area (1=observed, 0=outside FOV)
            is_left: True for left boundary (high row values)
            margin: Pixels from edge to consider as "at FOV edge"

        Returns:
            True if boundary is at FOV edge (should be invalidated)
        """
        if not boundary.detected or boundary.fitted_x is None:
            return False

        # Find the extent of the observed region
        # Sum observed pixels per row to find which rows have observations
        row_sums = np.sum(observed_mask, axis=1)
        observed_rows = np.where(row_sums > 0)[0]

        if len(observed_rows) == 0:
            return True  # No observed area

        min_observed_row = observed_rows.min()
        max_observed_row = observed_rows.max()

        # Get average boundary row position
        avg_boundary_row = np.mean(boundary.fitted_x)

        if is_left:
            # Left boundary has high row values (toward max_observed_row)
            # If boundary is within margin of top observed edge, it's at FOV edge
            return avg_boundary_row >= (max_observed_row - margin)
        else:
            # Right boundary has low row values (toward min_observed_row)
            # If boundary is within margin of bottom observed edge, it's at FOV edge
            return avg_boundary_row <= (min_observed_row + margin)

    def extract(self, semantic_img: np.ndarray) -> BoundaryExtractionResult:
        """
        Extract road boundaries from semantic segmentation image.

        Args:
            semantic_img: Mono8 semantic segmentation image

        Returns:
            BoundaryExtractionResult with left/right boundaries in base_link
        """
        # Import here to avoid circular dependency
        from mapping.utils.ipm_transform import convert_to_ipm

        result = BoundaryExtractionResult()

        if self.camera_fov is None:
            return result

        # 1. Create masks from semantic image
        # Drivable area = road + road_line
        drivable_mask = (
            (semantic_img == self.ROAD_ID) | (semantic_img == self.ROAD_LINE_ID)
        ).astype(np.uint8) * 255

        # Observed area = any pixel that has a semantic label (not unlabeled/0)
        # This prevents FOV edges from being detected as road boundaries
        observed_mask_raw = (semantic_img != 0).astype(np.uint8) * 255

        # 2. Transform both masks to BEV using IPM
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
            image=observed_mask_raw,
            fov=self.camera_fov,
            pitch=self.camera_pitch,
            camera_height=self.camera_height,
            bev_x_range=self.bev_x_range,
            bev_y_range=self.bev_y_range,
            pixels_per_meter=self.pixels_per_meter,
            interpolation=cv2.INTER_NEAREST,
        )

        # 3. Transform BEV to match boundary_extractor convention
        # convert_to_ipm output: rows=X (far at top), cols=Y (left at col 0)
        # boundary_extractor expects: rows=Y (right at top), cols=X (behind at col 0)
        # Transform: transpose, flip both axes
        bev_for_extractor = bev_drivable.T[::-1, ::-1]
        observed_for_extractor = bev_observed.T[::-1, ::-1]

        # Grid dimensions after transform
        grid_height, grid_width = bev_for_extractor.shape[:2]

        # Grid origin for boundary_to_base_link (in extractor convention)
        # After transform: row 0 = Y_min (right), col 0 = X_min (near)
        grid_origin_x = self.bev_x_range[0]  # X at col 0
        grid_origin_y = self.bev_y_range[0]  # Y at row 0

        # 4. Extract road boundaries from transformed BEV mask
        binary_mask = (bev_for_extractor > 127).astype(np.uint8)
        observed_mask = (observed_for_extractor > 127).astype(np.uint8)

        left, right, edge_img, l_diag, r_diag = extract_boundaries(binary_mask, observed_mask)

        # 5. Check if boundaries are at FOV edge (not real road boundaries)
        # A boundary at FOV edge should be invalidated so we fall back to prior knowledge
        fov_edge_margin = 15  # pixels from edge to consider as "at FOV edge"

        left_at_fov_edge = self._is_boundary_at_fov_edge(
            left, observed_mask, is_left=True, margin=fov_edge_margin
        )
        right_at_fov_edge = self._is_boundary_at_fov_edge(
            right, observed_mask, is_left=False, margin=fov_edge_margin
        )

        # 6. Convert pixel boundaries to base_link (skip if at FOV edge)
        left_pts = None
        if left.detected and not left_at_fov_edge:
            left_pts = boundary_to_base_link(
                left,
                grid_height=grid_height,
                grid_width=grid_width,
                origin_x=grid_origin_x,
                origin_y=grid_origin_y,
                resolution=1.0 / self.pixels_per_meter,
                sample_step=3,
            )

        right_pts = None
        if right.detected and not right_at_fov_edge:
            right_pts = boundary_to_base_link(
                right,
                grid_height=grid_height,
                grid_width=grid_width,
                origin_x=grid_origin_x,
                origin_y=grid_origin_y,
                resolution=1.0 / self.pixels_per_meter,
                sample_step=3,
            )

        result.left_boundary = left_pts
        result.right_boundary = right_pts
        result.valid = left_pts is not None or right_pts is not None

        # 7. Generate debug image
        result.debug_image = draw_boundary_overlay(
            bev_for_extractor, left, right, l_diag, r_diag, edge_img
        )

        return result
