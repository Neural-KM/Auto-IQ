"""
Road Boundary Extractor — Sliding Window on BEV Drivable-Area Mask
==================================================================

Extracts left and right continuous road boundaries from a binary Bird's-Eye
View (BEV) local lane data mask produced by Inverse Perspective Mapping.

The algorithm uses a highly constrained Sliding Window approach designed to
explicitly reject sudden road cutouts such as U-turns, intersections, and
missing mask data.

BEV coordinate convention (direct from IPM, no transforms):
    Image shape: (bev_height, bev_width)
    - rows = X / forward:  row 0 = far (X_max), row max = near (X_min)
    - cols = Y / lateral:  col 0 = left (Y_max), col max = right (Y_min)

    To convert to base_link:
        x_base = X_max - row * resolution  (forward distance)
        y_base = Y_max - col * resolution  (lateral position, +left)

The sliding window sweeps along **rows** (forward axis, from near to far)
and tracks boundary **column** positions (lateral).

Designed for Level 4 autonomous driving on heavy-duty trucks where lane
boundary stability is safety-critical.

Changelog:
---------
2026-04-05: Refactored to work directly with IPM output convention:
    - No transpose or flip operations
    - Sweep along rows (forward), track column positions (lateral)
2026-04-01: Fixed sweep range bug
2026-03-25: Improved curve handling

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

    Convention (matching IPM output):
        xs  = collected **column** positions (lateral / Y)
        ys  = collected **row** positions    (forward / X)
        fitted_x = polynomial-evaluated column values (lateral)
        fitted_y = row values used for evaluation (forward)
    """

    poly_coeffs: Optional[np.ndarray] = None  # 2nd-order [a, b, c]
    xs: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    ys: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    fitted_x: Optional[np.ndarray] = None     # col values  (lateral)
    fitted_y: Optional[np.ndarray] = None     # row values  (forward)
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    detected: bool = False


# ───────────────────── helper: edge extraction ───────────────────

def _extract_boundary_edges(binary_mask: np.ndarray) -> np.ndarray:
    """
    Extract road-boundary edges from a binary drivable-area mask.

    Road boundaries run roughly along the row (forward) axis.
    Transitions happen along the col (lateral) axis, so we use a
    **horizontal** Sobel gradient (dx = 1).

    Returns a binary edge image (uint8, 0 / 255).
    """
    mask_u8 = binary_mask.astype(np.uint8) * 255

    # Horizontal gradient — detects edges where intensity changes across cols
    sobel_x = cv2.Sobel(mask_u8, cv2.CV_64F, 1, 0, ksize=3)
    edge = np.uint8(np.absolute(sobel_x) > 30) * 255
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
        2-D binary image, shape (bev_height, bev_width).  1 = drivable, 0 = background.
        Uses IPM convention: rows = X/forward (row 0 = far), cols = Y/lateral (col 0 = left).
    valid_mask : np.ndarray, optional
        Same shape.  1 = observed by camera, 0 = unseen / UNKNOWN.

    Returns
    -------
    left_result  : BoundaryResult   (low-col boundary = left of vehicle)
    right_result : BoundaryResult   (high-col boundary = right of vehicle)
    edge_img     : np.ndarray       (intermediate edge image for debug)
    left_diags   : list[tuple]      (r_lo, r_hi, c_center, margin, frozen)
    right_diags  : list[tuple]      (r_lo, r_hi, c_center, margin, frozen)
    """
    # ── Hardcoded parameters ────────────────────────────────────
    NWINDOWS = 15
    MARGIN = 60                    # half-width of search window (cols)
    MINPIX = 10                    # min edge pixels to recenter
    MAX_LATERAL_SHIFT = 45         # max col-shift between consecutive windows
    LANE_WIDTH_TOLERANCE = 60      # allowed col-width deviation
    POLY_ORDER = 2
    VALID_MASK_ERODE = 5

    h, w = binary_warped.shape[:2]  # h = rows (forward), w = cols (lateral)

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

    # ── 2. Histogram initialization ─────────────────────────────
    # Sum edge pixels across ALL rows → one value per column.
    # Peaks indicate the two road boundary columns.
    histogram = np.sum(edge_img, axis=0).astype(np.float64)

    midpoint = w // 2
    # Col 0 = left side of vehicle → left boundary has col < midpoint
    # Col w = right side of vehicle → right boundary has col > midpoint
    left_base = int(np.argmax(histogram[:midpoint]))
    right_base = int(np.argmax(histogram[midpoint:])) + midpoint

    # Expected road width in columns
    expected_width = right_base - left_base
    if expected_width <= 0:
        expected_width = w // 2

    # ── 3. Sliding window setup ─────────────────────────────────
    # Sweep along rows from bottom (near) to top (far).
    # Row 0 = far, Row h-1 = near.
    # We start from near (high row indices) and move toward far (low row indices).
    sweep_start = max(10, int(h * 0.05))  # Skip 5% at far edge (noisy)
    sweep_end = h - max(10, int(h * 0.05))  # Skip 5% at near edge
    sweep_range = sweep_end - sweep_start
    window_h = max(1, sweep_range // NWINDOWS)
    margin = MARGIN
    minpix = MINPIX

    left_current = left_base
    right_current = right_base

    # Collectors (xs = col positions/lateral, ys = row positions/forward)
    left_xs: List[int] = []
    left_ys: List[int] = []
    right_xs: List[int] = []
    right_ys: List[int] = []

    left_diags: list = []
    right_diags: list = []

    nonzero = edge_img.nonzero()
    nz_row = np.array(nonzero[0])
    nz_col = np.array(nonzero[1])

    # ── 4. Core loop — near to far along rows ───────────────────
    for win_idx in range(NWINDOWS):
        # Sweep from near (high row) to far (low row)
        r_hi = sweep_end - win_idx * window_h
        r_lo = sweep_end - (win_idx + 1) * window_h

        # Left boundary search window (low cols)
        c_left_lo = left_current - margin
        c_left_hi = left_current + margin

        # Right boundary search window (high cols)
        c_right_lo = right_current - margin
        c_right_hi = right_current + margin

        # Edge pixels inside each window
        left_inds = (
            (nz_row >= r_lo) & (nz_row < r_hi) &
            (nz_col >= c_left_lo) & (nz_col < c_left_hi)
        )
        right_inds = (
            (nz_row >= r_lo) & (nz_row < r_hi) &
            (nz_col >= c_right_lo) & (nz_col < c_right_hi)
        )

        left_pix_r = nz_row[left_inds]
        left_pix_c = nz_col[left_inds]
        right_pix_r = nz_row[right_inds]
        right_pix_c = nz_col[right_inds]

        # Candidate new column centers
        left_new = int(np.mean(left_pix_c)) if len(left_pix_c) >= minpix else left_current
        right_new = int(np.mean(right_pix_c)) if len(right_pix_c) >= minpix else right_current

        # ── Constraint checks ──────────────────────────────────
        left_frozen = len(left_pix_c) < minpix
        right_frozen = len(right_pix_c) < minpix

        # Momentum / max lateral shift
        if not left_frozen and abs(left_new - left_current) > MAX_LATERAL_SHIFT:
            left_new = left_current
            left_frozen = True
        if not right_frozen and abs(right_new - right_current) > MAX_LATERAL_SHIFT:
            right_new = right_current
            right_frozen = True

        # Constant road width enforcement
        proposed_width = right_new - left_new
        if proposed_width > expected_width + LANE_WIDTH_TOLERANCE:
            left_delta = abs(left_new - left_current)
            right_delta = abs(right_new - right_current)
            if left_delta > right_delta:
                left_new = right_new - expected_width
                left_frozen = True
            else:
                right_new = left_new + expected_width
                right_frozen = True

        # ── Accept updates ──────────────────────────────────────
        left_current = left_new
        right_current = right_new

        if len(left_pix_c) > 0:
            left_xs.extend(left_pix_c.tolist())
            left_ys.extend(left_pix_r.tolist())
        if len(right_pix_c) > 0:
            right_xs.extend(right_pix_c.tolist())
            right_ys.extend(right_pix_r.tolist())

        # Diagnostics: (r_lo, r_hi, c_center, margin, frozen)
        left_diags.append((r_lo, r_hi, left_current, margin, left_frozen))
        right_diags.append((r_lo, r_hi, right_current, margin, right_frozen))

    # ── 5. Polynomial fitting  col = f(row) ────────────────────
    left_result = _fit_boundary(left_xs, left_ys, h, POLY_ORDER)
    right_result = _fit_boundary(right_xs, right_ys, h, POLY_ORDER)

    return left_result, right_result, edge_img, left_diags, right_diags


# ───────────────────── curve fitting helper ──────────────────────

def _fit_boundary(
    xs: List[int],
    ys: List[int],
    row_extent: int,
    order: int = 2,
) -> BoundaryResult:
    """Fit a polynomial  col = f(row)  to collected boundary pixels.

    xs = col positions (dependent), ys = row positions (independent).
    """
    result = BoundaryResult()
    if len(xs) < order + 1:
        return result

    xs_arr = np.array(xs, dtype=np.float64)   # cols (lateral)
    ys_arr = np.array(ys, dtype=np.float64)   # rows (forward)

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

    # Evaluate only within supported row range
    r_start = int(max(0, np.floor(result.y_min)))
    r_end = int(min(row_extent - 1, np.ceil(result.y_max)))
    if r_end < r_start:
        return result

    plot_row = np.arange(r_start, r_end + 1, dtype=np.float64)
    plot_col = np.polyval(coeffs, plot_row)
    result.fitted_x = plot_col    # col values  (lateral)
    result.fitted_y = plot_row    # row values  (forward)

    return result


# ───────────────────── visualization ─────────────────────────────

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

    BEV convention: row 0 = far (top), col 0 = left.
    No transforms applied - visualization matches BEV coordinate system.

    Windows: green = valid (active tracking), red = frozen / constrained.
    Curves: yellow = left boundary, cyan = right boundary.

    Each diag entry is a tuple: (r_lo, r_hi, c_center, margin, is_frozen).
    """
    # Build 3-channel canvas
    if bev_img.ndim == 2:
        vis = cv2.cvtColor(bev_img, cv2.COLOR_GRAY2BGR)
    else:
        vis = bev_img.copy()

    # Blend edge image (faintly)
    if edge_img is not None:
        edge_bgr = cv2.cvtColor(edge_img, cv2.COLOR_GRAY2BGR)
        vis = cv2.addWeighted(vis, 0.8, edge_bgr, 0.3, 0)

    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    YELLOW = (0, 255, 255)
    CYAN = (255, 255, 0)

    # Draw sliding windows with semi-transparent fill
    overlay = vis.copy()
    for r_lo, r_hi, c_center, margin, is_frozen in left_diags + right_diags:
        colour = RED if is_frozen else GREEN
        # Rectangle: (x1, y1) to (x2, y2) where x=col, y=row
        pt1 = (int(c_center - margin), int(r_lo))
        pt2 = (int(c_center + margin), int(r_hi))
        cv2.rectangle(overlay, pt1, pt2, colour, -1)
        cv2.rectangle(vis, pt1, pt2, colour, 3)

    vis = cv2.addWeighted(overlay, 0.3, vis, 0.7, 0)

    # Draw fitted curves
    for result, colour in [(left_result, YELLOW), (right_result, CYAN)]:
        if result.detected and result.fitted_x is not None:
            pts = np.column_stack([
                result.fitted_x.astype(np.int32),   # col → x
                result.fitted_y.astype(np.int32),   # row → y
            ])
            pts[:, 0] = np.clip(pts[:, 0], 0, vis.shape[1] - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, vis.shape[0] - 1)
            cv2.polylines(vis, [pts], isClosed=False, color=colour, thickness=3)

    return vis


# ───────────────────── BEV-pixel → base_link conversion ─────────

def boundary_to_base_link(
    result: BoundaryResult,
    bev_height: int,
    bev_width: int,
    x_max: float,
    y_max: float,
    resolution: float,
    sample_step: int = 5,
) -> Optional[np.ndarray]:
    """
    Convert a fitted boundary from BEV-pixel coordinates to ``base_link``
    metric coordinates.

    BEV convention (from IPM):
        row 0 = far (X_max), row max = near (X_min)
        col 0 = left (Y_max), col max = right (Y_min)

    Conversion:
        x_base = X_max - row * resolution  (forward)
        y_base = Y_max - col * resolution  (lateral, +left)

    In BoundaryResult:
        fitted_y = row values   (forward / X)
        fitted_x = col values   (lateral / Y)

    Returns
    -------
    points : np.ndarray, shape (N, 2)
        (x_base, y_base) in the vehicle frame, ordered near → far.
        *None* if the boundary was not detected.
    """
    if not result.detected or result.fitted_x is None or result.fitted_y is None:
        return None

    rows = result.fitted_y[::sample_step]    # forward
    cols = result.fitted_x[::sample_step]    # lateral

    # Clip to valid BEV range
    valid = (
        (rows >= 0) & (rows < bev_height) &
        (cols >= 0) & (cols < bev_width)
    )
    rows = rows[valid]
    cols = cols[valid]

    if len(rows) == 0:
        return None

    # Convert to base_link coordinates
    x_base = x_max - rows * resolution
    y_base = y_max - cols * resolution

    # Sort by x (forward distance) ascending (near to far)
    sorted_idx = np.argsort(x_base)
    x_base = x_base[sorted_idx]
    y_base = y_base[sorted_idx]

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
        drivable_mask: np.ndarray,
        is_left: bool,
        margin: int = 15,
    ) -> bool:
        """
        Check if a detected boundary is at the edge of the observed (FOV) region
        AND drivable area extends to that edge (meaning road continues beyond view).

        A boundary should only be invalidated if:
        1. It's near the FOV edge, AND
        2. Drivable area extends all the way to the FOV edge (road continues beyond)

        If the road actually ends before the FOV edge (non-drivable area exists
        between boundary and FOV edge), the boundary is REAL and should be kept.

        Args:
            boundary: BoundaryResult from extract_boundaries
            observed_mask: Binary mask of observed area (1=observed, 0=outside FOV)
            drivable_mask: Binary mask of drivable area (1=road, 0=not road)
            is_left: True for left boundary (low col values)
            margin: Pixels from edge to consider as "at FOV edge"

        Returns:
            True if boundary is at FOV edge AND road continues beyond (should be invalidated)
        """
        if not boundary.detected or boundary.fitted_x is None:
            return False

        # Find the extent of the observed region
        col_sums = np.sum(observed_mask, axis=0)
        observed_cols = np.where(col_sums > 0)[0]

        if len(observed_cols) == 0:
            return True  # No observed area

        min_observed_col = observed_cols.min()
        max_observed_col = observed_cols.max()

        # Get average boundary col position
        avg_boundary_col = np.mean(boundary.fitted_x)

        # Check if boundary is near FOV edge
        near_fov_edge = False
        if is_left:
            near_fov_edge = avg_boundary_col <= (min_observed_col + margin)
        else:
            near_fov_edge = avg_boundary_col >= (max_observed_col - margin)

        if not near_fov_edge:
            return False  # Boundary not near FOV edge, definitely real

        # Boundary IS near FOV edge. Now check if drivable area extends to FOV edge.
        # If there's non-drivable area between boundary and FOV edge, it's a real boundary.
        drivable_col_sums = np.sum(drivable_mask, axis=0)

        if is_left:
            # Check columns between FOV edge and boundary
            edge_cols = range(min_observed_col, int(avg_boundary_col))
            if len(edge_cols) > 0:
                drivable_at_edge = drivable_col_sums[min_observed_col:int(avg_boundary_col)]
                # If there's significant drivable area at the FOV edge, road continues beyond
                # If minimal/no drivable area at edge, this is a real road boundary
                if len(drivable_at_edge) > 0 and np.max(drivable_at_edge) < observed_mask.shape[0] * 0.1:
                    return False  # Little drivable area at edge = real boundary
        else:
            # Check columns between boundary and FOV edge
            edge_cols = range(int(avg_boundary_col), max_observed_col + 1)
            if len(edge_cols) > 0:
                drivable_at_edge = drivable_col_sums[int(avg_boundary_col):max_observed_col + 1]
                if len(drivable_at_edge) > 0 and np.max(drivable_at_edge) < observed_mask.shape[0] * 0.1:
                    return False  # Little drivable area at edge = real boundary

        return True  # Drivable area extends to FOV edge, likely false boundary

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
        observed_mask_raw = (semantic_img != 0).astype(np.uint8) * 255

        # 2. Transform both masks to BEV using IPM (no additional transforms needed)
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

        # 3. Extract road boundaries directly from BEV (no transform needed)
        bev_height, bev_width = bev_drivable.shape[:2]
        binary_mask = (bev_drivable > 127).astype(np.uint8)
        observed_mask = (bev_observed > 127).astype(np.uint8)

        left, right, edge_img, l_diag, r_diag = extract_boundaries(binary_mask, observed_mask)

        # 4. Check if boundaries are at FOV edge (not real road boundaries)
        # Use 5 meters as margin (scaled to pixels)
        fov_edge_margin = int(5.0 * self.pixels_per_meter)

        left_at_fov_edge = self._is_boundary_at_fov_edge(
            left, observed_mask, binary_mask, is_left=True, margin=fov_edge_margin
        )
        right_at_fov_edge = self._is_boundary_at_fov_edge(
            right, observed_mask, binary_mask, is_left=False, margin=fov_edge_margin
        )

        # 5. Convert pixel boundaries to base_link (skip if at FOV edge)
        resolution = 1.0 / self.pixels_per_meter

        left_pts = None
        if left.detected and not left_at_fov_edge:
            left_pts = boundary_to_base_link(
                left,
                bev_height=bev_height,
                bev_width=bev_width,
                x_max=self.bev_x_range[1],
                y_max=self.bev_y_range[1],
                resolution=resolution,
                sample_step=3,
            )

        right_pts = None
        if right.detected and not right_at_fov_edge:
            right_pts = boundary_to_base_link(
                right,
                bev_height=bev_height,
                bev_width=bev_width,
                x_max=self.bev_x_range[1],
                y_max=self.bev_y_range[1],
                resolution=resolution,
                sample_step=3,
            )

        result.left_boundary = left_pts
        result.right_boundary = right_pts
        result.valid = left_pts is not None or right_pts is not None

        # 6. Generate debug image
        result.debug_image = draw_boundary_overlay(
            bev_drivable, left, right, l_diag, r_diag, edge_img
        )

        return result
