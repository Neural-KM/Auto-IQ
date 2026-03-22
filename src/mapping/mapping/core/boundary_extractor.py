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
    MARGIN = 40                    # half-height of search window (rows)
    MINPIX = 10                    # min edge pixels to recenter
    MAX_LATERAL_SHIFT = 25         # max row-shift between consecutive windows
    LANE_WIDTH_TOLERANCE = 40      # allowed row-width deviation
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
    # Sweep along columns from centre of grid (vehicle) to far edge.
    sweep_start = w // 2
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

    Windows: green = valid,  red = frozen / constrained.
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

    # Draw sliding windows — cv2 uses (x, y) = (col, row)
    for c_lo, c_hi, r_center, margin, is_frozen in left_diags + right_diags:
        colour = RED if is_frozen else GREEN
        pt1 = (int(c_lo), int(r_center - margin))
        pt2 = (int(c_hi), int(r_center + margin))
        cv2.rectangle(vis, pt1, pt2, colour, 2)

    # Draw fitted curves
    for result, colour in [(left_result, YELLOW), (right_result, CYAN)]:
        if result.detected and result.fitted_x is not None:
            pts = np.column_stack([
                result.fitted_y.astype(np.int32),   # col → x
                result.fitted_x.astype(np.int32),   # row → y
            ])
            pts[:, 0] = np.clip(pts[:, 0], 0, vis.shape[1] - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, vis.shape[0] - 1)
            cv2.polylines(vis, [pts], isClosed=False, color=colour, thickness=2)

    # ── Display rotation ────────────────────────────────────────
    # Native grid: rows top→bot = right→left,  cols left→right = behind→ahead.
    # Desired:     top = ahead,  right = right.
    # 90° CCW then horizontal flip achieves this.
    vis = cv2.flip(cv2.rotate(vis, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)

    return vis


# ───────────────────── BEV-pixel → base_link conversion ─────────

def boundary_to_base_link(
    result: BoundaryResult,
    grid_cells: int,
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
        (cols >= 0) & (cols < grid_cells) &
        (rows >= 0) & (rows < grid_cells)
    )
    cols = cols[valid]
    rows = rows[valid]

    if len(cols) == 0:
        return None

    # Direct mapping — no transpose to undo
    x_base = origin_x + cols * resolution
    y_base = origin_y + rows * resolution

    return np.column_stack([x_base, y_base])
