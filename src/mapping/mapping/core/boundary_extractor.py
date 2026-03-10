"""
Road Boundary Extractor — Sliding Window on BEV Drivable-Area Mask
==================================================================

Extracts left and right continuous road boundaries from a binary Bird's-Eye
View (BEV) drivable area mask produced by Inverse Perspective Mapping.

The algorithm uses a highly constrained Sliding Window approach designed to
explicitly reject sudden road cutouts such as U-turns, intersections, and
missing mask data.

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
    """Container for a single road-boundary extraction result."""

    poly_coeffs: Optional[np.ndarray] = None  # 2nd-order [a, b, c]
    xs: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    ys: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    fitted_x: Optional[np.ndarray] = None     # Evaluated polynomial x(y)
    fitted_y: Optional[np.ndarray] = None     # y values used for evaluation
    y_min: Optional[float] = None             # min supported y used for fit/eval
    y_max: Optional[float] = None             # max supported y used for fit/eval
    detected: bool = False


# ───────────────────── helper: edge extraction ───────────────────

def _extract_boundary_edges(binary_mask: np.ndarray) -> np.ndarray:
    """
    Extract left/right boundary edges from a binary drivable area mask
    using a horizontal Sobel gradient and thresholding.

    Returns a binary edge image (uint8, 0/255).
    """
    # Ensure uint8 with values 0 / 255
    mask_u8 = (binary_mask.astype(np.uint8)) * 255

    # Horizontal gradient — detects vertical boundaries
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
        2-D binary image (1 = drivable, 0 = background).
    valid_mask : np.ndarray, optional
        2-D binary/boolean mask where 1/True means region was actually observed
        by camera/IPM (e.g., grid != UNKNOWN). If not provided, whole image is
        assumed valid.

    Returns
    -------
    left_result : BoundaryResult
    right_result : BoundaryResult
    edge_img : np.ndarray   – the intermediate edge image (for debug)
    left_diags : list[tuple] – (y_low, y_high, x_center, margin, is_frozen) per window
    right_diags : list[tuple] – (y_low, y_high, x_center, margin, is_frozen) per window
    """
    # ── Hardcoded parameters ────────────────────────────────────
    NWINDOWS = 15                  # number of sliding windows
    MARGIN = 40                    # half-width of search window (px)
    MINPIX = 30                    # min edge pixels to recenter
    MAX_LATERAL_SHIFT = 25         # px – momentum / max-shift rule
    LANE_WIDTH_TOLERANCE = 40      # px – allowed deviation from expected
    HISTOGRAM_BOTTOM_FRAC = 1 / 3  # fraction of image for base histogram
    POLY_ORDER = 2
    VALID_MASK_ERODE = 5           # px - remove FOV boundary artifacts

    # Grid axes: rows = Y (lateral), cols = X (forward).
    # The road extends along columns, but the sliding window sweeps
    # bottom-to-top along rows.  Transpose so forward becomes the row axis.
    binary_warped = binary_warped.T
    if valid_mask is None:
        valid_mask = np.ones_like(binary_warped, dtype=np.uint8)
    else:
        valid_mask = valid_mask.T.astype(np.uint8)

    h, w = binary_warped.shape[:2]

    # Erode observed region to avoid picking strong FOV border edges
    if VALID_MASK_ERODE > 0:
        kernel = np.ones((VALID_MASK_ERODE, VALID_MASK_ERODE), dtype=np.uint8)
        valid_mask = cv2.erode(valid_mask, kernel, iterations=1)

    # ── 1. Edge extraction ──────────────────────────────────────
    edge_img = _extract_boundary_edges(binary_warped)
    edge_img = (edge_img * valid_mask).astype(np.uint8)

    # ── 2. Histogram initialisation ─────────────────────────────
    hist_slice = edge_img[int(h * (1 - HISTOGRAM_BOTTOM_FRAC)):, :]
    histogram = np.sum(hist_slice, axis=0)

    midpoint = w // 2
    left_base = int(np.argmax(histogram[:midpoint]))
    right_base = int(np.argmax(histogram[midpoint:])) + midpoint

    # Derive expected lane width from starting positions
    expected_width = right_base - left_base
    if expected_width <= 0:
        expected_width = w // 2  # absolute fallback

    # ── 3. Sliding window setup ─────────────────────────────────
    nwindows = NWINDOWS
    margin = MARGIN
    minpix = MINPIX
    window_height = h // nwindows

    # Current x-centres
    left_current = left_base
    right_current = right_base

    # Collectors for valid boundary pixels
    left_xs: List[int] = []
    left_ys: List[int] = []
    right_xs: List[int] = []
    right_ys: List[int] = []

    # Diagnostics – each entry is (y_low, y_high, x_center, margin, is_frozen)
    left_diags: list = []
    right_diags: list = []

    nonzero = edge_img.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    # ── 4. Core loop — bottom to top ───────────────────────────
    for win_idx in range(nwindows):
        y_low = h - (win_idx + 1) * window_height
        y_high = h - win_idx * window_height

        # Window bounds
        left_x_low = left_current - margin
        left_x_high = left_current + margin
        right_x_low = right_current - margin
        right_x_high = right_current + margin

        # Identify non-zero pixels inside windows
        left_inds = (
            (nonzero_y >= y_low) & (nonzero_y < y_high) &
            (nonzero_x >= left_x_low) & (nonzero_x < left_x_high)
        )
        right_inds = (
            (nonzero_y >= y_low) & (nonzero_y < y_high) &
            (nonzero_x >= right_x_low) & (nonzero_x < right_x_high)
        )

        left_pix_x = nonzero_x[left_inds]
        left_pix_y = nonzero_y[left_inds]
        right_pix_x = nonzero_x[right_inds]
        right_pix_y = nonzero_y[right_inds]

        # Candidate new centres
        left_new = int(np.mean(left_pix_x)) if len(left_pix_x) >= minpix else left_current
        right_new = int(np.mean(right_pix_x)) if len(right_pix_x) >= minpix else right_current

        # ── Constraint checks ──────────────────────────────────
        left_frozen = False
        right_frozen = False

        # (a) Data starvation — already handled above (inherit previous)
        if len(left_pix_x) < minpix:
            left_frozen = True
        if len(right_pix_x) < minpix:
            right_frozen = True

        # (b) Momentum / max lateral shift
        if not left_frozen and abs(left_new - left_current) > MAX_LATERAL_SHIFT:
            left_new = left_current
            left_frozen = True
        if not right_frozen and abs(right_new - right_current) > MAX_LATERAL_SHIFT:
            right_new = right_current
            right_frozen = True

        # (c) Constant lane width enforcement
        proposed_width = right_new - left_new
        if proposed_width > expected_width + LANE_WIDTH_TOLERANCE:
            # One (or both) boundaries are diverging — freeze the diverger
            left_delta = abs(left_new - left_current)
            right_delta = abs(right_new - right_current)

            if left_delta > right_delta:
                # Left is diverging → mirror from stable right
                left_new = right_new - expected_width
                left_frozen = True
            else:
                # Right is diverging → mirror from stable left
                right_new = left_new + expected_width
                right_frozen = True

        # ── Accept updates ──────────────────────────────────────
        left_current = left_new
        right_current = right_new

        # Collect valid pixels
        if len(left_pix_x) > 0:
            left_xs.extend(left_pix_x.tolist())
            left_ys.extend(left_pix_y.tolist())
        if len(right_pix_x) > 0:
            right_xs.extend(right_pix_x.tolist())
            right_ys.extend(right_pix_y.tolist())

        # Record diagnostics
        left_diags.append((y_low, y_high, left_current, margin, left_frozen))
        right_diags.append((y_low, y_high, right_current, margin, right_frozen))

    # ── 5. Polynomial fitting ───────────────────────────────────
    left_result = _fit_boundary(left_xs, left_ys, h, POLY_ORDER)
    right_result = _fit_boundary(right_xs, right_ys, h, POLY_ORDER)

    return left_result, right_result, edge_img, left_diags, right_diags


# ───────────────────── curve fitting helper ──────────────────────

def _fit_boundary(
    xs: List[int],
    ys: List[int],
    img_height: int,
    order: int = 2,
) -> BoundaryResult:
    """Fit a polynomial x = f(y) to collected boundary pixels."""
    result = BoundaryResult()
    if len(xs) < order + 1:
        return result

    xs_arr = np.array(xs, dtype=np.float64)
    ys_arr = np.array(ys, dtype=np.float64)

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

    # Evaluate fitted curve only over supported y-range (no blind extrapolation)
    y_start = int(max(0, np.floor(result.y_min)))
    y_end = int(min(img_height - 1, np.ceil(result.y_max)))
    if y_end < y_start:
        return result

    plot_y = np.arange(y_start, y_end + 1, dtype=np.float64)
    plot_x = np.polyval(coeffs, plot_y)
    result.fitted_x = plot_x
    result.fitted_y = plot_y

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

    Windows are drawn:
    - **Green** for valid / recentered updates
    - **Red**   for frozen / constrained updates

    Fitted polynomial curves are drawn in **yellow** (left) and **cyan** (right).

    Each diag entry is a tuple: (y_low, y_high, x_center, margin, is_frozen).
    """
    # Transpose bev_img to match the transposed coordinate space used by
    # extract_boundaries (edge_img and diags are already in that space).
    if bev_img.ndim == 2:
        bev_img = bev_img.T
    else:
        bev_img = bev_img.transpose(1, 0, 2)

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

    # Draw sliding windows – each diag is (y_low, y_high, x_center, margin, is_frozen)
    for y_low, y_high, x_center, margin, is_frozen in left_diags + right_diags:
        colour = RED if is_frozen else GREEN
        pt1 = (x_center - margin, y_low)
        pt2 = (x_center + margin, y_high)
        cv2.rectangle(vis, pt1, pt2, colour, 2)

    # Draw fitted curves
    for result, colour in [(left_result, YELLOW), (right_result, CYAN)]:
        if result.detected and result.fitted_x is not None:
            pts = np.column_stack([
                result.fitted_x.astype(np.int32),
                result.fitted_y.astype(np.int32),
            ])
            # Clip to image bounds
            pts[:, 0] = np.clip(pts[:, 0], 0, vis.shape[1] - 1)
            cv2.polylines(vis, [pts], isClosed=False, color=colour, thickness=2)

    # Transpose back to original grid orientation (rows=lateral, cols=forward)
    return vis.transpose(1, 0, 2)


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
    Convert a fitted boundary from BEV-pixel coordinates to ``base_link``
    metric coordinates suitable for publishing as a ``nav_msgs/Path``.

    The BEV grid is centred on the vehicle:
        world_x = origin_x + col * resolution
        world_y = origin_y + row * resolution

    The polynomial is x = f(y) in *image* coordinates where:
        image row  → grid row  (same direction)
        image col  → grid col  (same direction)

    Parameters
    ----------
    result : BoundaryResult
        Fitted boundary from ``extract_boundaries``.
    grid_cells : int
        Grid dimension (assumed square).
    origin_x, origin_y : float
        Grid origin in base_link frame (metres).
    resolution : float
        Metres per cell.
    sample_step : int
        Sample every *n*-th row for the path to keep message size reasonable.

    Returns
    -------
    points : np.ndarray, shape (N, 2)
        Array of (x_base, y_base) in the vehicle frame, ordered from the
        bottom of the image (closest to vehicle) to the top (farthest).
        Returns *None* if the boundary was not detected.
    """
    if not result.detected or result.fitted_x is None or result.fitted_y is None:
        return None

    # Results are in *transposed* image coordinates:
    #   fitted_y  (sweep var)   = transposed rows = original grid cols (X / forward)
    #   fitted_x  (poly output) = transposed cols = original grid rows (Y / lateral)
    orig_cols = result.fitted_y[::sample_step]
    orig_rows = result.fitted_x[::sample_step]

    # Clip to valid grid range
    valid = (
        (orig_cols >= 0) & (orig_cols < grid_cells) &
        (orig_rows >= 0) & (orig_rows < grid_cells)
    )
    orig_cols = orig_cols[valid]
    orig_rows = orig_rows[valid]

    if len(orig_cols) == 0:
        return None

    # Grid (row, col) → metric (x, y) in base_link
    #   x_base = origin_x + col * resolution   (col encodes forward axis)
    #   y_base = origin_y + row * resolution   (row encodes lateral axis)
    x_base = origin_x + orig_cols * resolution
    y_base = origin_y + orig_rows * resolution

    points = np.column_stack([x_base, y_base])
    return points
