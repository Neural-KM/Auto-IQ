"""
Lane Detector — Adaptive Sliding Window with Temporal Tracking
==============================================================

Detects lane markings from semantic segmentation using adaptive-size sliding
windows that track lanes across frames with heavy temporal smoothing.

Pipeline:
---------
1. Extract road_line pixels from semantic image (class ID = 24)
2. Transform to BEV using Inverse Perspective Mapping (IPM)
3. If tracking: use previous frame's polynomial to predict window centers
   If detecting: use histogram peaks to find initial lane positions
4. Adaptive window sizing: small when locked, grows when searching
5. Robust polynomial fitting with outlier rejection
6. Heavy temporal smoothing (α = 0.7) for stability

BEV Coordinate System:
----------------------
- Origin: Bottom-center of BEV image (vehicle position)
- X-axis (columns): Lateral position, left is higher column index
- Y-axis (rows): Forward distance, row 0 = far, row H-1 = near vehicle
- Polynomials: x = f(y) where y is row index (y increases downward)

Mathematical Model:
-------------------
Lanes are modeled as parabolas: x(y) = a·y² + b·y + c

Where:
- a: Curvature coefficient [pixels / row²]
- b: Linear slope [pixels / row]
- c: X-intercept at row 0 [pixels]

Adaptive Window Algorithm:
--------------------------
The key insight is to separate TRACKING (have previous fit) from DETECTION
(no previous fit). This prevents false curve detection on initialization.

1. TRACKING MODE (previous fit available):
   - Predict window center from smoothed polynomial: x = a·y² + b·y + c
   - Start with SMALL window (margin_min = 25 pixels)
   - If enough pixels found → stay small (locked on lane)
   - If too few pixels → GROW window by growth_step until max_margin
   - This keeps windows tight on the lane, preventing drift into neighbors

2. DETECTION MODE (no previous fit):
   - Use bottom-half histogram to find initial peaks
   - Assume STRAIGHT lanes initially (a = 0, b = 0)
   - Use medium-sized windows for initial detection
   - Let temporal smoothing gradually learn any curvature

3. WINDOW SIZE DYNAMICS:
   For each window independently:

   margin[w] = margin_min                    if pixels >= threshold
             = min(margin[w] + growth, max)  if pixels < threshold

   This creates a "lock-on" behavior: once a lane is found, windows stay
   tight. If the lane is lost (occlusion, gap), windows widen to search.

4. TEMPORAL SMOOTHING:
   Polynomial coefficients are smoothed with exponential moving average:

   coeffs_smoothed = α · coeffs_prev + (1 - α) · coeffs_current

   With α = 0.7, the filter has ~3 frame time constant, providing stability
   while still responding to actual curves.

5. LANE ASSOCIATION:
   Match current detections to previous lanes by base position (x at bottom).
   This ensures temporal smoothing connects the same physical lane.

Quality Criteria:
-----------------
- Minimum 40 pixels for polynomial fit
- Maximum RMSE: 12 pixels (tighter than before)
- Maximum |a|: 0.015 (gentle curves only - let smoothing handle sharper)
- Minimum y-span: 30% of BEV height

Why This Works Better:
----------------------
1. No false curvature from noisy histogram peaks
2. Windows stay tight → don't capture neighboring lanes
3. Temporal smoothing learns true curvature gradually
4. Adaptive sizing handles occlusions gracefully

Changelog:
----------
2026-03-25: v1 - Initial curve-aware implementation
2026-03-25: v2 - Polynomial-guided windows
2026-03-25: v3 - Adaptive window sizing with temporal tracking:
    - Removed initial curve estimation from histogram (caused false curves)
    - Dynamic window margins: small when locked, grows when searching
    - Heavy temporal smoothing (α = 0.7) for stability
    - Tracking vs detection mode separation
2026-03-25: v3.1 - Improved dashed line detection:
    - Lowered histogram thresholds (150 min, prominence 80)
    - Reduced min_pixels from 40→25, min_pixels_locked from 20→8
    - Increased dilation (5x5 kernel, 2 iterations) to bridge gaps
    - Expanded histogram band from 30%→40% of BEV height
2026-03-25: v3.2 - Hybrid tracking + detection:
    - Run histogram detection EVERY frame (not just when no tracks)
    - New peaks not covered by existing tracks spawn new track candidates
    - Dashed lines that are missed in gaps get picked up in subsequent frames
2026-03-25: v3.3 - Multi-lane support tuning:
    - Reduced margins (min=12, max=30, initial=20) to avoid capturing adjacent lanes
    - Lighter dilation (3x3 kernel, 1 iteration) to preserve lane separation
    - Reduced histogram min_distance to 4% (~24px) for closer lane detection
    - Tighter association distance (35px) and "too close" threshold (18px)
    - Support up to 8 lanes
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any

import numpy as np
import cv2

from scipy.signal import find_peaks

from perception.data_classes.semantic_labels import SEMANTIC_BY_NAME
from mapping.data_classes.lane import LaneDetection
from mapping.utils.ipm_transform import convert_to_ipm

# Semantic class IDs
ROAD_ID = SEMANTIC_BY_NAME['road']
ROAD_LINE_ID = SEMANTIC_BY_NAME['road_line']


# ============================================================================
# Configuration
# ============================================================================


@dataclass(frozen=True)
class BEVConfig:
    """Bird's Eye View transform parameters."""
    bev_width: int = 600
    x_range: Tuple[float, float] = (5.0, 35.0)     # forward distance range (meters)
    y_range: Tuple[float, float] = (-15.0, 15.0)   # lateral range (meters)


@dataclass(frozen=True)
class OutputConfig:
    """Output paths configuration."""
    default_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[4] / 'data' / 'lane_detection')
    sensor_config: Path = field(default_factory=lambda: Path(__file__).resolve().parents[4] / 'config' / 'carla_sensors.yaml')


@dataclass(frozen=True)
class LaneDetectorConfig:
    """Main configuration for lane detector module."""
    bev: BEVConfig = field(default_factory=BEVConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


CONFIG = LaneDetectorConfig()


# ============================================================================
# Lane Detector
# ============================================================================


class LaneDetector:
    """
    Adaptive lane detection with temporal tracking.

    Uses dynamic window sizing and heavy temporal smoothing for stable
    lane tracking. See module docstring for algorithm details.
    """

    def __init__(self, image_width: int, image_height: int):
        """
        Initialize lane detector.

        Args:
            image_width: Camera image width in pixels
            image_height: Camera image height in pixels
        """
        self.width = image_width
        self.height = image_height

        # Camera extrinsics — set via set_camera_params()
        self.fov: Optional[float] = None
        self.pitch: Optional[float] = None
        self.camera_height: Optional[float] = None

        # BEV configuration
        self.bev_width = CONFIG.bev.bev_width
        self.bev_x_range = CONFIG.bev.x_range
        self.bev_y_range = CONFIG.bev.y_range
        self.pixels_per_meter = self.bev_width / (self.bev_y_range[1] - self.bev_y_range[0])

        x_span = self.bev_x_range[1] - self.bev_x_range[0]
        self.bev_height = int(x_span * self.pixels_per_meter)

        # Sliding window parameters
        self.n_windows = 12

        # Adaptive margin parameters (tuned for multi-lane detection)
        # Standard lane ~3.5m = ~70px at 20px/m, marking ~0.15m = ~3px
        self.margin_min = 12            # Tight margin when locked (< half lane width)
        self.margin_max = 30            # Max margin stays within own lane
        self.margin_initial = 20        # Conservative initial margin
        self.margin_growth = 6          # Smaller growth steps
        self.min_pixels_locked = 6      # Pixels needed to consider "locked" (low for dashed lines)

        # Polynomial fitting constraints (tighter for tracking)
        self.max_quadratic = 0.015      # Maximum |a| coefficient
        self.max_rmse = 12.0            # Maximum fit RMSE (pixels)
        self.min_pixels = 25            # Minimum pixels for fit (low for dashed lines)
        self.min_y_span_frac = 0.25     # Minimum vertical span (relaxed)

        # Temporal smoothing (heavy smoothing for stability)
        self.smoothing_alpha = 0.7      # Higher = more smoothing
        self.lane_width_smoothing = 0.95

        # State tracking
        self.left_lane_x: Optional[int] = None
        self.right_lane_x: Optional[int] = None
        self.lane_center = self.width // 2
        self.lane_width_pixels: Optional[int] = None
        self.avg_lane_width: Optional[float] = None

        # Tracked lanes: list of {coeffs, base_x, confidence, frames_tracked}
        self.tracked_lanes: List[Dict[str, Any]] = []
        self.max_lanes = 8              # Support more lanes
        self.max_association_dist = 35  # Tighter association for closely-spaced lanes

        # Debug images
        self.debug_images = {
            'bev_lane_mask': None,
            'lane_fit': None
        }

    def detect(self, semantic_image: np.ndarray) -> LaneDetection:
        """
        Detect lanes from semantic segmentation image.

        Uses adaptive windows with temporal tracking:
        - If tracked lanes exist: use their polynomials to predict window centers
        - If no tracked lanes: detect from histogram, assume straight

        Args:
            semantic_image: 2D numpy array of semantic labels (H x W, mono8)

        Returns:
            LaneDetection with polynomial fits for all detected lanes
        """
        if semantic_image is None:
            raise ValueError('semantic_image must not be None')

        _, w = semantic_image.shape[:2]

        # Extract road_line pixels
        lane_mask = (semantic_image == ROAD_LINE_ID).astype(np.uint8) * 255

        # Transform to bird's eye view
        bev_lane, _, _ = convert_to_ipm(
            image=lane_mask,
            fov=self.fov,
            pitch=self.pitch,
            camera_height=self.camera_height,
            bev_x_range=self.bev_x_range,
            bev_y_range=self.bev_y_range,
            pixels_per_meter=self.pixels_per_meter,
            interpolation=cv2.INTER_NEAREST
        )

        # Lighter dilation to avoid merging adjacent lane markings
        # Use smaller kernel and single iteration for multi-lane scenarios
        bev_lane = cv2.dilate(
            bev_lane,
            np.ones((3, 3), dtype=np.uint8),  # Smaller kernel
            iterations=1,  # Single iteration to preserve lane separation
        )
        self.debug_images['bev_lane_mask'] = bev_lane

        # Adaptive lane detection with tracking
        lane_fits, lane_fit_img = self._detect_lanes_adaptive(bev_lane)
        self.debug_images['lane_fit'] = lane_fit_img

        return self._compute_lane_metrics_multilane(lane_fits, w)

    # ========================================================================
    # Initial Lane Detection (No Previous Tracking)
    # ========================================================================

    def _detect_initial_lanes(self, bev_mask: np.ndarray) -> List[int]:
        """
        Detect initial lane positions using bottom-half histogram.

        Only used when no tracked lanes exist. Returns x-positions of peaks
        at the bottom of the BEV image, assuming STRAIGHT lanes initially.

        Args:
            bev_mask: Binary BEV mask of lane pixels

        Returns:
            List of x-positions (column indices) where lanes are detected
        """
        bev_h, bev_w = bev_mask.shape[:2]

        # Use bottom 40% of image for initial detection (larger band for dashed lines)
        bottom_fraction = 0.40
        bottom_rows = int(bev_h * bottom_fraction)
        bottom_band = bev_mask[bev_h - bottom_rows:, :]

        # Compute histogram
        hist = np.sum(bottom_band, axis=0).astype(np.float32)

        # Smooth histogram
        kernel_size = max(15, int(bev_w * 0.025))
        if kernel_size % 2 == 0:
            kernel_size += 1
        hist_smooth = cv2.GaussianBlur(hist.reshape(1, -1), (kernel_size, 1), 0).flatten()

        # Find peaks - tuned for multi-lane detection
        min_distance = int(bev_w * 0.04)  # Minimum 4% of width (~24px) allows closer lanes
        threshold = max(np.max(hist_smooth) * 0.10, 100)  # Lower threshold for dashed lines

        peaks, _ = find_peaks(
            hist_smooth,
            height=threshold,
            distance=min_distance,
            prominence=50  # Lower prominence to detect closely-spaced lanes
        )

        return list(peaks)

    # ========================================================================
    # Adaptive Lane Detection with Tracking
    # ========================================================================

    def _detect_lanes_adaptive(self, bev_lane_mask: np.ndarray):
        """
        Detect lanes using adaptive windows with temporal tracking.

        Algorithm:
        1. ALWAYS run histogram to find ALL peaks (potential lanes)
        2. For tracked lanes: use previous polynomial + adaptive windows
        3. For new peaks not near tracked lanes: try to initialize new track
        4. This hybrid approach maintains stable tracking while continuously
           looking for new lanes (important for dashed lines with gaps)
        """
        bev_h, bev_w = bev_lane_mask.shape[:2]

        # Create visualization
        vis_img = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
        vis_img[bev_lane_mask > 127] = [255, 255, 0]  # Yellow lane pixels
        overlay = vis_img.copy()

        # Get nonzero pixel coordinates
        nonzero = bev_lane_mask.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        if len(nonzero_x) == 0:
            cv2.line(vis_img, (bev_w // 2, 0), (bev_w // 2, bev_h), (255, 255, 255), 1)
            return [], vis_img

        # ALWAYS detect peaks from histogram (to find new lanes)
        peaks = self._detect_initial_lanes(bev_lane_mask)

        lane_candidates = []

        # Step 1: Process existing tracked lanes (tracking mode)
        if len(self.tracked_lanes) > 0:
            tracking_candidates = self._run_tracking_mode(
                nonzero_x, nonzero_y, bev_h, bev_w, overlay, vis_img
            )
            lane_candidates.extend(tracking_candidates)

        # Step 2: Check for NEW peaks not covered by tracked lanes
        y_eval = bev_h - 10
        for peak_x in peaks:
            # Check if this peak is already covered by a tracked lane
            is_covered = False
            for tracked in self.tracked_lanes:
                tracked_x = np.polyval(tracked['coeffs'], y_eval)
                if abs(peak_x - tracked_x) < self.margin_max:  # Use margin_max for coverage check
                    is_covered = True
                    break

            # Also check if covered by a candidate we just found
            for cand in lane_candidates:
                cand_x = np.polyval(cand['coeffs'], y_eval)
                if abs(peak_x - cand_x) < self.margin_max:  # Use margin_max for coverage check
                    is_covered = True
                    break

            if not is_covered:
                # Try to detect this as a new lane
                new_cand = self._detect_single_lane(
                    peak_x, nonzero_x, nonzero_y, bev_h, bev_w, overlay, vis_img
                )
                if new_cand is not None:
                    lane_candidates.append(new_cand)

        # Update tracked lanes with temporal smoothing
        self._update_tracked_lanes(lane_candidates, bev_h)

        # Prepare output fits from tracked lanes
        lane_fits = []
        for tracked in self.tracked_lanes:
            if tracked['confidence'] >= 0.3:
                lane_fits.append({
                    'coeffs': tracked['coeffs'],
                    'base_x': int(np.polyval(tracked['coeffs'], bev_h - 10)),
                    'pixel_count': tracked.get('pixel_count', 0),
                })

        # Draw fitted curves for tracked lanes
        lane_colors = [
            (255, 80, 80), (80, 255, 80), (80, 80, 255),
            (255, 200, 80), (255, 80, 255), (80, 255, 255),
        ]
        for i, tracked in enumerate(self.tracked_lanes):
            if tracked['confidence'] >= 0.3:
                color = lane_colors[i % len(lane_colors)]
                plot_y = np.linspace(0, bev_h - 1, bev_h)
                plot_x = np.polyval(tracked['coeffs'], plot_y)
                pts = np.column_stack((plot_x, plot_y)).astype(np.int32)
                cv2.polylines(vis_img, [pts], False, color, 3)

        # Blend overlay
        vis_img = cv2.addWeighted(overlay, 0.25, vis_img, 0.75, 0)

        # Draw center line
        cv2.line(vis_img, (bev_w // 2, 0), (bev_w // 2, bev_h), (255, 255, 255), 1)

        # Sort by x position
        lane_fits.sort(key=lambda lf: np.polyval(lf['coeffs'], y_eval))

        return lane_fits, vis_img

    def _detect_single_lane(
        self,
        peak_x: int,
        nonzero_x: np.ndarray,
        nonzero_y: np.ndarray,
        bev_h: int,
        bev_w: int,
        overlay: np.ndarray,
        vis_img: np.ndarray,
    ) -> Optional[Dict]:
        """
        Try to detect a single lane starting from a histogram peak.

        Used for initializing new lanes that aren't currently tracked.
        """
        n_windows = self.n_windows
        window_height = bev_h // n_windows

        # Pick a color for visualization (gray for new detection attempts)
        color = (150, 150, 150)

        lane_inds = []

        for window in range(n_windows):
            win_y_low = bev_h - (window + 1) * window_height
            win_y_high = bev_h - window * window_height

            # Use initial margin for new lane detection
            margin = self.margin_initial

            win_x_low = max(0, int(peak_x - margin))
            win_x_high = min(bev_w, int(peak_x + margin))

            # Draw window (lighter for new detection)
            cv2.rectangle(overlay, (win_x_low, win_y_low),
                          (win_x_high, win_y_high), color, -1)
            cv2.rectangle(vis_img, (win_x_low, win_y_low),
                          (win_x_high, win_y_high), color, 1)

            # Collect pixels
            good_inds = (
                (nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                (nonzero_x >= win_x_low) & (nonzero_x < win_x_high)
            ).nonzero()[0]
            lane_inds.append(good_inds)

        # Fit polynomial
        if lane_inds:
            all_inds = np.concatenate(lane_inds)
            if len(all_inds) >= self.min_pixels:
                fit, valid = self._fit_polynomial_robust(
                    nonzero_x[all_inds], nonzero_y[all_inds], bev_h
                )
                if fit is not None and valid:
                    return {
                        'coeffs': fit,
                        'base_x': int(peak_x),
                        'pixel_count': len(all_inds),
                        'from_tracked': None,
                    }

        return None

    def _run_tracking_mode(
        self,
        nonzero_x: np.ndarray,
        nonzero_y: np.ndarray,
        bev_h: int,
        bev_w: int,
        overlay: np.ndarray,
        vis_img: np.ndarray,
    ) -> List[Dict]:
        """
        Run adaptive windows using previous polynomial fits.

        Windows start small and grow only if pixels not found.
        """
        n_windows = self.n_windows
        window_height = bev_h // n_windows

        lane_colors = [
            (255, 80, 80), (80, 255, 80), (80, 80, 255),
            (255, 200, 80), (255, 80, 255), (80, 255, 255),
        ]

        candidates = []

        for lane_idx, tracked in enumerate(self.tracked_lanes):
            color = lane_colors[lane_idx % len(lane_colors)]
            coeffs = tracked['coeffs']

            lane_inds = []
            window_margins = []  # Track margin per window for visualization

            for window in range(n_windows):
                win_y_low = bev_h - (window + 1) * window_height
                win_y_high = bev_h - window * window_height
                win_y_center = (win_y_low + win_y_high) / 2.0

                # Predict center from polynomial
                x_predicted = np.polyval(coeffs, win_y_center)

                # Start with minimum margin
                margin = self.margin_min

                # Adaptive search: grow margin if needed
                while margin <= self.margin_max:
                    win_x_low = max(0, int(x_predicted - margin))
                    win_x_high = min(bev_w, int(x_predicted + margin))

                    good_inds = (
                        (nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                        (nonzero_x >= win_x_low) & (nonzero_x < win_x_high)
                    ).nonzero()[0]

                    if len(good_inds) >= self.min_pixels_locked:
                        # Found enough pixels - stay locked
                        break

                    # Grow margin for next iteration
                    margin += self.margin_growth

                # Clamp margin
                margin = min(margin, self.margin_max)
                window_margins.append(margin)

                # Final window bounds
                win_x_low = max(0, int(x_predicted - margin))
                win_x_high = min(bev_w, int(x_predicted + margin))

                # Draw window
                cv2.rectangle(overlay, (win_x_low, win_y_low),
                              (win_x_high, win_y_high), color, -1)
                cv2.rectangle(vis_img, (win_x_low, win_y_low),
                              (win_x_high, win_y_high), color, 2)

                # Collect pixels
                good_inds = (
                    (nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                    (nonzero_x >= win_x_low) & (nonzero_x < win_x_high)
                ).nonzero()[0]
                lane_inds.append(good_inds)

            # Fit polynomial to collected pixels
            if lane_inds:
                all_inds = np.concatenate(lane_inds)
                if len(all_inds) >= self.min_pixels:
                    fit, valid = self._fit_polynomial_robust(
                        nonzero_x[all_inds], nonzero_y[all_inds], bev_h
                    )
                    if fit is not None and valid:
                        candidates.append({
                            'coeffs': fit,
                            'base_x': int(np.polyval(fit, bev_h - 10)),
                            'pixel_count': len(all_inds),
                            'from_tracked': lane_idx,
                        })

        return candidates

    # ========================================================================
    # Temporal Tracking and Smoothing
    # ========================================================================

    def _update_tracked_lanes(self, candidates: List[Dict], bev_h: int):
        """
        Update tracked lanes with new detections using association and smoothing.

        Associates new detections with existing tracks by base position.
        Applies heavy temporal smoothing to polynomial coefficients.
        """
        y_eval = bev_h - 10
        used_candidates = set()

        # Match candidates to existing tracks
        for tracked in self.tracked_lanes:
            tracked_x = np.polyval(tracked['coeffs'], y_eval)
            best_match = None
            best_dist = float('inf')

            for i, cand in enumerate(candidates):
                if i in used_candidates:
                    continue
                cand_x = np.polyval(cand['coeffs'], y_eval)
                dist = abs(cand_x - tracked_x)
                if dist < best_dist and dist < self.max_association_dist:
                    best_dist = dist
                    best_match = i

            if best_match is not None:
                used_candidates.add(best_match)
                cand = candidates[best_match]

                # Apply temporal smoothing to coefficients
                old_coeffs = tracked['coeffs']
                new_coeffs = cand['coeffs']
                smoothed = self.smoothing_alpha * old_coeffs + (1 - self.smoothing_alpha) * new_coeffs

                tracked['coeffs'] = smoothed
                tracked['confidence'] = min(1.0, tracked['confidence'] + 0.2)
                tracked['frames_tracked'] += 1
                tracked['frames_lost'] = 0
                tracked['pixel_count'] = cand['pixel_count']
            else:
                # Track not matched - decay confidence
                tracked['confidence'] *= 0.7
                tracked['frames_lost'] += 1

        # Add new tracks from unmatched candidates
        for i, cand in enumerate(candidates):
            if i not in used_candidates:
                # Check not too close to existing tracks
                cand_x = np.polyval(cand['coeffs'], y_eval)
                too_close = False
                for tracked in self.tracked_lanes:
                    tracked_x = np.polyval(tracked['coeffs'], y_eval)
                    # Use smaller threshold to allow closely-spaced lanes
                    if abs(cand_x - tracked_x) < self.margin_min * 1.5:
                        too_close = True
                        break

                if not too_close and len(self.tracked_lanes) < self.max_lanes:
                    self.tracked_lanes.append({
                        'coeffs': cand['coeffs'],
                        'confidence': 0.5,
                        'frames_tracked': 1,
                        'frames_lost': 0,
                        'pixel_count': cand['pixel_count'],
                    })

        # Remove dead tracks
        self.tracked_lanes = [
            t for t in self.tracked_lanes
            if t['confidence'] > 0.1 and t['frames_lost'] < 5
        ]

        # Sort tracks by x position
        self.tracked_lanes.sort(key=lambda t: np.polyval(t['coeffs'], y_eval))

    # ========================================================================
    # Polynomial Fitting
    # ========================================================================

    def _fit_polynomial_robust(
        self, xs: np.ndarray, ys: np.ndarray, bev_height: int
    ) -> Tuple[Optional[np.ndarray], bool]:
        """
        Fit 2nd order polynomial x = ay² + by + c with outlier rejection.
        """
        if len(xs) < self.min_pixels:
            return None, False

        # Initial fit
        try:
            coeffs = np.polyfit(ys, xs, 2)
        except Exception:
            return None, False

        pred = np.polyval(coeffs, ys)
        residual = xs - pred
        rmse = float(np.sqrt(np.mean(residual ** 2)))

        # Quality checks
        if rmse > self.max_rmse:
            return None, False
        if abs(float(coeffs[0])) > self.max_quadratic:
            # Curvature too high - fit linear instead
            try:
                coeffs_lin = np.polyfit(ys, xs, 1)
                coeffs = np.array([0.0, float(coeffs_lin[0]), float(coeffs_lin[1])])
            except Exception:
                return None, False

        # Refit after trimming outliers
        outlier_threshold = max(5.0, 2.0 * rmse)
        inliers = np.abs(residual) <= outlier_threshold
        n_inliers = np.count_nonzero(inliers)

        if n_inliers >= self.min_pixels:
            try:
                coeffs = np.polyfit(ys[inliers], xs[inliers], 2)
                if abs(float(coeffs[0])) > self.max_quadratic:
                    coeffs[0] = np.sign(coeffs[0]) * self.max_quadratic
            except Exception:
                pass

        # Check vertical span
        y_span = float(np.max(ys) - np.min(ys))
        min_span = bev_height * self.min_y_span_frac
        if y_span < min_span:
            return None, False

        return coeffs, True

    # ========================================================================
    # Metrics Computation
    # ========================================================================

    def _compute_lane_metrics_multilane(self, lane_fits: List[Dict], orig_width: int) -> LaneDetection:
        """
        Compute lane metrics from multiple polynomial fits.
        """
        y_eval = self.bev_height - 10
        all_fits = [lf['coeffs'] for lf in lane_fits]

        # Identify left/right ego-lane boundaries
        left_fit = None
        right_fit = None
        center_x = self.bev_width / 2

        for lf in lane_fits:
            x_at_base = np.polyval(lf['coeffs'], y_eval)
            if x_at_base < center_x:
                if left_fit is None or x_at_base > np.polyval(left_fit, y_eval):
                    left_fit = lf['coeffs']
            else:
                if right_fit is None or x_at_base < np.polyval(right_fit, y_eval):
                    right_fit = lf['coeffs']

        confidence = min(1.0, len(lane_fits) * 0.3)

        lane_width_bev = None
        if left_fit is not None and right_fit is not None:
            left_x = np.polyval(left_fit, y_eval)
            right_x = np.polyval(right_fit, y_eval)
            lane_width_bev = right_x - left_x

            if self.avg_lane_width is None:
                self.avg_lane_width = lane_width_bev
            else:
                self.avg_lane_width = (
                    self.lane_width_smoothing * self.avg_lane_width +
                    (1 - self.lane_width_smoothing) * lane_width_bev
                )
            confidence = min(1.0, confidence + 0.2)

        if left_fit is not None:
            left_x = int(np.polyval(left_fit, y_eval))
            self.left_lane_x = int((left_x / self.bev_width) * orig_width)
        if right_fit is not None:
            right_x = int(np.polyval(right_fit, y_eval))
            self.right_lane_x = int((right_x / self.bev_width) * orig_width)
        if lane_width_bev is not None:
            self.lane_width_pixels = int(lane_width_bev * (orig_width / self.bev_width))

        return LaneDetection(
            valid=len(lane_fits) > 0,
            confidence=confidence,
            lane_center=self.lane_center,
            left_lane=self.left_lane_x,
            right_lane=self.right_lane_x,
            lane_width=self.lane_width_pixels,
            left_fit=left_fit,
            right_fit=right_fit,
            lane_width_bev=lane_width_bev,
            bev_params={
                'width': self.bev_width,
                'height': self.bev_height,
                'forward_range': self.bev_x_range,
                'lateral_range': self.bev_y_range
            },
            all_fits=all_fits,
        )

    # ========================================================================
    # Camera Setup & Reset
    # ========================================================================

    def set_camera_params(self, fov: float, pitch: float, camera_height: float):
        """Set camera extrinsics."""
        self.fov = fov
        self.pitch = pitch
        self.camera_height = camera_height

    def reset(self):
        """Reset detector state."""
        self.left_lane_x = None
        self.right_lane_x = None
        self.lane_center = self.width // 2
        self.lane_width_pixels = None
        self.tracked_lanes = []
        self.avg_lane_width = None

    def get_debug_images(self) -> Dict[str, Optional[np.ndarray]]:
        """Get debug images."""
        return self.debug_images


# ============================================================================
# CLI
# ============================================================================

def _process_single(detector, image_path, output_dir, show=False):
    """Run lane detection on a single semantic image."""
    image_path = Path(image_path)
    semantic_img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if semantic_img is None:
        print(f'Error: Could not read image: {image_path}')
        return 1

    result = detector.detect(semantic_img)
    debug = detector.get_debug_images()

    base_name = image_path.stem

    print(f'\n--- {base_name} ---')
    print(f'  Valid:      {result.valid}')
    print(f'  Confidence: {result.confidence:.2f}')
    print(f'  Lanes:      {len(result.all_fits) if result.all_fits else 0}')
    if result.left_fit is not None:
        print(f'  Left fit:   a={result.left_fit[0]:.6f}')
    if result.right_fit is not None:
        print(f'  Right fit:  a={result.right_fit[0]:.6f}')

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        bev_mask = debug.get('bev_lane_mask')
        lane_fit = debug.get('lane_fit')

        if bev_mask is not None:
            cv2.imwrite(str(output_dir / f'{base_name}_bev_lane_mask.png'), bev_mask)
        if lane_fit is not None:
            cv2.imwrite(str(output_dir / f'{base_name}_lane_fit.png'), lane_fit)

        print(f'  Saved to: {output_dir}')

    if show:
        lane_fit = debug.get('lane_fit')
        if lane_fit is not None:
            cv2.imshow(f'Lane Fit - {base_name}', lane_fit)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return 0


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Curve-aware lane detection')

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str, help='Single semantic image')
    input_group.add_argument('--folder', type=str, help='Folder of images')

    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--fov', type=float, default=90.0)
    parser.add_argument('--pitch', type=float, default=0.0)
    parser.add_argument('--camera-height', type=float, default=1.5, dest='camera_height')

    args = parser.parse_args()
    output_dir = args.output or str(CONFIG.output.default_dir)

    if args.image:
        image_paths = [Path(args.image)]
    else:
        folder = Path(args.folder)
        image_paths = sorted(
            p for p in folder.iterdir()
            if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        )
        if not image_paths:
            print(f'No images found in {folder}')
            return 1

    first_img = cv2.imread(str(image_paths[0]), cv2.IMREAD_GRAYSCALE)
    if first_img is None:
        print(f'Error: Could not read image: {image_paths[0]}')
        return 1
    img_h, img_w = first_img.shape[:2]

    print(f'Image:  {img_w}x{img_h}')
    print(f'Camera: fov={args.fov}deg, pitch={args.pitch}deg, height={args.camera_height}m')

    detector = LaneDetector(image_width=img_w, image_height=img_h)
    detector.set_camera_params(
        fov=args.fov,
        pitch=args.pitch,
        camera_height=args.camera_height,
    )

    for path in image_paths:
        _process_single(detector, path, output_dir, show=args.show)
        detector.reset()

    print(f'\nDone! Processed {len(image_paths)} image(s).')
    return 0


if __name__ == '__main__':
    exit(main())
