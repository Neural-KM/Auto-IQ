"""
Lane Detector — Multi-Lane Polynomial Fitting from Semantic BEV
================================================================

Detects lane markings from semantic segmentation images using a sliding
window approach on a Bird's Eye View (BEV) transform.

Pipeline:
---------
1. Extract road_line pixels from semantic image (class ID = 24)
2. Transform to BEV using Inverse Perspective Mapping (IPM)
3. Find lane base positions using histogram peak detection
4. Run sliding windows from vehicle (bottom) to horizon (top)
5. Fit 2nd-order polynomials with quality checks
6. Apply temporal smoothing for stability

BEV Coordinate System:
----------------------
- Origin: Bottom-center of BEV image (vehicle position)
- X-axis (columns): Lateral position, left is higher column index
- Y-axis (rows): Forward distance, row 0 = far, row H-1 = near vehicle
- Polynomials: x = f(y) where y is row index

Sliding Window Algorithm:
-------------------------
For each detected histogram peak (lane base):
1. Start at base position (bottom of BEV)
2. Search within margin (50px) of current_x for lane pixels
3. Recenter window on mean pixel position if enough pixels found
4. Continue to next window (moving toward horizon)

Visualization (debug 'lane_fit' image):
---------------------------------------
- Yellow pixels: Detected lane marking pixels in BEV
- Colored rectangles: Sliding windows (semi-transparent fill, solid border)
- Colored polylines: Fitted polynomial curves
- White vertical line: Vehicle centerline

Quality Checks:
---------------
- Minimum pixel count (50) for polynomial fit
- Maximum RMSE (15 pixels) between fit and points
- Maximum curvature coefficient (0.015) to reject noise but allow curves
- Minimum vertical span (20% of BEV height)

Changelog:
----------
2026-03-25: Fixed curve handling:
    - Increased max_quadratic from 0.002 to 0.015 (allows tighter curves)
    - Semi-transparent filled windows for clearer visualization
    - Window borders thickness=2 for visibility
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
# Configuration (Dataclasses - frozen for immutability)
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


# Default configuration instance
CONFIG = LaneDetectorConfig()


class LaneDetector:
    """
    Lane detection using semantic segmentation with bird's eye view transform.

    See module docstring for full algorithm description.
    """

    def __init__(self, image_width: int, image_height: int):
        """
        Initialize lane detector.

        Args:
            image_width: Camera image width in pixels
            image_height: Camera image height in pixels

        Note:
            Camera extrinsics (fov, pitch, height) must be set via
            set_camera_params() before calling detect().
        """
        self.width = image_width
        self.height = image_height

        # Camera extrinsics — must be set via set_camera_params() before detect()
        self.fov: Optional[float] = None
        self.pitch: Optional[float] = None
        self.camera_height: Optional[float] = None

        # BEV configuration
        self.bev_width = CONFIG.bev.bev_width
        self.bev_x_range = CONFIG.bev.x_range    # Forward distance range (meters)
        self.bev_y_range = CONFIG.bev.y_range    # Lateral range (meters)
        self.pixels_per_meter = self.bev_width / (self.bev_y_range[1] - self.bev_y_range[0])

        # Derive BEV height from the forward span
        x_span = self.bev_x_range[1] - self.bev_x_range[0]
        self.bev_height = int(x_span * self.pixels_per_meter)

        # Sliding window parameters
        self.n_windows = 9
        self.sw_margin = 50
        self.sw_min_pixels = 30

        # Temporal smoothing
        self.smoothing = 0.6
        self.lane_width_smoothing = 0.95

        # State tracking
        self.left_lane_x: Optional[int] = None
        self.right_lane_x: Optional[int] = None
        self.lane_center = self.width // 2
        self.lane_width_pixels: Optional[int] = None
        self.avg_lane_width: Optional[float] = None

        # Previous polynomial fits for temporal smoothing
        self.prev_lane_fits: List[Dict[str, Any]] = []

        # Debug images
        self.debug_images = {
            'bev_lane_mask': None,
            'lane_fit': None
        }

    def detect(self, semantic_image: np.ndarray) -> LaneDetection:
        """
        Detect lanes from semantic segmentation image.

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

        # Slight dilation to connect nearby pixels
        bev_lane = cv2.dilate(
            bev_lane,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )
        self.debug_images['bev_lane_mask'] = bev_lane

        # Detect multiple lanes using sliding windows
        lane_fits, lane_fit_img = self._fit_multilane_sliding_window(bev_lane)
        self.debug_images['lane_fit'] = lane_fit_img

        # Apply temporal smoothing
        lane_fits = self._smooth_lane_fits(lane_fits)

        return self._compute_lane_metrics_multilane(lane_fits, w)

    def _fit_multilane_sliding_window(self, bev_lane_mask: np.ndarray):
        """
        Detect multiple lanes using histogram peak detection and sliding windows.

        Args:
            bev_lane_mask: Binary BEV image with lane pixels (H x W, uint8)

        Returns:
            (list of lane fit dicts, visualization image)
        """
        bev_h, bev_w = bev_lane_mask.shape[:2]

        # Create visualization image
        vis_img = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
        vis_img[bev_lane_mask > 127] = [255, 255, 0]  # Yellow lane pixels

        # Overlay for semi-transparent windows
        overlay = vis_img.copy()

        # Histogram of bottom portion to find lane bases
        hist_region = bev_lane_mask[int(bev_h * 0.6):, :]
        histogram = np.sum(hist_region, axis=0).astype(np.float32)

        # Smooth histogram for cleaner peak detection
        kernel_size = max(15, int(bev_w * 0.03))
        if kernel_size % 2 == 0:
            kernel_size += 1
        histogram_smooth = cv2.GaussianBlur(
            histogram.reshape(1, -1), (kernel_size, 1), 0
        ).flatten()

        # Find peaks (lane bases)
        min_peak_height = np.max(histogram_smooth) * 0.15
        min_peak_distance = int(bev_w * 0.08)

        peaks, _ = find_peaks(
            histogram_smooth,
            height=max(min_peak_height, 500),
            distance=min_peak_distance,
            prominence=200,
        )

        if len(peaks) == 0:
            cv2.line(vis_img, (bev_w // 2, 0), (bev_w // 2, bev_h), (255, 255, 255), 1)
            return [], vis_img

        # Get nonzero pixels
        nonzero = bev_lane_mask.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        # Sliding window parameters
        n_windows = self.n_windows
        window_height = bev_h // n_windows

        lane_fits = []
        lane_colors = [
            (255, 80, 80),   # Blue
            (80, 255, 80),   # Green
            (80, 80, 255),   # Red
            (255, 200, 80),  # Cyan
            (255, 80, 255),  # Magenta
            (80, 255, 255),  # Yellow
        ]

        # Run sliding window for each detected peak
        for peak_idx, base_x in enumerate(peaks):
            current_x = int(base_x)
            lane_inds = []

            for window in range(n_windows):
                win_y_low = bev_h - (window + 1) * window_height
                win_y_high = bev_h - window * window_height

                win_x_low = max(0, current_x - self.sw_margin)
                win_x_high = min(bev_w, current_x + self.sw_margin)

                # Draw window on overlay (filled) and main image (border)
                color = lane_colors[peak_idx % len(lane_colors)]
                cv2.rectangle(overlay, (win_x_low, win_y_low),
                              (win_x_high, win_y_high), color, -1)
                cv2.rectangle(vis_img, (win_x_low, win_y_low),
                              (win_x_high, win_y_high), color, 2)

                # Find pixels in window
                good_inds = (
                    (nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                    (nonzero_x >= win_x_low) & (nonzero_x < win_x_high)
                ).nonzero()[0]
                lane_inds.append(good_inds)

                # Recenter window if enough pixels found
                if len(good_inds) > self.sw_min_pixels:
                    current_x = int(np.mean(nonzero_x[good_inds]))

            # Collect all pixels for this lane
            if lane_inds:
                lane_inds = np.concatenate(lane_inds)
                if len(lane_inds) > 50:
                    lane_x = nonzero_x[lane_inds]
                    lane_y = nonzero_y[lane_inds]

                    # Fit polynomial with quality check
                    fit, quality = self._fit_polynomial_robust(lane_x, lane_y, bev_h)
                    if fit is not None and quality:
                        lane_fits.append({
                            'coeffs': fit,
                            'base_x': base_x,
                            'pixel_count': len(lane_x),
                        })

                        # Draw fitted curve
                        plot_y = np.linspace(0, bev_h - 1, bev_h)
                        plot_x = np.polyval(fit, plot_y)
                        pts = np.column_stack((plot_x, plot_y)).astype(np.int32)
                        cv2.polylines(vis_img, [pts], False, color, 3)

        # Blend overlay for semi-transparent windows
        vis_img = cv2.addWeighted(overlay, 0.25, vis_img, 0.75, 0)

        # Sort lanes left to right by position at bottom of BEV
        y_eval = bev_h - 10
        lane_fits.sort(key=lambda lf: np.polyval(lf['coeffs'], y_eval))

        # Draw vehicle center line
        cv2.line(vis_img, (bev_w // 2, 0), (bev_w // 2, bev_h), (255, 255, 255), 1)

        return lane_fits, vis_img

    def _fit_polynomial_robust(self, xs: np.ndarray, ys: np.ndarray, bev_height: int):
        """
        Fit 2nd order polynomial with outlier rejection and quality checks.

        Args:
            xs: X coordinates (lateral position in BEV)
            ys: Y coordinates (forward position in BEV, 0=far, H=near)
            bev_height: Height of BEV image for span validation

        Returns:
            (coefficients, is_valid) tuple
        """
        if len(xs) < 50:
            return None, False

        # Initial fit: x = a*y^2 + b*y + c
        coeffs = np.polyfit(ys, xs, 2)
        pred = np.polyval(coeffs, ys)
        residual = xs - pred
        rmse = float(np.sqrt(np.mean(residual ** 2)))

        # Quality thresholds
        max_rmse = 15.0
        # Allow curvature for turns: 0.015 handles ~7m radius curves at 30m range
        max_quadratic = 0.015

        if rmse > max_rmse:
            return None, False
        if abs(float(coeffs[0])) > max_quadratic:
            return None, False

        # Refit after trimming outliers
        inliers = np.abs(residual) <= max(5.0, 2.0 * rmse)
        if np.count_nonzero(inliers) >= 50:
            coeffs = np.polyfit(ys[inliers], xs[inliers], 2)

        # Check vertical span (should cover reasonable portion of BEV)
        y_span = float(np.max(ys) - np.min(ys))
        min_span = bev_height * 0.2
        if y_span < min_span:
            return None, False

        return coeffs, True

    def _smooth_lane_fits(self, lane_fits: List[Dict]) -> List[Dict]:
        """
        Apply temporal smoothing to lane fits using position-based tracking.

        Matches current fits to previous fits by base position, then applies
        exponential smoothing to polynomial coefficients.
        """
        if not lane_fits:
            self.prev_lane_fits = []
            return []

        smoothed = []
        used_prev = set()

        for lf in lane_fits:
            coeffs = lf['coeffs']
            base_x = lf['base_x']

            # Find matching previous fit by base position
            best_match = None
            best_dist = float('inf')
            for i, prev_lf in enumerate(self.prev_lane_fits):
                if i in used_prev:
                    continue
                dist = abs(prev_lf['base_x'] - base_x)
                if dist < best_dist and dist < self.sw_margin * 2:
                    best_dist = dist
                    best_match = i

            if best_match is not None:
                used_prev.add(best_match)
                prev_coeffs = self.prev_lane_fits[best_match]['coeffs']
                coeffs = self.smoothing * prev_coeffs + (1 - self.smoothing) * coeffs

            smoothed.append({
                'coeffs': coeffs,
                'base_x': base_x,
                'pixel_count': lf['pixel_count'],
            })

        self.prev_lane_fits = smoothed
        return smoothed

    def _compute_lane_metrics_multilane(self, lane_fits: List[Dict], orig_width: int) -> LaneDetection:
        """
        Compute lane metrics from multiple polynomial fits.

        Identifies left/right ego-lane boundaries and computes lane width.
        """
        y_eval = self.bev_height - 10

        # Extract coefficients for output
        all_fits = [lf['coeffs'] for lf in lane_fits]

        # Identify left/right lanes relative to vehicle center
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

        # Compute confidence
        confidence = min(1.0, len(lane_fits) * 0.3)

        # Compute lane width if both ego lanes found
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

        # Map to original image coords
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

    def set_camera_params(self, fov: float, pitch: float, camera_height: float):
        """
        Set camera extrinsics (called by node once TF is available).

        Args:
            fov: Camera horizontal field of view in degrees
            pitch: Camera pitch angle in degrees
            camera_height: Camera height above ground in meters
        """
        self.fov = fov
        self.pitch = pitch
        self.camera_height = camera_height

    def reset(self):
        """Reset detector state for new sequence."""
        self.left_lane_x = None
        self.right_lane_x = None
        self.lane_center = self.width // 2
        self.lane_width_pixels = None
        self.prev_lane_fits = []
        self.avg_lane_width = None

    def get_debug_images(self) -> Dict[str, Optional[np.ndarray]]:
        """
        Get intermediate debug images for visualization.

        Returns:
            dict with:
                - 'bev_lane_mask': Binary lane markings in bird's eye view
                - 'lane_fit': BEV with sliding windows and fitted polynomials
        """
        return self.debug_images


# ============================================================================
# CLI
# ============================================================================
def _process_single(detector, image_path, output_dir, show=False):
    """Run lane detection on a single semantic image and save results."""
    image_path = Path(image_path)
    semantic_img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if semantic_img is None:
        print(f'Error: Could not read image: {image_path}')
        return 1

    result = detector.detect(semantic_img)
    debug = detector.get_debug_images()

    base_name = image_path.stem

    # Print results
    print(f'\n--- {base_name} ---')
    print(f'  Valid:      {result.valid}')
    print(f'  Confidence: {result.confidence:.2f}')
    if result.left_fit is not None:
        print(f'  Left fit:   {result.left_fit}')
    if result.right_fit is not None:
        print(f'  Right fit:  {result.right_fit}')
    if result.lane_width_bev is not None and result.bev_params:
        lateral_range = result.bev_params['lateral_range']
        mpp = (lateral_range[1] - lateral_range[0]) / result.bev_params['width']
        print(f'  Lane width: {result.lane_width_bev * mpp:.2f} m '
              f'({result.lane_width_bev:.0f} px BEV)')

    # Save debug images
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
    """CLI entry point for lane detection.

    Input Requirements:
        --image           Path to a single semantic segmentation image (mono8/grayscale)
        --folder          Path to folder of semantic segmentation images

    Camera Parameters:
        --fov             Camera horizontal field of view in degrees (default: 90.0)
        --pitch           Camera pitch angle in degrees (default: 0.0)
        --camera-height   Camera height above ground in meters (default: 1.5)

    Output:
        --output          Output directory for debug images (default: data/lane_detection/)
        --show            Display results in OpenCV window
    """
    cli_examples = """
Examples:
    # Single semantic image
    python -m mapping.core.lane_fitting --image semantic.png

    # Batch process a folder
    python -m mapping.core.lane_fitting --folder /data/semantic_maps/ --output ./lane_results/

    # Display results
    python -m mapping.core.lane_fitting --image semantic.png --show
"""

    parser = argparse.ArgumentParser(
        description='Detect lanes from semantic segmentation images',
        epilog=cli_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str, help='Path to single semantic image (grayscale/mono8)')
    input_group.add_argument('--folder', type=str, help='Path to folder of semantic images')

    # Output
    parser.add_argument('--output', type=str, default=None,
                        help=f'Output directory (default: {CONFIG.output.default_dir})')
    parser.add_argument('--show', action='store_true', help='Display results in OpenCV window')

    # Camera parameters
    parser.add_argument('--fov', type=float, default=90.0,
                        help='Camera horizontal field of view in degrees (default: 90.0)')
    parser.add_argument('--pitch', type=float, default=0.0,
                        help='Camera pitch angle in degrees (default: 0.0)')
    parser.add_argument('--camera-height', type=float, default=1.5, dest='camera_height',
                        help='Camera height above ground in meters (default: 1.5)')

    args = parser.parse_args()

    output_dir = args.output or str(CONFIG.output.default_dir)

    # Collect images
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

    # Read first image to get dimensions
    first_img = cv2.imread(str(image_paths[0]), cv2.IMREAD_GRAYSCALE)
    if first_img is None:
        print(f'Error: Could not read image: {image_paths[0]}')
        return 1
    img_h, img_w = first_img.shape[:2]

    # Create detector
    print(f'Image:  {img_w}x{img_h}')
    print(f'Camera: fov={args.fov}deg, pitch={args.pitch}deg, height={args.camera_height}m')

    detector = LaneDetector(
        image_width=img_w,
        image_height=img_h,
    )
    detector.set_camera_params(
        fov=args.fov,
        pitch=args.pitch,
        camera_height=args.camera_height,
    )

    # Process
    for path in image_paths:
        _process_single(detector, path, output_dir, show=args.show)
        detector.reset()

    print(f'\nDone! Processed {len(image_paths)} image(s).')
    return 0


if __name__ == '__main__':
    exit(main())
