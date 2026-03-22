"""
Local Lane Estimator
====================

Unified module that estimates local road structure from:
1. BEV occupancy grid (for road boundary extraction)
2. Semantic segmentation (for lane marking polynomial fits)

Returns a single LocalRoadEstimate containing:
- Road boundaries (left/right edges of local lane data)
- Lane polynomial fits (from actual road_line markings in image)
- Estimated lane structure (guessed from boundaries using a priori knowledge)

NOTE: The "local_lane_data" lanes are NOT detected - they are estimated/guessed
by dividing the detected road width using configured lane_count and road_width
parameters. This uses a priori knowledge about expected road structure.

The "lane_fits" are actual polynomial fits to painted lane markings detected
in the semantic segmentation image.
"""

from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any

from mapping.core.boundary_extractor import (
    extract_boundaries,
    boundary_to_base_link,
)
from mapping.core.local_lane_data_generator import (
    LocalLaneDataGenerator,
    LocalLaneDataResult,
    LaneResult,
    STATUS_OK,
    STATUS_NO_BOUNDARIES,
)
from mapping.utils.ipm_transform import convert_to_ipm
from perception.data_classes.semantic_labels import SEMANTIC_BY_NAME


ROAD_ID = SEMANTIC_BY_NAME['road']
ROAD_LINE_ID = SEMANTIC_BY_NAME['road_line']

# ============================================================================
# Result Dataclasses
# ============================================================================


@dataclass
class LanePolyFit:
    """Polynomial fit result for a single lane marking."""
    coeffs: np.ndarray           # Polynomial coefficients [a, b, c] for x = ay^2 + by + c
    confidence: float            # Detection confidence (0-1)
    pixel_count: int             # Number of pixels used for fit
    y_range: Tuple[float, float] # Valid y range in BEV pixels


@dataclass
class LaneMarkingFits:
    """Polynomial fits for detected lane markings (from actual road_line pixels)."""
    valid: bool = False
    all_fits: List[LanePolyFit] = field(default_factory=list)
    bev_params: Dict[str, Any] = field(default_factory=dict)
    debug_image: Optional[np.ndarray] = None


@dataclass
class LocalRoadEstimate:
    """
    Complete local road structure estimate.

    Contains two types of lane information:
    1. lane_fits: Actual polynomial fits to painted lane markings (road_line pixels)
    2. local_lane_data: Estimated/guessed lanes from road boundaries using a priori knowledge

    The local_lane_data lanes are NOT detected - they are computed by dividing
    the detected road width equally based on configured lane_count parameter.
    """
    # Validity
    valid: bool = False

    # Road boundaries (from BEV local lane data extraction)
    left_boundary: Optional[np.ndarray] = None   # (N, 2) x, y in base_link
    right_boundary: Optional[np.ndarray] = None  # (N, 2) x, y in base_link

    # Lane marking polynomial fits (from semantic road_line pixels - ACTUAL detection)
    lane_fits: Optional[LaneMarkingFits] = None

    # Estimated lane structure (guessed from boundaries using a priori knowledge)
    local_lane_data: Optional[LocalLaneDataResult] = None

    # Debug images
    debug_images: Dict[str, np.ndarray] = field(default_factory=dict)


# ============================================================================
# Lane Marking Polynomial Fitter
# ============================================================================

class LaneMarkingFitter:
    """
    Fits polynomial curves to lane markings using sliding window search.

    Operates on BEV-transformed semantic images to find road_line pixels
    and fit 2nd-order polynomials. This detects ACTUAL painted lane markings.
    """

    def __init__(
        self,
        bev_x_range: Tuple[float, float] = (5.0, 35.0),
        bev_y_range: Tuple[float, float] = (-15.0, 15.0),
        pixels_per_meter: float = 10.0,
        n_windows: int = 9,
        margin: int = 50,
        min_pixels: int = 30,
    ):
        """
        Initialize lane marking fitter.

        Args:
            bev_x_range: Forward range in meters (min, max)
            bev_y_range: Lateral range in meters (left positive)
            pixels_per_meter: BEV resolution
            n_windows: Number of sliding windows
            margin: Window half-width in pixels
            min_pixels: Minimum pixels to recenter window
        """
        self.bev_x_range = bev_x_range
        self.bev_y_range = bev_y_range
        self.pixels_per_meter = pixels_per_meter
        self.n_windows = n_windows
        self.margin = margin
        self.min_pixels = min_pixels
        self.min_fit_pixels = 60
        self.min_fit_span_px = 40
        self.max_component_width_px = int(max(12, 3.0 * self.pixels_per_meter))
        self.max_rmse_px = 10.0
        self.max_abs_quadratic = 0.0012

        # Derived BEV dimensions
        x_span = bev_x_range[1] - bev_x_range[0]
        y_span = bev_y_range[1] - bev_y_range[0]
        self.bev_height = int(x_span * pixels_per_meter)
        self.bev_width = int(y_span * pixels_per_meter)

    def fit(
        self,
        semantic_img: np.ndarray,
        fov: float,
        pitch: float,
        camera_height: float,
    ) -> LaneMarkingFits:
        """
        Extract and fit lane markings from semantic segmentation.

        Args:
            semantic_img: Mono8 semantic segmentation image
            fov: Camera horizontal FOV in degrees
            pitch: Camera pitch in degrees
            camera_height: Camera height in meters

        Returns:
            LaneMarkingFits with polynomial coefficients
        """
        # 1) Project full drivable area (road + markings) to BEV.
        drivable_mask = ((semantic_img == ROAD_ID) | (semantic_img == ROAD_LINE_ID)).astype(np.uint8) * 255
        bev_drivable, _, _ = convert_to_ipm(
            image=drivable_mask,
            fov=fov,
            pitch=pitch,
            camera_height=camera_height,
            bev_x_range=self.bev_x_range,
            bev_y_range=self.bev_y_range,
            pixels_per_meter=self.pixels_per_meter,
            interpolation=cv2.INTER_NEAREST,
        )

        # 2) Project lane-marking pixels to BEV and keep only those inside drivable ROI.
        lane_mask = (semantic_img == ROAD_LINE_ID).astype(np.uint8) * 255
        bev_lane_raw, _, _ = convert_to_ipm(
            image=lane_mask,
            fov=fov,
            pitch=pitch,
            camera_height=camera_height,
            bev_x_range=self.bev_x_range,
            bev_y_range=self.bev_y_range,
            pixels_per_meter=self.pixels_per_meter,
            interpolation=cv2.INTER_NEAREST,
        )

        drivable_roi = bev_drivable > 127
        lane_candidates = (bev_lane_raw > 127) & drivable_roi

        # Slight dilation keeps thin projected lane pixels connected for sliding windows.
        lane_candidates = cv2.dilate(
            lane_candidates.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )
        bev_lane = (lane_candidates * 255).astype(np.uint8)

        # Update actual dimensions from transform
        bev_height, bev_width = bev_lane.shape[:2]

        # Multi-lane extraction from connected components + robust polynomial fitting.
        tracks, debug_img = self._fit_from_connected_components(bev_lane, bev_width, bev_height)

        # Compute results
        result = LaneMarkingFits(
            valid=(len(tracks) > 0),
            bev_params={
                'width': bev_width,
                'height': bev_height,
                'x_range': self.bev_x_range,
                'y_range': self.bev_y_range,
                'pixels_per_meter': self.pixels_per_meter,
            },
            debug_image=debug_img,
        )

        for track in tracks:
            result.all_fits.append(
                LanePolyFit(
                    coeffs=track['coeffs'],
                    confidence=float(min(1.0, track['pixel_count'] / 300.0)),
                    pixel_count=int(track['pixel_count']),
                    y_range=track['y_range'],
                )
            )

        return result

    def _fit_from_connected_components(
        self,
        bev_mask: np.ndarray,
        bev_width: int,
        bev_height: int,
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Perform sliding window search and polynomial fitting.

        Returns:
            (lane_tracks, visualization_image)
        """
        # Visualization
        vis = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)
        vis[bev_mask > 127] = [255, 255, 0]  # Yellow lane pixels

        lane_bin = (bev_mask > 127).astype(np.uint8)

        # Keep component extraction direction-agnostic.
        # A fixed vertical close can incorrectly merge curved dashed markings.
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(lane_bin, connectivity=8)

        tracks: List[Dict[str, Any]] = []

        for label in range(1, num_labels):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])

            if area < self.min_fit_pixels:
                continue
            if h < self.min_fit_span_px:
                continue
            if w > self.max_component_width_px:
                continue

            component_mask = (labels == label)

            # Fit on original lane pixels inside the connected component.
            ys, xs = np.where(component_mask & (lane_bin > 0))
            if len(xs) < self.min_fit_pixels:
                continue

            coeffs = np.polyfit(ys, xs, 2)
            pred = np.polyval(coeffs, ys)
            residual = xs - pred
            rmse = float(np.sqrt(np.mean(residual ** 2)))

            # Reject unrealistic fits.
            if rmse > self.max_rmse_px:
                continue
            if abs(float(coeffs[0])) > self.max_abs_quadratic:
                continue

            # One robust refit after trimming outliers.
            inliers = np.abs(residual) <= max(4.0, 2.0 * rmse)
            if np.count_nonzero(inliers) >= self.min_fit_pixels:
                ys_in = ys[inliers]
                xs_in = xs[inliers]
                coeffs = np.polyfit(ys_in, xs_in, 2)
                ys_fit = ys_in
                pixel_count = int(len(xs_in))
            else:
                ys_fit = ys
                pixel_count = int(len(xs))

            tracks.append({
                'coeffs': coeffs,
                'pixel_count': pixel_count,
                'y_range': (float(np.min(ys_fit)), float(np.max(ys_fit))),
                'x_ref': float(np.polyval(coeffs, bev_height - 10)),
            })

            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 180, 0), 1)

        # Sort lanes from left to right at near-ego y for stable ordering.
        tracks.sort(key=lambda t: t['x_ref'])

        # Draw fitted curves
        lane_colors = [
            (255, 80, 80),
            (80, 255, 80),
            (80, 80, 255),
            (255, 200, 80),
            (255, 80, 255),
            (80, 255, 255),
        ]
        plot_y = np.linspace(0, bev_height - 1, bev_height)

        for idx, track in enumerate(tracks):
            plot_x = np.polyval(track['coeffs'], plot_y)
            pts = np.column_stack((plot_x, plot_y)).astype(np.int32)
            cv2.polylines(vis, [pts], False, lane_colors[idx % len(lane_colors)], 3)

        # Vehicle center line
        cv2.line(vis, (bev_width // 2, 0), (bev_width // 2, bev_height), (255, 255, 255), 1)

        return tracks, vis

# ============================================================================
# Unified Local Lane Estimator
# ============================================================================

class LocalLaneEstimator:
    """
    Unified estimator for local road structure.

    Produces two types of lane information:
    1. lane_fits: Actual polynomial fits to painted lane markings (detected)
    2. local_lane_data: Estimated lanes from boundaries (guessed using a priori knowledge)

    The local_lane_data uses configured lane_count and total_road_width to
    ESTIMATE/GUESS lane positions by dividing the detected road boundaries.
    This is NOT actual lane detection - it's geometric estimation.

    Usage:
        estimator = LocalLaneEstimator(...)
        result = estimator.estimate(grid, semantic_img, ...)
    """

    def __init__(
        self,
        # Grid parameters
        grid_cells: int,
        origin_x: float,
        origin_y: float,
        resolution: float,
        # Lane structure parameters (a priori knowledge for estimation)
        lane_count: int = 4,
        total_road_width: float = 14.0,
        clearance_from_wall: float = 0.3,
        clearance_from_divider: float = 0.3,
        min_road_width: float = 10.0,
        max_road_width: float = 30.0,
        # Lane fitting BEV parameters
        lane_fit_bev_x_range: Tuple[float, float] = (5.0, 35.0),
        lane_fit_bev_y_range: Tuple[float, float] = (-15.0, 15.0),
        lane_fit_pixels_per_meter: float = 10.0,
    ):
        """
        Initialize the local lane estimator.

        Args:
            grid_cells: Number of cells in occupancy grid (square)
            origin_x: Grid origin X in base_link (meters)
            origin_y: Grid origin Y in base_link (meters)
            resolution: Grid cell size (meters)
            lane_count: Expected number of lanes (a priori knowledge)
            total_road_width: Expected total road width in meters (a priori)
            clearance_from_wall: Safety margin from right edge (meters)
            clearance_from_divider: Safety margin from left edge (meters)
            min_road_width: Minimum valid road width (meters)
            max_road_width: Maximum valid road width (meters)
            lane_fit_bev_x_range: Forward range for lane fitting BEV
            lane_fit_bev_y_range: Lateral range for lane fitting BEV
            lane_fit_pixels_per_meter: Resolution for lane fitting BEV
        """
        self.grid_cells = grid_cells
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.resolution = resolution

        # Drivable area generator (uses a priori knowledge to estimate lanes)
        self.local_lane_data_generator = LocalLaneDataGenerator(
            lane_count=lane_count,
            total_road_width=total_road_width,
            clearance_from_wall=clearance_from_wall,
            clearance_from_divider=clearance_from_divider,
            min_road_width=min_road_width,
            max_road_width=max_road_width,
        )

        # Lane marking fitter (detects actual painted lane markings)
        self.lane_fitter = LaneMarkingFitter(
            bev_x_range=lane_fit_bev_x_range,
            bev_y_range=lane_fit_bev_y_range,
            pixels_per_meter=lane_fit_pixels_per_meter,
        )

        # Constants
        self.FREE = 0
        self.UNKNOWN = -1
        self.min_path_points = 8
        self.min_forward_span_m = 3.0

    def estimate(
        self,
        grid: np.ndarray,
        semantic_img: Optional[np.ndarray] = None,
        camera_fov: Optional[float] = None,
        camera_pitch: Optional[float] = None,
        camera_height: Optional[float] = None,
    ) -> LocalRoadEstimate:
        """
        Estimate local road structure from occupancy grid and semantic image.

        Args:
            grid: Occupancy grid (grid_cells x grid_cells) with FREE=0, UNKNOWN=-1, OCCUPIED=100
            semantic_img: Optional mono8 semantic segmentation for lane marking fitting
            camera_fov: Camera horizontal FOV (required if semantic_img provided)
            camera_pitch: Camera pitch angle (required if semantic_img provided)
            camera_height: Camera height (required if semantic_img provided)

        Returns:
            LocalRoadEstimate with boundaries, lane fits, and estimated local lane data
        """
        result = LocalRoadEstimate()

        # 1. Extract road boundaries from occupancy grid
        binary_mask = np.uint8(grid == self.FREE)
        observed_mask = np.uint8(grid != self.UNKNOWN)

        left, right, edge_img, l_diag, r_diag = extract_boundaries(binary_mask, observed_mask)

        # Convert pixel boundaries to base_link coordinates
        left_pts = self._boundary_to_coords(left)
        right_pts = self._boundary_to_coords(right)

        result.left_boundary = left_pts
        result.right_boundary = right_pts

        # Store boundary debug image
        bev_vis = (binary_mask * 255).astype(np.uint8)
        from mapping.core.boundary_extractor import draw_boundary_overlay
        result.debug_images['boundary_overlay'] = draw_boundary_overlay(
            bev_vis, left, right, l_diag, r_diag, edge_img
        )

        # 2. Estimate lane structure from boundaries (using a priori knowledge)
        drivable_result = self.local_lane_data_generator.generate(left_pts, right_pts)
        result.local_lane_data = drivable_result

        # 3. Fit lane marking polynomials (actual detection from road_line pixels)
        if semantic_img is not None and camera_fov is not None:
            lane_fits = self.lane_fitter.fit(
                semantic_img=semantic_img,
                fov=camera_fov,
                pitch=camera_pitch or 0.0,
                camera_height=camera_height or 1.5,
            )
            result.lane_fits = lane_fits

            if lane_fits.debug_image is not None:
                result.debug_images['lane_fit'] = lane_fits.debug_image

        # Set validity based on having useful data
        result.valid = (
            drivable_result.valid or
            (result.lane_fits is not None and result.lane_fits.valid)
        )

        return result

    def _boundary_to_coords(self, boundary_result) -> Optional[np.ndarray]:
        """Convert pixel boundary to base_link coordinates."""
        pts = boundary_to_base_link(
            boundary_result,
            grid_cells=self.grid_cells,
            origin_x=self.origin_x,
            origin_y=self.origin_y,
            resolution=self.resolution,
            sample_step=3,
        )

        if pts is None or len(pts) < self.min_path_points:
            return None

        forward_span = float(np.max(pts[:, 0]) - np.min(pts[:, 0]))
        if forward_span < self.min_forward_span_m:
            return None

        return pts
