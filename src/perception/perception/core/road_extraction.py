#!/usr/bin/env python3
"""
Road Extraction (Segmentation Model Agnostic)
==============================================

This module provides the RoadExtractor class for extracting road masks and lane 
markings from images. It is segmentation model agnostic - it receives pre-computed
segmentation maps or road masks from any upstream source.

RoadExtractor Class:
    Parameters:
        road_grayscale_id (int): Pixel value ID for road in the grayscale segmentation map (default: 90) we are converting the rgb id of the object to grayscale to get this value
    Core Methods:
        extract_road_mask(segmentation)              - Binary road mask from pre-computed segmentation
        extract_road_only(image, road_mask)          - Original image with only road pixels visible
        visualize_road(image, road_mask, mode)       - Visualization (overlay/colored/mask_only modes)
        extract_lanes(image, road_mask)              - Lane detection on perspective images
        extract_lanes_ipm(image, road_mask, ...)     - Lane detection with Inverse Perspective Mapping

Configuration:
    RoadExtractionConfig dataclass provides default settings for:
        - IPMConfig: x_range, y_range, pixels_per_meter
        - OutputConfig: default_dir

Example:
    from perception.core.road_extraction import RoadExtractor
    
    extractor = RoadExtractor(road_grayscale_id=90)
    
    # Option 1: Provide segmentation map (class IDs per pixel)
    road_mask = extractor.extract_road_mask(segmentation_map)
    
    # Option 2: Use pre-computed road mask directly
    road_only, ipm_road, ipm_lanes, ipm_mask, size = extractor.extract_lanes_ipm(
        image, road_mask, fov=110, pitch=5, camera_height=1.2
    )
"""

import numpy as np
import cv2
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from perception.utils.lane_detection import _extract_lanes_from_ipm
from perception.utils.ipm_transform import convert_to_ipm
from perception.data_classes.semantic_labels import SEMANTIC_BY_NAME

# ============================================================================
# Configuration (Dataclasses - frozen for immutability)
# ============================================================================

@dataclass(frozen=True)
class IPMConfig:
    """Inverse Perspective Mapping transform parameters."""
    x_range: Tuple[float, float] = (3.0, 30.0)     # meters forward (near to far)
    y_range: Tuple[float, float] = (-10.0, 10.0)   # meters left/right from vehicle center
    pixels_per_meter: float = 30.0                  # resolution: 1 meter = 30 pixels


@dataclass(frozen=True)
class OutputConfig:
    """Output paths configuration."""
    default_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[4] / 'data' / 'rosbag_images_15jan' / 'inference')

@dataclass(frozen=True)
class RoadExtractionConfig:
    """Main configuration for road extraction module."""
    road_grayscale_id: int = 90          # Pixel intensity value for road (e.g. 90 for Cityscapes)
    ipm: IPMConfig = field(default_factory=IPMConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


# Default configuration instance
CONFIG = RoadExtractionConfig()


class RoadExtractor:
    """
    Extracts road mask from images using pre-computed segmentation.
    
    This class is segmentation model agnostic - it receives segmentation maps
    or road masks from any upstream source (SegFormer, DeepLab, etc.).
    """
    
    def __init__(self, road_grayscale_id: int = 90):
        """
        Initialize RoadExtractor.
        
        Args:
            road_grayscale_id: Pixel value corresponding to road in the segmentation map.
                               For grayscale visualizations, this is the intensity value (e.g., 90).
        """
        self.road_grayscale_id = road_grayscale_id
    
    def extract_road_mask(self, segmentation: np.ndarray) -> np.ndarray:

        """
        Extract binary road mask from segmentation map.
        
        Args:
            segmentation: Segmentation map (H, W)
            
        Returns:
            Binary mask (H, W) where 255 = road, 0 = non-road
        """
        # Extract road pixels based on configured road value
        return (segmentation == self.road_grayscale_id).astype(np.uint8) * 255
    
    def extract_road_with_confidence(
        self, 
        segmentation: np.ndarray,
        confidence: np.ndarray,
        threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract road mask with confidence filtering.
        
        Args:
            segmentation: Segmentation map (H, W)
            confidence: Confidence scores (H, W) for each pixel
            threshold: Minimum confidence threshold for road pixels
            
        Returns:
            road_mask: Binary mask (H, W) where 255 = confident road
            road_confidence: Confidence scores for road pixels (H, W)
        """
        # Road pixels with high confidence
        is_road = segmentation == self.road_grayscale_id
        is_confident = confidence >= threshold
        
        road_mask = (is_road & is_confident).astype(np.uint8) * 255
        road_confidence = confidence * is_road
        
        return road_mask, road_confidence
    
    def visualize_road(
        self, 
        image: np.ndarray,
        road_mask: np.ndarray, 
        mode: str = 'overlay',
        alpha: float = 0.5,
        road_color: Tuple[int, int, int] = (128, 64, 128)
    ) -> np.ndarray:
        """
        Visualize road extraction results.
        
        Args:
            image: RGB image (H, W, 3)
            road_mask: Binary road mask (H, W) where 255 = road
            mode: Visualization mode:
                - 'overlay': Blend road color on original image
                - 'colored': Road colored, rest black
                - 'mask_only': Return original pixels only where road is detected
            alpha: Blend factor for overlay mode (0-1)
            road_color: RGB color for road visualization
            
        Returns:
            RGB visualization image
        """
        if mode == 'colored':
            # Road colored, rest black
            result = np.zeros_like(image)
            result[road_mask == 255] = road_color
        elif mode == 'mask_only':
            # Original pixels where road is detected
            result = cv2.bitwise_and(image, image, mask=road_mask)
        else:  # 'overlay' (default)
            # Blend road color on original image
            overlay = image.copy()
            overlay[road_mask == 255] = road_color
            result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
        
        return result
    
    def extract_road_only(self, image: np.ndarray, road_mask: np.ndarray) -> np.ndarray:
        """
        Extract only road pixels from original image (Separate road from real image).
        
        Args:
            image: RGB image (H, W, 3)
            road_mask: Binary road mask (H, W) where 255 = road
            
        Returns:
            RGB image with only road pixels visible, rest black
        """
        # Apply mask to isolate road pixels, rest becomes black
        return cv2.bitwise_and(image, image, mask=road_mask)

    # =========================================================================
    # Inverse Perspective Mapping (IPM) Methods
    # =========================================================================
    
    def extract_lanes_ipm(
        self, 
        image: np.ndarray,
        road_mask: np.ndarray,
        fov: float = None,
        pitch: float = None,
        camera_height: float = None,
        ipm_x_range: Tuple[float, float] = None,
        ipm_y_range: Tuple[float, float] = None,
        pixels_per_meter: float = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
        """
        Extract lane markings using the correct pipeline:
        Road Mask → IPM (Original Image) → Lane Detection
        
        This method is segmentation model agnostic - it receives the road mask
        from any upstream source.
        
        Pipeline:
        1. Receive road mask from upstream (extracted from segmentation map)
        2. Convert ORIGINAL image to IPM (for full context)
        3. Convert road mask to IPM
        4. Find lane markings in IPM image (using IPM road mask to filter)
        
        Args:
            image: RGB image (H, W, 3)
            road_mask: Binary road mask (H, W) where 255 = road, 0 = non-road
            fov: Camera horizontal field of view in degrees (required).
            pitch: Camera pitch angle in degrees (required).
            camera_height: Camera height above ground in meters (required).
            ipm_x_range: (x_min, x_max) forward range in meters. Uses CONFIG.ipm.x_range if None.
            ipm_y_range: (y_min, y_max) lateral range in meters (left/right). Uses CONFIG.ipm.y_range if None.
            pixels_per_meter: Resolution - pixels per meter. Uses CONFIG.ipm.pixels_per_meter if None.
            
        Returns:
            road_only: Separated road image (original perspective) - for visualization
            ipm_image: Inverse Perspective Mapping of ORIGINAL image
            ipm_lane_mask: Lane markings mask in IPM space
            ipm_road_mask: Road mask in IPM space
            ipm_size: (width, height) of the IPM output
        
        Raises:
            ValueError: If fov, pitch, or camera_height is None.
        """
        # Camera extrinsics must be provided by the caller
        if fov is None or pitch is None or camera_height is None:
            raise ValueError(
                'fov, pitch, and camera_height are required. '
                'These must come from camera_info topic / tf lookup via the node.')
        
        height, width = image.shape[:2]
        
        # 1. Convert convert ORIGINAL image to IPM (Use full image for better context)
        print("Step 1: Converting original image to IPM...")
        ipm_image, ipm_size, M = convert_to_ipm(
            image,
            fov=fov,
            pitch=pitch,
            camera_height=camera_height,
            bev_x_range=ipm_x_range,
            bev_y_range=ipm_y_range,
            pixels_per_meter=pixels_per_meter,
        )
        
        print(f"IPM size: {ipm_size[0]}x{ipm_size[1]} pixels (auto-calculated for equal scale)")

        # 2. Transform road mask to IPM for reference/filtering
        ipm_road_mask = cv2.warpPerspective(
            road_mask, M, ipm_size, 
            flags=cv2.INTER_NEAREST
        )
        
        # Check if road is visible in this IPM view
        road_pixel_count = np.count_nonzero(ipm_road_mask)
        print(f"DEBUG: Road pixels in IPM view: {road_pixel_count}")
        
        if road_pixel_count == 0:
            print("WARNING: No road pixels visible in the configured IPM region!")
            print(f"         IPM X-Range: {ipm_x_range}")
            print(f"         IPM Y-Range: {ipm_y_range}")
            print("         Try adjusting the ranges or camera parameters.")
            # return empty lane mask to avoid crash
            ipm_lane_mask = np.zeros_like(ipm_road_mask)
        else:
            # 3. Find lane markings in IPM image
            print("Step 2: Detecting lane markings in IPM...")
            ipm_lane_mask = _extract_lanes_from_ipm(ipm_image, ipm_road_mask)
        
        # 4. Create road_only for return visualization (perspective)
        road_only = cv2.bitwise_and(image, image, mask=road_mask)
        
        return road_only, ipm_image, ipm_lane_mask, ipm_road_mask, ipm_size
    
    
    
    def extract_lanes(self, image: np.ndarray, road_mask: np.ndarray) -> np.ndarray:
        """
        Extract lane markings using Adaptive Thresholding on the Road ROI.
        
        Args:
            image: RGB image (H, W, 3)
            road_mask: Binary road mask (H, W) where 255 = road
            
        Returns:
            Binary mask (H, W) where 255 = lane markings
        """
        # 1. Extract road ROI using provided mask
        road_only = cv2.bitwise_and(image, image, mask=road_mask)
        
        # 2. Convert to Grayscale
        gray = cv2.cvtColor(road_only, cv2.COLOR_RGB2GRAY)
        
        # 3. Apply Adaptive Thresholding
        mean_road_val = cv2.mean(gray, mask=road_mask)[0]
        filled_gray = gray.copy()
        filled_gray[road_mask == 0] = int(mean_road_val)
        
        # Use simple Gaussian Adaptive thresholding
        # Block size 21, C = -10 (Pixel > Mean + 10)
        lane_mask = cv2.adaptiveThreshold(
            filled_gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            -10
        )
        
        # 4. Clean up with morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel)
        
        # Enforce ROI
        lane_mask = cv2.bitwise_and(lane_mask, road_mask)
        
        return lane_mask

    # =========================================================================
    # Road Lane Extraction (Lane Marking Injection)
    # =========================================================================

    def extract_road_lane_markings(
        self,
        semantic_map: np.ndarray,
        rgb_image: np.ndarray,
        road_class_id: int,
        road_line_class_id: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject lane markings into a semantic map that lacks a road_line class.

        SegFormer (trained on Cityscapes) lumps lane markings into the 'road'
        class.  This method detects bright lane markings from the RGB image
        within the road mask using adaptive thresholding, then writes the
        road_line class ID into the semantic map at those pixels.

        The returned map is pixel-identical in format to CARLA ground truth,
        so the entire downstream stack (mapping, planning, control) works
        without any changes.

        Args:
            semantic_map: Mono8 class-ID map (H, W) — e.g. from SegFormer.
                          Modified **in-place** and also returned.
            rgb_image:    RGB image (H, W, 3) — used for lane detection.
            road_class_id:      Class ID for road (default: SEMANTIC_BY_NAME['road'] = 1).
            road_line_class_id: Class ID for road_line to inject
                                (default: SEMANTIC_BY_NAME['road_line'] = 24).

        Returns:
            extracted_map:  The semantic map with road_line pixels injected (same array as input).
            lane_mask:  Binary mask (H, W) where 255 = detected lane marking.
        """
        # 1. Build road mask from class IDs
        road_mask = (semantic_map == road_class_id).astype(np.uint8) * 255

        road_pixels = np.count_nonzero(road_mask)
        if road_pixels == 0:
            # No road visible — nothing to extract
            empty = np.zeros(semantic_map.shape[:2], dtype=np.uint8)
            return semantic_map, empty

        # 2. Detect lane markings in perspective space (adaptive threshold)
        lane_mask = self.extract_lanes(rgb_image, road_mask)

        # 3. Write road_line class into the semantic map
        semantic_map[lane_mask > 0] = road_line_class_id

        return semantic_map, lane_mask


def process_single_image(extractor: RoadExtractor, args, segmentation_map: np.ndarray = None):
    """Process a single image.
    
    Args:
        extractor: RoadExtractor instance
        args: Command line arguments
        segmentation_map: Pre-computed segmentation map (H, W) with class IDs.
                         If None, a dummy road mask will be created for demonstration.
    """
    img_path = Path(args.image)
    
    if not img_path.exists():
        print(f'Error: Image not found: {img_path}')
        return 1

    # Load image
    image_bgr = cv2.imread(str(img_path))
    if image_bgr is None:
        print(f'Error: Could not load image: {img_path}')
        return 1

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    print(f'Processing: {img_path.name} ({image_rgb.shape[1]}x{image_rgb.shape[0]})')
    
    # Get road mask from segmentation or load from file
    if args.seg_map_path:
        # Load pre-computed segmentation map (Class IDs)
        seg_map = cv2.imread(args.seg_map_path, cv2.IMREAD_GRAYSCALE)
        if seg_map is None:
            print(f'Error: Could not load segmentation map from: {args.seg_map_path}')
            return 1
        print(f'Loaded segmentation map from: {args.seg_map_path}')
        
        # DEBUG: Print unique values in segmentation map
        unique_classes = np.unique(seg_map)
        print(f"DEBUG: Unique pixel values found in segmentation map: {unique_classes}")
        print(f"DEBUG: Using road_grayscale_id={extractor.road_grayscale_id}")
        
        # Extract road mask
        road_mask = extractor.extract_road_mask(seg_map)
        
        # DEBUG: Check if mask is empty
        road_pixels = np.count_nonzero(road_mask)
        print(f"DEBUG: Extracted road mask has {road_pixels} pixels ({(road_pixels/road_mask.size)*100:.2f}% of image)")
        
        if road_pixels == 0:
            print("ERROR: Road mask is empty! Check if road_grayscale_id matches the segmentation map values.")
            print(f"       Available values: {unique_classes}")
            return 1
            
        print('Extracted road mask from segmentation map')

    elif segmentation_map is not None:
        # Extract road mask from provided segmentation
        road_mask = extractor.extract_road_mask(segmentation_map)
        print('Extracted road mask from provided segmentation map')
    else:
        print('Error: No segmentation map provided.')
        print('Use --seg-map-path to provide a pre-computed segmentation map file.')
        return 1
    
    # Determine output paths
    if args.output:
        out_dir = Path(args.output)
    else:
        out_dir = CONFIG.output.default_dir
        
    out_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = img_path.stem
    
    # =========================================================================
    # Inverse Perspective Mapping (IPM) Processing Mode
    # Pipeline: Road Mask → Separate Road → IPM → Lane Detection
    # =========================================================================
    if args.ipm:
        ipm_x_range = (args.ipm_x_min, args.ipm_x_max)
        ipm_y_range = (args.ipm_y_min, args.ipm_y_max)
        
        print('Pipeline: Road Mask → Separate Road → IPM → Lane Detection')
        print(f'IPM range: X=[{args.ipm_x_min}, {args.ipm_x_max}]m, Y=[{args.ipm_y_min}, {args.ipm_y_max}]m')
        print(f'Resolution: {args.pixels_per_meter} pixels/meter')
        
        # Run the pipeline
        road_only, ipm_road_image, ipm_lane_mask, ipm_road_mask, ipm_size = extractor.extract_lanes_ipm(
            image_rgb,
            road_mask,
            fov=None,
            pitch=None,
            camera_height=None,
            ipm_x_range=ipm_x_range,
            ipm_y_range=ipm_y_range,
            pixels_per_meter=args.pixels_per_meter
        )
        
        print(f'Final IPM size: {ipm_size[0]}x{ipm_size[1]} pixels')

        # Save separated road image (perspective)
        road_only_bgr = cv2.cvtColor(road_only, cv2.COLOR_RGB2BGR)
        road_only_path = out_dir / f'{base_name}_road_only.png'
        cv2.imwrite(str(road_only_path), road_only_bgr)
        print(f'Saved separated road: {road_only_path}')

        # Save IPM road image (separated road in inverse perspective mapping)
        ipm_bgr = cv2.cvtColor(ipm_road_image, cv2.COLOR_RGB2BGR)
        ipm_path = out_dir / f'{base_name}_ipm.png'
        cv2.imwrite(str(ipm_path), ipm_bgr)
        print(f'Saved IPM road image: {ipm_path}')
        
        # Save original road mask
        if args.save_mask:
            orig_mask_path = out_dir / f'{base_name}_road_mask.png'
            cv2.imwrite(str(orig_mask_path), road_mask)
            print(f'Saved original road mask: {orig_mask_path}')
            
            ipm_mask_path = out_dir / f'{base_name}_ipm_road_mask.png'
            cv2.imwrite(str(ipm_mask_path), ipm_road_mask)
            print(f'Saved IPM road mask: {ipm_mask_path}')
        
        # Save IPM lane mask
        if args.save_lanes:
            ipm_lane_path = out_dir / f'{base_name}_ipm_lanes.png'
            cv2.imwrite(str(ipm_lane_path), ipm_lane_mask)
            print(f'Saved IPM lane mask: {ipm_lane_path}')
        
        # Save IPM lane overlay
        if args.save_lane_overlay:
            # Create lane overlay on IPM image (Red lanes)
            ipm_lane_overlay = ipm_road_image.copy()
            ipm_lane_overlay[ipm_lane_mask == 255] = (255, 0, 0)  # Red
            
            # Blend
            ipm_lane_overlay_blended = cv2.addWeighted(
                ipm_road_image, 1 - args.alpha, 
                ipm_lane_overlay, args.alpha, 0
            )
            
            ipm_lane_overlay_bgr = cv2.cvtColor(ipm_lane_overlay_blended, cv2.COLOR_RGB2BGR)
            ipm_lane_overlay_path = out_dir / f'{base_name}_ipm_lanes_overlay.png'
            cv2.imwrite(str(ipm_lane_overlay_path), ipm_lane_overlay_bgr)
            print(f'Saved IPM lane overlay: {ipm_lane_overlay_path}')
        
        # Save IPM road overlay
        if args.overlay:
            ipm_road_overlay = ipm_road_image.copy()
            ipm_road_overlay[ipm_road_mask == 255] = (128, 64, 128)  # Purple
            
            ipm_road_overlay_blended = cv2.addWeighted(
                ipm_road_image, 1 - args.alpha, 
                ipm_road_overlay, args.alpha, 0
            )
            
            ipm_road_overlay_bgr = cv2.cvtColor(ipm_road_overlay_blended, cv2.COLOR_RGB2BGR)
            ipm_road_overlay_path = out_dir / f'{base_name}_ipm_road_overlay.png'
            cv2.imwrite(str(ipm_road_overlay_path), ipm_road_overlay_bgr)
            print(f'Saved IPM road overlay: {ipm_road_overlay_path}')
        
        # Save IPM grayscale road texture (road pixels only in grayscale)
        if args.save_road_gray:
            # Convert IPM road image to grayscale
            ipm_road_gray = cv2.cvtColor(ipm_road_image, cv2.COLOR_RGB2GRAY)
            ipm_road_gray_path = out_dir / f'{base_name}_ipm_road_grayscale.png'
            cv2.imwrite(str(ipm_road_gray_path), ipm_road_gray)
            print(f'Saved IPM grayscale road: {ipm_road_gray_path}')
        
        # Display IPM results if requested
        if args.show:
            # Create side-by-side comparison
            # Resize original to match IPM height for comparison
            ipm_h = ipm_size[1]
            scale = ipm_h / image_bgr.shape[0]
            orig_resized = cv2.resize(image_bgr, None, fx=scale, fy=scale)
            
            # Create IPM lane overlay for display
            ipm_display = ipm_bgr.copy()
            ipm_display[ipm_lane_mask == 255] = (0, 0, 255)  # Red in BGR
            
            # Pad original if needed to match heights
            if orig_resized.shape[0] != ipm_display.shape[0]:
                pad = abs(orig_resized.shape[0] - ipm_display.shape[0])
                if orig_resized.shape[0] < ipm_display.shape[0]:
                    orig_resized = cv2.copyMakeBorder(
                        orig_resized, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
                    )
                else:
                    ipm_display = cv2.copyMakeBorder(
                        ipm_display, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
                    )
            
            display = np.hstack([orig_resized, ipm_display])
            cv2.imshow('Original | IPM with Lanes', display)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return 0
    
    # =========================================================================
    # Normal Processing Mode (Original Image)
    # =========================================================================
    # Use provided road mask
    
    # Save binary mask
    if args.save_mask:
        mask_path = out_dir / f'{base_name}_road_mask.png'
        cv2.imwrite(str(mask_path), road_mask)
        print(f'Saved road mask: {mask_path}')

    # Save colored road (optional)
    if args.colored:
        colored = extractor.visualize_road(image_rgb, road_mask, mode='colored')
        colored_bgr = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
        colored_path = out_dir / f'{base_name}_road_colored.png'
        cv2.imwrite(str(colored_path), colored_bgr)
        print(f'Saved colored road: {colored_path}')
    # Save overlay (optional)
    if args.overlay:
        overlay = extractor.visualize_road(image_rgb, road_mask, mode='overlay', alpha=args.alpha)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        overlay_path = out_dir / f'{base_name}_road_overlay.png'
        cv2.imwrite(str(overlay_path), overlay_bgr)
        print(f'Saved overlay: {overlay_path}')
    
    # Save lane markings and overlays
    if args.save_lanes or args.save_lane_overlay:
        lane_mask = extractor.extract_lanes(image_rgb, road_mask)
        
        # Save Grayscale Threshold result
        if args.save_lanes:
            lane_path = out_dir / f'{base_name}_lanes.png'
            cv2.imwrite(str(lane_path), lane_mask)
            print(f'Saved separated lanes (Grayscale Threshold): {lane_path}')
            
        # Save Lane Overlay
        if args.save_lane_overlay:
            # Create lane overlay (Red lanes for better visibility)
            lane_overlay = image_rgb.copy()
            lane_overlay[lane_mask == 255] = (255, 0, 0)
            
            # Blend
            lane_overlay_blended = cv2.addWeighted(image_rgb, 1 - args.alpha, lane_overlay, args.alpha, 0)
            
            lane_overlay_bgr = cv2.cvtColor(lane_overlay_blended, cv2.COLOR_RGB2BGR)
            lane_overlay_path = out_dir / f'{base_name}_lanes_overlay.png'
            cv2.imwrite(str(lane_overlay_path), lane_overlay_bgr)
            print(f'Saved lane overlay: {lane_overlay_path}')
    
    # Save Grayscale Road Texture (Requested)
    if args.save_road_gray:
        road_rgb = extractor.extract_road_only(image_rgb, road_mask)
        road_gray = cv2.cvtColor(road_rgb, cv2.COLOR_RGB2GRAY)
        road_gray_path = out_dir / f'{base_name}_road_grayscale.png'
        cv2.imwrite(str(road_gray_path), road_gray)
        print(f'Saved grayscale road: {road_gray_path}')

    # Display if requested
    if args.show:
        display = np.hstack([
            image_bgr,
            cv2.cvtColor(road_mask, cv2.COLOR_GRAY2BGR),
        ])
        cv2.imshow('Original | Road Mask', display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return 0


def process_folder(extractor: RoadExtractor, args):
    """Process all images in a folder.
    
    Note: For batch processing, segmentation maps must be provided in a parallel folder
    structure (or same folder) with matching filenames.
    """
    folder = Path(args.folder)
    
    if not folder.exists():
        print(f'Error: Folder not found: {folder}')
        return 1
    
    # Find all images
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    images = []
    for ext in extensions:
        images.extend(folder.glob(ext))
        images.extend(folder.glob(ext.upper()))
    
    images = sorted(set(images))
    
    if not images:
        print(f'Error: No images found in: {folder}')
        return 1
    
    print(f'Found {len(images)} images to process')
    
    # Segmentation map folder
    seg_folder = Path(args.seg_folder) if args.seg_folder else None
    if seg_folder and not seg_folder.exists():
        print(f'Error: Segmentation map folder not found: {seg_folder}')
        return 1
    
    # Output directory
    if args.output:
        out_dir = Path(args.output)
    else:
        out_dir = CONFIG.output.default_dir
        
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    for i, img_path in enumerate(images):
        print(f'[{i+1}/{len(images)}] Processing: {img_path.name}')
        
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f'Warning: Skipping: Could not load image: {img_path.name}')
            continue
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        base_name = img_path.stem
        
        # Load corresponding segmentation map
        if seg_folder:
            # Look for map in folder with same name or similar
            # Try exact name first, then common suffixes
            map_candidates = [
                seg_folder / f'{base_name}.png',
                seg_folder / f'{base_name}_seg.png',
                seg_folder / f'{base_name}_mask.png',
            ]
            seg_map = None
            for map_path in map_candidates:
                if map_path.exists():
                    seg_map = cv2.imread(str(map_path), cv2.IMREAD_GRAYSCALE)
                    break
            
            if seg_map is None:
                print(f'Warning: Skipping: No segmentation map found for {img_path.name}')
                continue
                
            # Extract road mask
            road_mask = extractor.extract_road_mask(seg_map)
            
        else:
            print(f'Warning: Skipping: No segmentation map folder provided. Use --seg-folder.')
            continue
        
        # Save binary mask
        if args.save_mask:
            mask_path = out_dir / f'{base_name}_road_mask.png'
            cv2.imwrite(str(mask_path), road_mask)
        
        # Save colored road (optional)
        if args.colored:
            colored = extractor.visualize_road(image_rgb, road_mask, mode='colored')
            colored_bgr = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
            colored_path = out_dir / f'{base_name}_road_colored.png'
            cv2.imwrite(str(colored_path), colored_bgr)
        
        # Save overlay (optional)
        if args.overlay:
            overlay = extractor.visualize_road(image_rgb, road_mask, mode='overlay', alpha=args.alpha)
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            overlay_path = out_dir / f'{base_name}_road_overlay.png'
            cv2.imwrite(str(overlay_path), overlay_bgr)
        
        # Save lane markings and overlays
        if args.save_lanes or args.save_lane_overlay:
            lane_mask = extractor.extract_lanes(image_rgb, road_mask)
            
            # Save Grayscale Threshold result
            if args.save_lanes:
                lane_path = out_dir / f'{base_name}_lanes.png'
                cv2.imwrite(str(lane_path), lane_mask)
                print(f'Saved separated lanes: {lane_path}')

            # Save Lane Overlay
            if args.save_lane_overlay:
                # Create lane overlay (Red lanes for better visibility)
                lane_overlay = image_rgb.copy()
                lane_overlay[lane_mask == 255] = (255, 0, 0)
                
                # Blend
                lane_overlay_blended = cv2.addWeighted(image_rgb, 1 - args.alpha, lane_overlay, args.alpha, 0)
                
                lane_overlay_bgr = cv2.cvtColor(lane_overlay_blended, cv2.COLOR_RGB2BGR)
                lane_overlay_path = out_dir / f'{base_name}_lanes_overlay.png'
                cv2.imwrite(str(lane_overlay_path), lane_overlay_bgr)

        # Save Grayscale Road Texture (Requested)
        if args.save_road_gray:
            road_rgb = extractor.extract_road_only(image_rgb, road_mask)
            road_gray = cv2.cvtColor(road_rgb, cv2.COLOR_RGB2GRAY)
            road_gray_path = out_dir / f'{base_name}_road_grayscale.png'
            cv2.imwrite(str(road_gray_path), road_gray)
    
    print(f'Done! Results saved to: {out_dir}')
    return 0


def main():
    """CLI entry point for road extraction.
    
    This CLI is model-agnostic and requires pre-computed road masks or segmentation.
    
    Input Requirements:
        --seg-map-path        Path to pre-computed segmentation map (for single image)
        --seg-folder          Path to folder with segmentation maps (for batch processing)

    Output Flags:
        --save-mask           Save binary road mask
        --save-lanes          Save extracted lane binary mask
        --save-lane-overlay   Save lane overlay (Red lanes)
        --save-road-gray      Save grayscale road texture
        --colored             Save colored road visualization
        --overlay             Save road overlay on original image
        --all                 Save all outputs (enables all above flags)

    Inverse Perspective Mapping (IPM) Mode (DEFAULT):
        IPM mode is enabled by default for better lane detection accuracy.
        Camera params (fov, pitch, height) are not configurable via CLI —
        they come from camera_info topic and tf lookup in the ROS pipeline.
        --no-ipm              To run it without taking IPM
        --pixels-per-meter    IPM resolution (default: 30)
        --ipm-x-min/max       IPM lateral range in meters
        --ipm-y-min/max       IPM forward range in meters

    Logging:
        --verbose, -v         Enable debug logging
        --quiet, -q           Suppress info messages, only show warnings/errors
    """
    
    cli_examples = """
Examples:
    # Single image with pre-computed segmentation map - IPM mode (default)
    python -m perception.core.road_extraction --image input.jpg --seg-map-path seg_map.png --output ./output/
    
    # Single image with all outputs
    python -m perception.core.road_extraction --image input.jpg --seg-map-path seg_map.png --all
    
    # To run it without taking IPM
    python -m perception.core.road_extraction --image input.jpg --seg-map-path seg_map.png --no-ipm --save-lanes --save-lane-overlay
    
    # Batch process folder with masks
    python -m perception.core.road_extraction --folder /data/images/ --seg-folder /data/masks/ --output ./road_masks/
    
    # Quiet mode (only warnings/errors)
    python -m perception.core.road_extraction --folder /data/images/ --seg-folder /data/masks/ --all -q
"""
    
    parser = argparse.ArgumentParser(
        description='Extract road mask from images using pre-computed segmentation (model-agnostic)',
        epilog=cli_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str, help='Path to single image')
    input_group.add_argument('--folder', type=str, help='Path to folder of images')
    
    # Mask input options (required for processing)
    parser.add_argument('--seg-map-path', type=str, dest='seg_map_path',
                        help='Path to pre-computed segmentation map file (for single image)')
    parser.add_argument('--seg-folder', type=str, dest='seg_folder',
                        help='Path to folder containing segmentation maps (for batch processing)')
    parser.add_argument('--road-grayscale-id', type=int, default=CONFIG.road_grayscale_id, dest='road_grayscale_id',
                        help=f'Pixel/Grayscale intensity value for road in segmentation map. (default: {CONFIG.road_grayscale_id})')
    
    # Output options
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--colored', action='store_true', help='Save colored road visualization')
    parser.add_argument('--overlay', action='store_true', help='Save overlay on original image')
    parser.add_argument('--road-only', action='store_true', dest='road_only', 
                        help='Save original image with only road visible')
    parser.add_argument('--alpha', type=float, default=0.5, help='Overlay blend factor (default: 0.5)')
    
    # Granular save options
    parser.add_argument('--save-mask', action='store_true', help='Save binary road mask')
    parser.add_argument('--save-lanes', action='store_true', help='Save extracted lane binary mask')
    parser.add_argument('--save-lane-overlay', action='store_true', help='Save lane overlay')
    parser.add_argument('--save-road-gray', action='store_true', help='Save grayscale road texture')
    parser.add_argument('--all', action='store_true', help='Save all outputs')
    
    # =========================================================================
    # Inverse Perspective Mapping (IPM) Options
    # =========================================================================
    parser.add_argument('--ipm', action='store_true', default=False, dest='ipm',
                        help='Enable IPM mode (requires camera params from node, not available in CLI)')
    parser.add_argument('--no-ipm', action='store_false', dest='ipm',
                        help='Disable IPM mode (default in CLI)')
    parser.add_argument('--ipm-x-min', type=float, default=CONFIG.ipm.x_range[0], dest='ipm_x_min',
                        help=f'IPM near boundary in meters (default: {CONFIG.ipm.x_range[0]})')
    parser.add_argument('--ipm-x-max', type=float, default=CONFIG.ipm.x_range[1], dest='ipm_x_max',
                        help=f'IPM far boundary in meters (default: {CONFIG.ipm.x_range[1]})')
    parser.add_argument('--ipm-y-min', type=float, default=CONFIG.ipm.y_range[0], dest='ipm_y_min',
                        help=f'IPM right boundary in meters (default: {CONFIG.ipm.y_range[0]})')
    parser.add_argument('--ipm-y-max', type=float, default=CONFIG.ipm.y_range[1], dest='ipm_y_max',
                        help=f'IPM left boundary in meters (default: {CONFIG.ipm.y_range[1]})')
    parser.add_argument('--pixels-per-meter', type=float, default=CONFIG.ipm.pixels_per_meter, dest='pixels_per_meter',
                        help=f'IPM resolution in pixels per meter (default: {CONFIG.ipm.pixels_per_meter}). Higher = more detail.')
    
    # Display
    parser.add_argument('--show', action='store_true', help='Display results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose/debug logging')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress info messages, only show warnings/errors')
    args = parser.parse_args()

    # IPM mode in CLI does not support camera params — they come from the node.
    # CLI IPM will pass None for fov/pitch/camera_height, which extract_lanes_ipm
    # will reject with a clear error.

    # Handle --all flag
    if args.all:
        args.save_mask = True
        args.save_lanes = True
        args.save_lane_overlay = True
        args.save_road_gray = True
        args.colored = True
        args.overlay = True

    # Default behavior if no output flags are provided
    if not (args.save_mask or args.save_lanes or args.save_lane_overlay or 
            args.save_road_gray or args.colored or args.overlay or args.road_only or args.show):
        print("No output format specified. Defaulting to --save-mask and --overlay.")
        args.save_mask = True
        args.overlay = True
    
    # Initialize extractor (model-agnostic)
    print(f'Initializing RoadExtractor (road_grayscale_id={args.road_grayscale_id})...')
    extractor = RoadExtractor(road_grayscale_id=args.road_grayscale_id)
    
    # Process
    if args.image:
        return process_single_image(extractor, args)
    else:
        return process_folder(extractor, args)


if __name__ == '__main__':
    exit(main())

