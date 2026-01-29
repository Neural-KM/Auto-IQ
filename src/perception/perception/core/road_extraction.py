#!/usr/bin/env python3
"""
Road Extraction from SegFormer Segmentation
=============================================

This module uses SegFormer to extract only the road mask from images.
Cityscapes class 0 = road

Usage:
    python -m perception.core.road_extraction --image path/to/image.jpg --output path/to/output/
    python -m perception.core.road_extraction --folder path/to/images/ --output path/to/output/

Output Flags:
    --save-mask           Save binary road mask
    --save-lanes          Save extracted lane binary mask (grayscale threshold)
    --save-lane-overlay   Save lane overlay (Red lanes)
    --save-road-gray      Save grayscale road texture
    --colored             Save colored road visualization
    --overlay             Save road overlay
    --all                 Save all outputs (enables all above flags)

Bird's Eye View (BEV) Mode:
    --bev                 Enable Bird's Eye View transformation
    --fov                 Camera FOV in degrees (default: 110)
    --pitch               Camera pitch in degrees (default: 0)
    --camera-height       Camera height in meters (default: 1.2)
    --bev-width           BEV output width (default: 1000)
    --bev-height          BEV output height (default: 1000)

Examples:
    # Single image - extract road mask
    python -m perception.core.road_extraction --image input.jpg --output ./output/
    
    # Single image with all outputs
    python -m perception.core.road_extraction --image input.jpg --colored --overlay --road-only
    
    # Bird's Eye View lane detection (RECOMMENDED for lane detection)
    python -m perception.core.road_extraction --image input.jpg --bev --save-lanes --save-lane-overlay --show
    
    # BEV with custom camera parameters
    python -m perception.core.road_extraction --image input.jpg --bev --fov 90 --pitch 5 --camera-height 1.5 --all
    
    # Batch process folder
    python -m perception.core.road_extraction --folder /data/images/ --output ./road_masks/
    
    # Use larger model for better accuracy
    python -m perception.core.road_extraction --image input.jpg --model b5
"""

import numpy as np
import cv2
import argparse
import yaml
from pathlib import Path
from typing import Optional, Tuple

from perception.core.segformer import SegFormer, CITYSCAPES_PALETTE

from perception.utils.bev_transform import convert_to_bev
# ============================================================================
# Bird's Eye View (BEV) Transform Functions
# ============================================================================

# Default BEV parameters (single source of truth)
DEFAULT_BEV_X_RANGE = (-10.0, 10.0)   # meters left/right from vehicle center
DEFAULT_BEV_Y_RANGE = (3.0, 30.0)     # meters forward (near to far)
DEFAULT_PIXELS_PER_METER = 30.0       # resolution: 1 meter = 30 pixels


# Road class index in Cityscapes
ROAD_CLASS_ID = 0

# Default output directory
DEFAULT_OUTPUT_DIR = Path('/Auto-IQ/data/rosbag_images_15jan/inference')
DEFAULT_SENSOR_CONFIG = Path('/Auto-IQ/config/carla_sensors.yaml')



class RoadExtractor:
    """
    Extracts road mask from images using SegFormer segmentation.
    """
    
    def __init__(self, model_name: str = 'b0', device: Optional[str] = None):
        """
        Initialize RoadExtractor with SegFormer model.
        
        Args:
            model_name: SegFormer model variant (b0, b1, b2, b3, b4, b5)
            device: 'cuda', 'cpu', or None for auto-detection
        """
        self.segformer = SegFormer(model_name=model_name, device=device)
    
    def extract_road_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Extract binary road mask from image.
        
        Args:
            image: RGB image as numpy array (H, W, 3), uint8
            
        Returns:
            Binary mask (H, W) where 255 = road, 0 = non-road
        """
        # Get segmentation prediction
        segmentation = self.segformer.predict(image)
        
        # Extract road pixels (class 0)
        road_mask = (segmentation == ROAD_CLASS_ID).astype(np.uint8) * 255
        
        return road_mask
    
    def extract_road_with_confidence(
        self, 
        image: np.ndarray, 
        threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract road mask with confidence filtering.
        
        Args:
            image: RGB image (H, W, 3)
            threshold: Minimum confidence threshold for road pixels
            
        Returns:
            road_mask: Binary mask (H, W) where 255 = confident road
            confidence: Confidence scores for road pixels (H, W)
        """
        segmentation, confidence = self.segformer.predict_with_confidence(image)
        
        # Road pixels with high confidence
        is_road = segmentation == ROAD_CLASS_ID
        is_confident = confidence >= threshold
        
        road_mask = (is_road & is_confident).astype(np.uint8) * 255
        road_confidence = confidence * is_road
        
        return road_mask, road_confidence
    
    def extract_road_colored(self, image: np.ndarray) -> np.ndarray:
        """
        Extract road and return colored visualization.
        
        Args:
            image: RGB image (H, W, 3)
            
        Returns:
            RGB image with road colored (purple) and rest black
        """
        road_mask = self.extract_road_mask(image)
        
        # Create colored output
        colored = np.zeros_like(image)
        colored[road_mask == 255] = CITYSCAPES_PALETTE[ROAD_CLASS_ID]  # Purple for road
        
        return colored
    
    def extract_road_overlay(
        self, 
        image: np.ndarray, 
        alpha: float = 0.5,
        road_color: Tuple[int, int, int] = (128, 64, 128)
    ) -> np.ndarray:
        """
        Overlay road mask on original image.
        
        Args:
            image: RGB image (H, W, 3)
            alpha: Blend factor for overlay
            road_color: RGB color for road overlay
            
        Returns:
            RGB image with road overlay
        """
        road_mask = self.extract_road_mask(image)
        
        # Create overlay
        overlay = image.copy()
        overlay[road_mask == 255] = road_color
        
        # Blend
        result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
        
        return result
    
    def extract_road_only(self, image: np.ndarray) -> np.ndarray:
        """
        Extract only road pixels from original image (Separate road from real image).
        
        Args:
            image: RGB image (H, W, 3)
            
        Returns:
            RGB image with only road pixels visible, rest black
        """
        road_mask = self.extract_road_mask(image)
        
        # Separate road from the real image using the mask
        result = cv2.bitwise_and(image, image, mask=road_mask)
        
        return result

    # =========================================================================
    # Bird's Eye View (BEV) Methods
    # =========================================================================
    
    def extract_lanes_bev(
        self, 
        image: np.ndarray,
        fov: float = 110.0,
        pitch: float = 0.0,
        camera_height: float = 1.2,
        bev_x_range: Tuple[float, float] = None,
        bev_y_range: Tuple[float, float] = None,
        pixels_per_meter: float = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
        """
        Extract lane markings using the correct pipeline:
        SegFormer → Road Mask → Separate Road → BEV → Lane Detection
        
        Pipeline:
        1. Run SegFormer on original image → get road mask
        2. Separate road from image (road_only = image AND mask)
        3. Convert separated road image to BEV
        4. Find lane markings in BEV image
        
        Args:
            image: RGB image (H, W, 3)
            fov: Camera horizontal field of view in degrees
            pitch: Camera pitch angle in degrees
            camera_height: Camera height above ground in meters
            bev_x_range: (x_min, x_max) range in meters (left/right). Uses DEFAULT_BEV_X_RANGE if None.
            bev_y_range: (y_min, y_max) range in meters (forward distance). Uses DEFAULT_BEV_Y_RANGE if None.
            pixels_per_meter: Resolution - pixels per meter. Uses DEFAULT_PIXELS_PER_METER if None.
            
        Returns:
            road_only: Separated road image (original perspective)
            bev_road_image: Bird's Eye View of separated road
            bev_lane_mask: Lane markings mask in BEV space
            original_road_mask: Road mask from SegFormer (perspective)
            bev_road_mask: Road mask in BEV space
            bev_size: (width, height) of the BEV output
        """
        height, width = image.shape[:2]
        
        # 1. Run SegFormer on ORIGINAL image → get road mask
        print("  Step 1: Running SegFormer to get road mask...")
        original_road_mask = self.extract_road_mask(image)
        
        # 2. Separate road from image (keep only road pixels)
        print("  Step 2: Separating road from image...")
        road_only = cv2.bitwise_and(image, image, mask=original_road_mask)
        
        # 3. Convert separated road image to BEV (Using Camera Geometry)
        # Dimensions are auto-calculated from ranges and pixels_per_meter
        print("  Step 3: Converting separated road to BEV...")
        bev_road_image, bev_size, M = convert_to_bev(
            road_only,
            fov=fov,
            pitch=pitch,
            camera_height=camera_height,
            bev_x_range=bev_x_range,
            bev_y_range=bev_y_range,
            pixels_per_meter=pixels_per_meter,
            return_transform=True
        )
        
        print(f"  BEV size: {bev_size[0]}x{bev_size[1]} pixels (auto-calculated for equal scale)")
        
        # Also transform road mask to BEV for reference
        bev_road_mask = cv2.warpPerspective(
            original_road_mask, M, bev_size, 
            flags=cv2.INTER_NEAREST
        )
        
        # 5. Find lane markings in BEV image
        print("  Step 4: Detecting lane markings in BEV...")
        bev_lane_mask = self._extract_lanes_from_bev(bev_road_image, bev_road_mask)
        
        return road_only, bev_road_image, bev_lane_mask, original_road_mask, bev_road_mask, bev_size
    
    def _extract_lanes_from_bev(
        self, 
        bev_image: np.ndarray, 
        bev_road_mask: np.ndarray
    ) -> np.ndarray:
        """
        Extract lane markings from BEV image using improved multi-scale adaptive thresholding.
        
        Args:
            bev_image: BEV transformed RGB image
            bev_road_mask: Binary road mask in BEV space
            
        Returns:
            Binary lane mask (H, W) where 255 = lane markings
        """
        # Extract road ROI
        road_only = cv2.bitwise_and(bev_image, bev_image, mask=bev_road_mask)
        
        # Convert to grayscale
        gray = cv2.cvtColor(road_only, cv2.COLOR_RGB2GRAY)
        
        # =====================================================================
        # IMPROVEMENT 1: Use CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # This enhances local contrast making lane markings more visible
        # against varying road surfaces (shadows, different asphalt colors)
        # =====================================================================
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        
        # Fill non-road areas with mean road intensity (prevents edge artifacts)
        road_pixels = enhanced_gray[bev_road_mask == 255]
        if len(road_pixels) > 0:
            mean_val = np.mean(road_pixels)
        else:
            mean_val = 128
        filled_gray = enhanced_gray.copy()
        filled_gray[bev_road_mask == 0] = int(mean_val)
        
        # =====================================================================
        # IMPROVEMENT 2: Multi-scale Gaussian blur for noise reduction
        # Light blur preserves lane edges, heavier blur removes small noise
        # =====================================================================
        blurred_light = cv2.GaussianBlur(filled_gray, (3, 3), 0)  # Preserve fine details
        blurred_medium = cv2.GaussianBlur(filled_gray, (5, 5), 0)  # Balance
        
        # =====================================================================
        # IMPROVEMENT 3: Multi-scale Adaptive Thresholding
        # Different block sizes capture lanes at different distances/scales
        # Small block (11): Captures thin/distant lane markings
        # Medium block (21): General lane detection  
        # Large block (31): Captures wide lane markings, less affected by noise
        # =====================================================================
        
        # Small scale - good for thin lane markings
        lane_mask_small = cv2.adaptiveThreshold(
            blurred_light,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,   # Small block size
            -8    # C value: detect pixels brighter than local mean + 8
        )
        
        # Medium scale - balanced detection
        lane_mask_medium = cv2.adaptiveThreshold(
            blurred_medium,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,   # Medium block size
            -12   # C value
        )
        
        # Large scale - for thick lane markings, robust to noise
        lane_mask_large = cv2.adaptiveThreshold(
            blurred_medium,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,   # Large block size
            -15   # C value
        )
        
        # =====================================================================
        # IMPROVEMENT 4: Statistical thresholding with Otsu
        # Otsu automatically finds optimal threshold - good for high contrast lanes
        # =====================================================================
        if len(road_pixels) > 100:
            # Apply Otsu only on road pixels for better threshold estimation
            _, lane_mask_otsu = cv2.threshold(
                filled_gray, 0, 255, 
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            lane_mask_otsu = np.zeros_like(gray)
        
        # =====================================================================
        # IMPROVEMENT 5: Top-hat transform to extract bright regions
        # This is excellent for extracting lane markings (bright on dark road)
        # Removes large-scale illumination variations
        # =====================================================================
        kernel_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        tophat = cv2.morphologyEx(filled_gray, cv2.MORPH_TOPHAT, kernel_tophat)
        
        # Threshold the top-hat result
        if len(road_pixels) > 0:
            tophat_thresh = np.mean(tophat[bev_road_mask == 255]) + 2.0 * np.std(tophat[bev_road_mask == 255])
            _, lane_mask_tophat = cv2.threshold(tophat, max(tophat_thresh, 20), 255, cv2.THRESH_BINARY)
        else:
            lane_mask_tophat = np.zeros_like(gray)
        
        # =====================================================================
        # IMPROVEMENT 6: Smart combination using weighted voting
        # Instead of simple OR, use voting - pixel is lane if detected by
        # at least 2 methods (reduces false positives)
        # =====================================================================
        # Convert to binary 0/1 for voting
        votes = (
            (lane_mask_small > 0).astype(np.uint8) +
            (lane_mask_medium > 0).astype(np.uint8) +
            (lane_mask_large > 0).astype(np.uint8) +
            (lane_mask_otsu > 0).astype(np.uint8) +
            (lane_mask_tophat > 0).astype(np.uint8)
        )
        
        # Require at least 3 votes (detected by 3+ methods)
        lane_mask = ((votes >= 2) * 255).astype(np.uint8)
        
        # =====================================================================
        # IMPROVEMENT 7: Enhanced morphological cleanup
        # =====================================================================
        # Remove small isolated noise (opening)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel_open)
        
        # Connect nearby lane segments (closing with rectangular kernel for lanes)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))  # Vertical bias for lanes
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel_close)
        
        # =====================================================================
        # IMPROVEMENT 8: Remove small blobs using connected components
        # Lanes have minimum area - remove tiny noise blobs
        # =====================================================================
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(lane_mask, connectivity=8)
        min_area = 50  # Minimum lane marking area in pixels
        
        # Create clean mask keeping only components above minimum area
        lane_mask_clean = np.zeros_like(lane_mask)
        for i in range(1, num_labels):  # Skip background (label 0)
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                lane_mask_clean[labels == i] = 255
        
        # Enforce road mask
        lane_mask_clean = cv2.bitwise_and(lane_mask_clean, bev_road_mask)
        
        return lane_mask_clean
    
    def extract_road_bev(
        self, 
        image: np.ndarray,
        fov: float = 110.0,
        pitch: float = 0.0,
        camera_height: float = 1.2,
        bev_x_range: Tuple[float, float] = None,
        bev_y_range: Tuple[float, float] = None,
        pixels_per_meter: float = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract road from Bird's Eye View transformed image.
        
        Args:
            image: RGB image (H, W, 3)
            fov: Camera horizontal field of view in degrees
            pitch: Camera pitch angle in degrees
            camera_height: Camera height above ground in meters
            bev_x_range: (x_min, x_max) range in meters (left/right). Uses DEFAULT_BEV_X_RANGE if None.
            bev_y_range: (y_min, y_max) range in meters (forward distance). Uses DEFAULT_BEV_Y_RANGE if None.
            pixels_per_meter: Resolution - pixels per meter. Uses DEFAULT_PIXELS_PER_METER if None.
            
        Returns:
            bev_image: Bird's Eye View transformed image
            bev_road_mask: Road mask in BEV space
        """
        # Convert to Bird's Eye View
        bev_image, _, _ = convert_to_bev(
            image, fov, pitch, camera_height,
            bev_x_range, bev_y_range, pixels_per_meter
        )
        
        # Extract road mask from BEV image
        bev_road_mask = self.extract_road_mask(bev_image)
        
        return bev_image, bev_road_mask
    
    def extract_lanes(self, image: np.ndarray) -> np.ndarray:
        """
        Extract lane markings using Adaptive Thresholding on the Road ROI.
        
        Args:
            image: RGB image (H, W, 3)
            
        Returns:
            Binary mask (H, W) where 255 = lane markings
        """
        # 1. Get road mask
        road_mask = self.extract_road_mask(image)
        
        # 2. Extract road ROI
        road_only = cv2.bitwise_and(image, image, mask=road_mask)
        
        # 3. Convert to Grayscale
        gray = cv2.cvtColor(road_only, cv2.COLOR_RGB2GRAY)
        
        # 4. Apply Adaptive Thresholding
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
        
        # 5. Clean up with morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel)
        
        # Enforce ROI
        lane_mask = cv2.bitwise_and(lane_mask, road_mask)
        
        return lane_mask


def process_single_image(extractor: RoadExtractor, args):
    """Process a single image."""
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
    
    # Determine output paths
    if args.output:
        out_dir = Path(args.output)
    else:
        out_dir = DEFAULT_OUTPUT_DIR
        
    out_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = img_path.stem
    
    # =========================================================================
    # Bird's Eye View (BEV) Processing Mode
    # Pipeline: SegFormer → Road Mask → Separate Road → BEV → Lane Detection
    # =========================================================================
    if args.bev:
        bev_x_range = (args.bev_x_min, args.bev_x_max)
        bev_y_range = (args.bev_y_min, args.bev_y_max)
        
        print(f'Pipeline: SegFormer → Road Mask → Separate Road → BEV → Lane Detection')
        print(f'BEV range: X=[{args.bev_x_min}, {args.bev_x_max}]m, Y=[{args.bev_y_min}, {args.bev_y_max}]m')
        print(f'Resolution: {args.pixels_per_meter} pixels/meter')
        
        # Run the pipeline
        road_only, bev_road_image, bev_lane_mask, original_road_mask, bev_road_mask, bev_size = extractor.extract_lanes_bev(
            image_rgb,
            fov=args.fov,
            pitch=args.pitch,
            camera_height=args.camera_height,
            bev_x_range=bev_x_range,
            bev_y_range=bev_y_range,
            pixels_per_meter=args.pixels_per_meter
        )
        
        print(f'Final BEV size: {bev_size[0]}x{bev_size[1]} pixels')
        
        # Save separated road image (perspective)
        road_only_bgr = cv2.cvtColor(road_only, cv2.COLOR_RGB2BGR)
        road_only_path = out_dir / f'{base_name}_road_only.png'
        cv2.imwrite(str(road_only_path), road_only_bgr)
        print(f'Saved separated road: {road_only_path}')
        
        # Save BEV road image (separated road in bird's eye view)
        bev_bgr = cv2.cvtColor(bev_road_image, cv2.COLOR_RGB2BGR)
        bev_path = out_dir / f'{base_name}_bev.png'
        cv2.imwrite(str(bev_path), bev_bgr)
        print(f'Saved BEV road image: {bev_path}')
        
        # Save original road mask (from SegFormer on perspective image)
        if args.save_mask:
            orig_mask_path = out_dir / f'{base_name}_road_mask.png'
            cv2.imwrite(str(orig_mask_path), original_road_mask)
            print(f'Saved original road mask: {orig_mask_path}')
            
            bev_mask_path = out_dir / f'{base_name}_bev_road_mask.png'
            cv2.imwrite(str(bev_mask_path), bev_road_mask)
            print(f'Saved BEV road mask: {bev_mask_path}')
        
        # Save BEV lane mask
        if args.save_lanes:
            bev_lane_path = out_dir / f'{base_name}_bev_lanes.png'
            cv2.imwrite(str(bev_lane_path), bev_lane_mask)
            print(f'Saved BEV lane mask: {bev_lane_path}')
        
        # Save BEV lane overlay
        if args.save_lane_overlay:
            # Create lane overlay on BEV image (Red lanes)
            bev_lane_overlay = bev_road_image.copy()
            bev_lane_overlay[bev_lane_mask == 255] = (255, 0, 0)  # Red
            
            # Blend
            bev_lane_overlay_blended = cv2.addWeighted(
                bev_road_image, 1 - args.alpha, 
                bev_lane_overlay, args.alpha, 0
            )
            
            bev_lane_overlay_bgr = cv2.cvtColor(bev_lane_overlay_blended, cv2.COLOR_RGB2BGR)
            bev_lane_overlay_path = out_dir / f'{base_name}_bev_lanes_overlay.png'
            cv2.imwrite(str(bev_lane_overlay_path), bev_lane_overlay_bgr)
            print(f'Saved BEV lane overlay: {bev_lane_overlay_path}')
        
        # Save BEV road overlay
        if args.overlay:
            bev_road_overlay = bev_road_image.copy()
            bev_road_overlay[bev_road_mask == 255] = (128, 64, 128)  # Purple
            
            bev_road_overlay_blended = cv2.addWeighted(
                bev_road_image, 1 - args.alpha, 
                bev_road_overlay, args.alpha, 0
            )
            
            bev_road_overlay_bgr = cv2.cvtColor(bev_road_overlay_blended, cv2.COLOR_RGB2BGR)
            bev_road_overlay_path = out_dir / f'{base_name}_bev_road_overlay.png'
            cv2.imwrite(str(bev_road_overlay_path), bev_road_overlay_bgr)
            print(f'Saved BEV road overlay: {bev_road_overlay_path}')
        
        # Save BEV grayscale road texture (road pixels only in grayscale)
        if args.save_road_gray:
            # Convert BEV road image to grayscale
            bev_road_gray = cv2.cvtColor(bev_road_image, cv2.COLOR_RGB2GRAY)
            bev_road_gray_path = out_dir / f'{base_name}_bev_road_grayscale.png'
            cv2.imwrite(str(bev_road_gray_path), bev_road_gray)
            print(f'Saved BEV grayscale road: {bev_road_gray_path}')
        
        # Display BEV results if requested
        if args.show:
            # Create side-by-side comparison
            # Resize original to match BEV height for comparison
            bev_h = bev_size[1]
            scale = bev_h / image_bgr.shape[0]
            orig_resized = cv2.resize(image_bgr, None, fx=scale, fy=scale)
            
            # Create BEV lane overlay for display
            bev_display = bev_bgr.copy()
            bev_display[bev_lane_mask == 255] = (0, 0, 255)  # Red in BGR
            
            # Pad original if needed to match heights
            if orig_resized.shape[0] != bev_display.shape[0]:
                pad = abs(orig_resized.shape[0] - bev_display.shape[0])
                if orig_resized.shape[0] < bev_display.shape[0]:
                    orig_resized = cv2.copyMakeBorder(
                        orig_resized, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
                    )
                else:
                    bev_display = cv2.copyMakeBorder(
                        bev_display, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
                    )
            
            display = np.hstack([orig_resized, bev_display])
            cv2.imshow('Original | BEV with Lanes', display)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return 0
    
    # =========================================================================
    # Normal Processing Mode (Original Image)
    # =========================================================================
    # Extract road mask
    road_mask = extractor.extract_road_mask(image_rgb)
    
    # Save binary mask
    if args.save_mask:
        mask_path = out_dir / f'{base_name}_road_mask.png'
        cv2.imwrite(str(mask_path), road_mask)
        print(f'Saved road mask: {mask_path}')
    
    # Save colored road (optional)
    if args.colored:
        colored = extractor.extract_road_colored(image_rgb)
        colored_bgr = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
        colored_path = out_dir / f'{base_name}_road_colored.png'
        cv2.imwrite(str(colored_path), colored_bgr)
        print(f'Saved colored road: {colored_path}')
    # Save overlay (optional)
    if args.overlay:
        overlay = extractor.extract_road_overlay(image_rgb, alpha=args.alpha)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        overlay_path = out_dir / f'{base_name}_road_overlay.png'
        cv2.imwrite(str(overlay_path), overlay_bgr)
        print(f'Saved overlay: {overlay_path}')
    
    # Save lane markings and overlays
    if args.save_lanes or args.save_lane_overlay:
        lane_mask = extractor.extract_lanes(image_rgb)
        
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
        road_rgb = extractor.extract_road_only(image_rgb)
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
    """Process all images in a folder."""
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
        print(f'No images found in: {folder}')
        return 1
    
    print(f'Found {len(images)} images to process')
    
    # Output directory
    if args.output:
        out_dir = Path(args.output)
    else:
        out_dir = DEFAULT_OUTPUT_DIR
        
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    for i, img_path in enumerate(images):
        print(f'[{i+1}/{len(images)}] Processing: {img_path.name}')
        
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f'  Skipping: Could not load image')
            continue
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        base_name = img_path.stem
        
        # Save binary mask
        if args.save_mask:
            road_mask = extractor.extract_road_mask(image_rgb)
            mask_path = out_dir / f'{base_name}_road_mask.png'
            cv2.imwrite(str(mask_path), road_mask)
        
        # Save colored road (optional)
        if args.colored:
            colored = extractor.extract_road_colored(image_rgb)
            colored_bgr = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
            colored_path = out_dir / f'{base_name}_road_colored.png'
            cv2.imwrite(str(colored_path), colored_bgr)
        
        # Save overlay (optional)
        if args.overlay:
            overlay = extractor.extract_road_overlay(image_rgb, alpha=args.alpha)
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            overlay_path = out_dir / f'{base_name}_road_overlay.png'
            cv2.imwrite(str(overlay_path), overlay_bgr)
        
        # Save lane markings and overlays
        if args.save_lanes or args.save_lane_overlay:
            lane_mask = extractor.extract_lanes(image_rgb)
            
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

        # Save Grayscale Road Texture (Requested)
        if args.save_road_gray:
            road_rgb = extractor.extract_road_only(image_rgb)
            road_gray = cv2.cvtColor(road_rgb, cv2.COLOR_RGB2GRAY)
            road_gray_path = out_dir / f'{base_name}_road_grayscale.png'
            cv2.imwrite(str(road_gray_path), road_gray)
    
    print(f'\nDone! Results saved to: {out_dir}')
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Extract road mask from images using SegFormer',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str, help='Path to single image')
    input_group.add_argument('--folder', type=str, help='Path to folder of images')
    
    # Output options
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--config', type=str, help='Path to sensor config yaml')
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
    # Bird's Eye View (BEV) Options
    # =========================================================================
    parser.add_argument('--bev', action='store_true', 
                        help='Convert to Bird\'s Eye View before processing')
    parser.add_argument('--fov', type=float, default=None,
                        help='Camera horizontal field of view in degrees (default: 110 or from config)')
    parser.add_argument('--pitch', type=float, default=None,
                        help='Camera pitch angle in degrees (default: 0 or from config)')
    parser.add_argument('--camera-height', type=float, default=None, dest='camera_height',
                        help='Camera height above ground in meters (default: 1.2 or from config)')
    parser.add_argument('--bev-x-min', type=float, default=DEFAULT_BEV_X_RANGE[0], dest='bev_x_min',
                        help=f'BEV left boundary in meters (default: {DEFAULT_BEV_X_RANGE[0]})')
    parser.add_argument('--bev-x-max', type=float, default=DEFAULT_BEV_X_RANGE[1], dest='bev_x_max',
                        help=f'BEV right boundary in meters (default: {DEFAULT_BEV_X_RANGE[1]})')
    parser.add_argument('--bev-y-min', type=float, default=DEFAULT_BEV_Y_RANGE[0], dest='bev_y_min',
                        help=f'BEV near boundary in meters (default: {DEFAULT_BEV_Y_RANGE[0]})')
    parser.add_argument('--bev-y-max', type=float, default=DEFAULT_BEV_Y_RANGE[1], dest='bev_y_max',
                        help=f'BEV far boundary in meters (default: {DEFAULT_BEV_Y_RANGE[1]})')
    parser.add_argument('--pixels-per-meter', type=float, default=DEFAULT_PIXELS_PER_METER, dest='pixels_per_meter',
                        help=f'BEV resolution in pixels per meter (default: {DEFAULT_PIXELS_PER_METER}). Higher = more detail.')
    
    # Model options
    parser.add_argument('--model', type=str, default='b0', 
                        choices=['b0', 'b1', 'b2', 'b3', 'b4', 'b5'],
                        help='SegFormer model variant (default: b0)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                        help='Device to use (default: auto-detect)')
    
    # Display
    parser.add_argument('--show', action='store_true', help='Display results')
    args = parser.parse_args()
    
    # Load defaults from config if available
    config_fov = 110.0
    config_pitch = 0.0
    config_height = 1.2
    
    config_path = Path(args.config) if args.config else DEFAULT_SENSOR_CONFIG
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                if 'cameras' in config and 'front_rgb' in config['cameras']:
                    cam_cfg = config['cameras']['front_rgb']
                    if 'attributes' in cam_cfg and 'fov' in cam_cfg['attributes']:
                        config_fov = float(cam_cfg['attributes']['fov'])
                    if 'rotation' in cam_cfg and len(cam_cfg['rotation']) >= 1:
                        config_pitch = float(cam_cfg['rotation'][0])
                    if 'location' in cam_cfg and len(cam_cfg['location']) >= 3:
                        config_height = float(cam_cfg['location'][2])
                    print(f"Loaded camera config from {config_path}: FOV={config_fov}, Pitch={config_pitch}, Height={config_height}")
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")

    # Set values if not provided by CLI
    if args.fov is None:
        args.fov = config_fov
    if args.pitch is None:
        args.pitch = config_pitch
    if args.camera_height is None:
        args.camera_height = config_height

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
        print("Note: No output format specified. Defaulting to --save-mask and --overlay.")
        args.save_mask = True
        args.overlay = True
    
    # Initialize extractor
    print(f'Loading SegFormer-{args.model} for road extraction...')
    extractor = RoadExtractor(model_name=args.model, device=args.device)
    
    # Process
    if args.image:
        return process_single_image(extractor, args)
    else:
        return process_folder(extractor, args)


if __name__ == '__main__':
    exit(main())

