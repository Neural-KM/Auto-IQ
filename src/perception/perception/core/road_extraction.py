#!/usr/bin/env python3
"""
Road Extraction from SegFormer Segmentation
=============================================

This module uses SegFormer to extract road masks and lane markings from images.
Cityscapes class 0 = road

Core Methods:
    extract_road_mask()         - Binary road mask from SegFormer
    extract_road_with_confidence() - Road mask with confidence filtering
    visualize_road()            - Unified visualization (overlay/colored/mask_only modes)
    extract_lanes()             - Lane detection on perspective images
    extract_lanes_bev()         - Lane detection with Bird's Eye View transform (recommended)

Usage:
    python -m perception.core.road_extraction --image path/to/image.jpg --output path/to/output/
    python -m perception.core.road_extraction --folder path/to/images/ --output path/to/output/

Output Flags:
    --save-mask           Save binary road mask
    --save-lanes          Save extracted lane binary mask
    --save-lane-overlay   Save lane overlay (Red lanes)
    --save-road-gray      Save grayscale road texture
    --colored             Save colored road visualization
    --overlay             Save road overlay on original image
    --all                 Save all outputs (enables all above flags)

Bird's Eye View (BEV) Mode:
    --bev                 Enable Bird's Eye View transformation
    --fov                 Camera FOV in degrees (default: 110)
    --pitch               Camera pitch in degrees (default: 0)
    --camera-height       Camera height in meters (default: 1.2)
    --pixels-per-meter    BEV resolution (default: 30)
    --bev-x-min/max       BEV lateral range in meters
    --bev-y-min/max       BEV forward range in meters

Examples:
    # Single image - extract road mask
    python -m perception.core.road_extraction --image input.jpg --output ./output/
    
    # Single image with all outputs
    python -m perception.core.road_extraction --image input.jpg --colored --overlay --all
    
    # Bird's Eye View lane detection (RECOMMENDED for lane detection)
    python -m perception.core.road_extraction --image input.jpg --bev --save-lanes --save-lane-overlay --show
    
    # BEV with custom camera parameters
    python -m perception.core.road_extraction --image input.jpg --bev --fov 90 --pitch 5 --camera-height 1.5 --all
    
    # Batch process folder
    python -m perception.core.road_extraction --folder /data/images/ --output ./road_masks/
    
    # Use larger model for better accuracy
    python -m perception.core.road_extraction --image input.jpg --model b5
"""

import logging
import numpy as np
import cv2
import argparse
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from perception.core.segformer import SegFormer, CITYSCAPES_PALETTE

# Configure module logger
logger = logging.getLogger(__name__)
from perception.utils.lane_detection import _extract_lanes_from_bev
from perception.utils.bev_transform import convert_to_bev

# ============================================================================
# Configuration (Dataclasses - frozen for immutability)
# ============================================================================

@dataclass(frozen=True)
class BEVConfig:
    """Bird's Eye View transform parameters."""
    x_range: Tuple[float, float] = (-10.0, 10.0)   # meters left/right from vehicle center
    y_range: Tuple[float, float] = (3.0, 30.0)     # meters forward (near to far)
    pixels_per_meter: float = 30.0                  # resolution: 1 meter = 30 pixels


@dataclass(frozen=True)
class CameraConfig:
    """Camera intrinsic/extrinsic parameters."""
    fov: float = 110.0          # horizontal field of view in degrees
    pitch: float = 0.0          # camera pitch in degrees
    height: float = 1.2         # camera height above ground in meters


@dataclass(frozen=True)
class OutputConfig:
    """Output paths configuration."""
    default_dir: Path = field(default_factory=lambda: Path('/Auto-IQ/data/rosbag_images_15jan/inference'))
    sensor_config: Path = field(default_factory=lambda: Path('/Auto-IQ/config/carla_sensors.yaml'))


@dataclass(frozen=True)
class RoadExtractionConfig:
    """Main configuration for road extraction module."""
    model: str = 'b0'                    # SegFormer variant: b0, b1, b2, b3, b4, b5
    device: Optional[str] = None         # 'cuda', 'cpu', or None for auto-detect
    road_class_id: int = 0               # Cityscapes class ID for road
    bev: BEVConfig = field(default_factory=BEVConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


# Default configuration instance
CONFIG = RoadExtractionConfig()


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
        road_mask = (segmentation == CONFIG.road_class_id).astype(np.uint8) * 255
        
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
        is_road = segmentation == CONFIG.road_class_id
        is_confident = confidence >= threshold
        
        road_mask = (is_road & is_confident).astype(np.uint8) * 255
        road_confidence = confidence * is_road
        
        return road_mask, road_confidence
    
    def visualize_road(
        self, 
        image: np.ndarray, 
        mode: str = 'overlay',
        alpha: float = 0.5,
        road_color: Tuple[int, int, int] = (128, 64, 128)
    ) -> np.ndarray:
        """
        Visualize road extraction results.
        
        Args:
            image: RGB image (H, W, 3)
            mode: Visualization mode:
                - 'overlay': Blend road color on original image
                - 'colored': Road colored, rest black
                - 'mask_only': Return original pixels only where road is detected
            alpha: Blend factor for overlay mode (0-1)
            road_color: RGB color for road visualization
            
        Returns:
            RGB visualization image
        """
        road_mask = self.extract_road_mask(image)
        
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
        fov: float = None,
        pitch: float = None,
        camera_height: float = None,
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
            fov: Camera horizontal field of view in degrees. Uses CONFIG.camera.fov if None.
            pitch: Camera pitch angle in degrees. Uses CONFIG.camera.pitch if None.
            camera_height: Camera height above ground in meters. Uses CONFIG.camera.height if None.
            bev_x_range: (x_min, x_max) range in meters (left/right). Uses CONFIG.bev.x_range if None.
            bev_y_range: (y_min, y_max) range in meters (forward distance). Uses CONFIG.bev.y_range if None.
            pixels_per_meter: Resolution - pixels per meter. Uses CONFIG.bev.pixels_per_meter if None.
            
        Returns:
            road_only: Separated road image (original perspective)
            bev_road_image: Bird's Eye View of separated road
            bev_lane_mask: Lane markings mask in BEV space
            original_road_mask: Road mask from SegFormer (perspective)
            bev_road_mask: Road mask in BEV space
            bev_size: (width, height) of the BEV output
        """
        # Use CONFIG defaults if not provided
        fov = fov if fov is not None else CONFIG.camera.fov
        pitch = pitch if pitch is not None else CONFIG.camera.pitch
        camera_height = camera_height if camera_height is not None else CONFIG.camera.height
        
        height, width = image.shape[:2]
        
        # 1. Run SegFormer on ORIGINAL image → get road mask
        logger.debug("Step 1: Running SegFormer to get road mask...")
        original_road_mask = self.extract_road_mask(image)
        
        # 2. Separate road from image (keep only road pixels)
        logger.debug("Step 2: Separating road from image...")
        road_only = cv2.bitwise_and(image, image, mask=original_road_mask)
        
        # 3. Convert separated road image to BEV (Using Camera Geometry)
        # Dimensions are auto-calculated from ranges and pixels_per_meter
        logger.debug("Step 3: Converting separated road to BEV...")
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
        
        logger.debug("BEV size: %dx%d pixels (auto-calculated for equal scale)", bev_size[0], bev_size[1])
        
        # Also transform road mask to BEV for reference
        bev_road_mask = cv2.warpPerspective(
            original_road_mask, M, bev_size, 
            flags=cv2.INTER_NEAREST
        )
        
        # 5. Find lane markings in BEV image
        logger.debug("Step 4: Detecting lane markings in BEV...")
        bev_lane_mask = self._extract_lanes_from_bev(bev_road_image, bev_road_mask)
        
        return road_only, bev_road_image, bev_lane_mask, original_road_mask, bev_road_mask, bev_size
    
    
    
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
        logger.error('Image not found: %s', img_path)
        return 1
    
    # Load image
    image_bgr = cv2.imread(str(img_path))
    if image_bgr is None:
        logger.error('Could not load image: %s', img_path)
        return 1
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    logger.info('Processing: %s (%dx%d)', img_path.name, image_rgb.shape[1], image_rgb.shape[0])
    
    # Determine output paths
    if args.output:
        out_dir = Path(args.output)
    else:
        out_dir = CONFIG.output.default_dir
        
    out_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = img_path.stem
    
    # =========================================================================
    # Bird's Eye View (BEV) Processing Mode
    # Pipeline: SegFormer → Road Mask → Separate Road → BEV → Lane Detection
    # =========================================================================
    if args.bev:
        bev_x_range = (args.bev_x_min, args.bev_x_max)
        bev_y_range = (args.bev_y_min, args.bev_y_max)
        
        logger.info('Pipeline: SegFormer → Road Mask → Separate Road → BEV → Lane Detection')
        logger.info('BEV range: X=[%s, %s]m, Y=[%s, %s]m', args.bev_x_min, args.bev_x_max, args.bev_y_min, args.bev_y_max)
        logger.info('Resolution: %s pixels/meter', args.pixels_per_meter)
        
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
        
        logger.info('Final BEV size: %dx%d pixels', bev_size[0], bev_size[1])
        
        # Save separated road image (perspective)
        road_only_bgr = cv2.cvtColor(road_only, cv2.COLOR_RGB2BGR)
        road_only_path = out_dir / f'{base_name}_road_only.png'
        cv2.imwrite(str(road_only_path), road_only_bgr)
        logger.info('Saved separated road: %s', road_only_path)
        
        # Save BEV road image (separated road in bird's eye view)
        bev_bgr = cv2.cvtColor(bev_road_image, cv2.COLOR_RGB2BGR)
        bev_path = out_dir / f'{base_name}_bev.png'
        cv2.imwrite(str(bev_path), bev_bgr)
        logger.info('Saved BEV road image: %s', bev_path)
        
        # Save original road mask (from SegFormer on perspective image)
        if args.save_mask:
            orig_mask_path = out_dir / f'{base_name}_road_mask.png'
            cv2.imwrite(str(orig_mask_path), original_road_mask)
            logger.info('Saved original road mask: %s', orig_mask_path)
            
            bev_mask_path = out_dir / f'{base_name}_bev_road_mask.png'
            cv2.imwrite(str(bev_mask_path), bev_road_mask)
            logger.info('Saved BEV road mask: %s', bev_mask_path)
        
        # Save BEV lane mask
        if args.save_lanes:
            bev_lane_path = out_dir / f'{base_name}_bev_lanes.png'
            cv2.imwrite(str(bev_lane_path), bev_lane_mask)
            logger.info('Saved BEV lane mask: %s', bev_lane_path)
        
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
            logger.info('Saved BEV lane overlay: %s', bev_lane_overlay_path)
        
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
            logger.info('Saved BEV road overlay: %s', bev_road_overlay_path)
        
        # Save BEV grayscale road texture (road pixels only in grayscale)
        if args.save_road_gray:
            # Convert BEV road image to grayscale
            bev_road_gray = cv2.cvtColor(bev_road_image, cv2.COLOR_RGB2GRAY)
            bev_road_gray_path = out_dir / f'{base_name}_bev_road_grayscale.png'
            cv2.imwrite(str(bev_road_gray_path), bev_road_gray)
            logger.info('Saved BEV grayscale road: %s', bev_road_gray_path)
        
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
        logger.info('Saved road mask: %s', mask_path)
    
    # Save colored road (optional)
    if args.colored:
        colored = extractor.visualize_road(image_rgb, mode='colored')
        colored_bgr = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
        colored_path = out_dir / f'{base_name}_road_colored.png'
        cv2.imwrite(str(colored_path), colored_bgr)
        logger.info('Saved colored road: %s', colored_path)
    # Save overlay (optional)
    if args.overlay:
        overlay = extractor.visualize_road(image_rgb, mode='overlay', alpha=args.alpha)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        overlay_path = out_dir / f'{base_name}_road_overlay.png'
        cv2.imwrite(str(overlay_path), overlay_bgr)
        logger.info('Saved overlay: %s', overlay_path)
    
    # Save lane markings and overlays
    if args.save_lanes or args.save_lane_overlay:
        lane_mask = extractor.extract_lanes(image_rgb)
        
        # Save Grayscale Threshold result
        if args.save_lanes:
            lane_path = out_dir / f'{base_name}_lanes.png'
            cv2.imwrite(str(lane_path), lane_mask)
            logger.info('Saved separated lanes (Grayscale Threshold): %s', lane_path)
            
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
            logger.info('Saved lane overlay: %s', lane_overlay_path)
    
    # Save Grayscale Road Texture (Requested)
    if args.save_road_gray:
        road_rgb = extractor.extract_road_only(image_rgb)
        road_gray = cv2.cvtColor(road_rgb, cv2.COLOR_RGB2GRAY)
        road_gray_path = out_dir / f'{base_name}_road_grayscale.png'
        cv2.imwrite(str(road_gray_path), road_gray)
        logger.info('Saved grayscale road: %s', road_gray_path)

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
        logger.error('Folder not found: %s', folder)
        return 1
    
    # Find all images
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    images = []
    for ext in extensions:
        images.extend(folder.glob(ext))
        images.extend(folder.glob(ext.upper()))
    
    images = sorted(set(images))
    
    if not images:
        logger.error('No images found in: %s', folder)
        return 1
    
    logger.info('Found %d images to process', len(images))
    
    # Output directory
    if args.output:
        out_dir = Path(args.output)
    else:
        out_dir = CONFIG.output.default_dir
        
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    for i, img_path in enumerate(images):
        logger.info('[%d/%d] Processing: %s', i+1, len(images), img_path.name)
        
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            logger.warning('Skipping: Could not load image: %s', img_path.name)
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
            colored = extractor.visualize_road(image_rgb, mode='colored')
            colored_bgr = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
            colored_path = out_dir / f'{base_name}_road_colored.png'
            cv2.imwrite(str(colored_path), colored_bgr)
        
        # Save overlay (optional)
        if args.overlay:
            overlay = extractor.visualize_road(image_rgb, mode='overlay', alpha=args.alpha)
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
                logger.debug('Saved separated lanes: %s', lane_path)

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
    
    logger.info('Done! Results saved to: %s', out_dir)
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
    parser.add_argument('--bev-x-min', type=float, default=CONFIG.bev.x_range[0], dest='bev_x_min',
                        help=f'BEV left boundary in meters (default: {CONFIG.bev.x_range[0]})')
    parser.add_argument('--bev-x-max', type=float, default=CONFIG.bev.x_range[1], dest='bev_x_max',
                        help=f'BEV right boundary in meters (default: {CONFIG.bev.x_range[1]})')
    parser.add_argument('--bev-y-min', type=float, default=CONFIG.bev.y_range[0], dest='bev_y_min',
                        help=f'BEV near boundary in meters (default: {CONFIG.bev.y_range[0]})')
    parser.add_argument('--bev-y-max', type=float, default=CONFIG.bev.y_range[1], dest='bev_y_max',
                        help=f'BEV far boundary in meters (default: {CONFIG.bev.y_range[1]})')
    parser.add_argument('--pixels-per-meter', type=float, default=CONFIG.bev.pixels_per_meter, dest='pixels_per_meter',
                        help=f'BEV resolution in pixels per meter (default: {CONFIG.bev.pixels_per_meter}). Higher = more detail.')
    
    # Model options
    parser.add_argument('--model', type=str, default=CONFIG.model, 
                        choices=['b0', 'b1', 'b2', 'b3', 'b4', 'b5'],
                        help=f'SegFormer model variant (default: {CONFIG.model})')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                        default=CONFIG.device,
                        help='Device to use (default: auto-detect)')
    
    # Display
    parser.add_argument('--show', action='store_true', help='Display results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose/debug logging')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress info messages, only show warnings/errors')
    args = parser.parse_args()
    
    # Configure logging based on verbosity flags
    if args.quiet:
        log_level = logging.WARNING
    elif args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Load defaults from dataclass config, optionally override from carla_sensors.yaml
    config_fov = CONFIG.camera.fov
    config_pitch = CONFIG.camera.pitch
    config_height = CONFIG.camera.height
    
    config_path = Path(args.config) if args.config else CONFIG.output.sensor_config
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
                    logger.info("Loaded camera config from %s: FOV=%s, Pitch=%s, Height=%s", config_path, config_fov, config_pitch, config_height)
        except Exception as e:
            logger.warning("Failed to load config from %s: %s", config_path, e)

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
        logger.info("No output format specified. Defaulting to --save-mask and --overlay.")
        args.save_mask = True
        args.overlay = True
    
    # Initialize extractor
    logger.info('Loading SegFormer-%s for road extraction...', args.model)
    extractor = RoadExtractor(model_name=args.model, device=args.device)
    
    # Process
    if args.image:
        return process_single_image(extractor, args)
    else:
        return process_folder(extractor, args)


if __name__ == '__main__':
    exit(main())

