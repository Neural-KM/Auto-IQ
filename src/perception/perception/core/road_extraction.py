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
    extract_lanes_ipm()         - Lane detection with Inverse Perspective Mapping (default)

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

Inverse Perspective Mapping (IPM) Mode (DEFAULT):
    IPM mode is enabled by default for better lane detection accuracy.
    --no-ipm              To run it without taking IPM
    --fov                 Camera FOV in degrees (default: 110)
    --pitch               Camera pitch in degrees (default: 0)
    --camera-height       Camera height in meters (default: 1.2)
    --pixels-per-meter    IPM resolution (default: 30)
    --ipm-x-min/max       IPM lateral range in meters
    --ipm-y-min/max       IPM forward range in meters

Logging:
    --verbose, -v         Enable debug logging
    --quiet, -q           Suppress info messages, only show warnings/errors

Examples:
    # Single image - IPM mode (default)
    python -m perception.core.road_extraction --image input.jpg --output ./output/
    
    # Single image with all outputs
    python -m perception.core.road_extraction --image input.jpg --all
    
    # IPM with custom camera parameters
    python -m perception.core.road_extraction --image input.jpg --fov 90 --pitch 5 --camera-height 1.5 --all
    
    # To run it without taking IPM
    python -m perception.core.road_extraction --image input.jpg --no-ipm --save-lanes --save-lane-overlay
    
    # Batch process folder
    python -m perception.core.road_extraction --folder /data/images/ --output ./road_masks/
    
    # Use larger model for better accuracy
    python -m perception.core.road_extraction --image input.jpg --model b5
    
    # Quiet mode (only warnings/errors)
    python -m perception.core.road_extraction --folder /data/images/ --all -q
"""

import numpy as np
import cv2
import argparse
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from perception.core.segformer import SegFormer, CITYSCAPES_PALETTE
from perception.utils.lane_detection import _extract_lanes_from_ipm
from perception.utils.ipm_transform import convert_to_ipm 

# ============================================================================
# Configuration (Dataclasses - frozen for immutability)
# ============================================================================

@dataclass(frozen=True)
class IPMConfig:
    """Inverse Perspective Mapping transform parameters."""
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
    ipm: IPMConfig = field(default_factory=IPMConfig)
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
    # Inverse Perspective Mapping (IPM) Methods
    # =========================================================================
    
    def extract_lanes_ipm(
        self, 
        image: np.ndarray,
        fov: float = None,
        pitch: float = None,
        camera_height: float = None,
        ipm_x_range: Tuple[float, float] = None,
        ipm_y_range: Tuple[float, float] = None,
        pixels_per_meter: float = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
        """
        Extract lane markings using the correct pipeline:
        SegFormer → Road Mask → Separate Road → IPM → Lane Detection
        
        Pipeline:
        1. Run SegFormer on original image → get road mask
        2. Separate road from image (road_only = image AND mask)
        3. Convert separated road image to IPM
        4. Find lane markings in IPM image
        
        Args:
            image: RGB image (H, W, 3)
            fov: Camera horizontal field of view in degrees. Uses CONFIG.camera.fov if None.
            pitch: Camera pitch angle in degrees. Uses CONFIG.camera.pitch if None.
            camera_height: Camera height above ground in meters. Uses CONFIG.camera.height if None.
            ipm_x_range: (x_min, x_max) range in meters (left/right). Uses CONFIG.ipm.x_range if None.
            ipm_y_range: (y_min, y_max) range in meters (forward distance). Uses CONFIG.ipm.y_range if None.
            pixels_per_meter: Resolution - pixels per meter. Uses CONFIG.ipm.pixels_per_meter if None.
            
        Returns:
            road_only: Separated road image (original perspective)
            ipm_road_image: Inverse Perspective Mapping of separated road
            ipm_lane_mask: Lane markings mask in IPM space
            original_road_mask: Road mask from SegFormer (perspective)
            ipm_road_mask: Road mask in IPM space
            ipm_size: (width, height) of the IPM output
        """
        # Use CONFIG defaults if not provided
        fov = fov if fov is not None else CONFIG.camera.fov
        pitch = pitch if pitch is not None else CONFIG.camera.pitch
        camera_height = camera_height if camera_height is not None else CONFIG.camera.height
        
        height, width = image.shape[:2]
        
        # 1. Run SegFormer on ORIGINAL image → get road mask
        print("Step 1: Running SegFormer to get road mask...")
        original_road_mask = self.extract_road_mask(image)
        
        # 2. Separate road from image (keep only road pixels)
        print("Step 2: Separating road from image...")
        road_only = cv2.bitwise_and(image, image, mask=original_road_mask)
        
        # 3. Convert separated road image to IPM (Using Camera Geometry)
        # Dimensions are auto-calculated from ranges and pixels_per_meter
        print("Step 3: Converting separated road to IPM...")
        ipm_road_image, ipm_size, M = convert_to_ipm(
            road_only,
            fov=fov,
            pitch=pitch,
            camera_height=camera_height,
            bev_x_range=ipm_x_range,
            bev_y_range=ipm_y_range,
            pixels_per_meter=pixels_per_meter,
            return_transform=True
        )
        
        print(f"IPM size: {ipm_size[0]}x{ipm_size[1]} pixels (auto-calculated for equal scale)")

        # Also transform road mask to IPM for reference
        ipm_road_mask = cv2.warpPerspective(
            original_road_mask, M, ipm_size, 
            flags=cv2.INTER_NEAREST
        )
        
        # 5. Find lane markings in IPM image
        print("Step 4: Detecting lane markings in IPM...")
        ipm_lane_mask = _extract_lanes_from_ipm(ipm_road_image, ipm_road_mask)
        
        return road_only, ipm_road_image, ipm_lane_mask, original_road_mask, ipm_road_mask, ipm_size
    
    
    
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
        out_dir = CONFIG.output.default_dir
        
    out_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = img_path.stem
    
    # =========================================================================
    # Inverse Perspective Mapping (IPM) Processing Mode
    # Pipeline: SegFormer → Road Mask → Separate Road → IPM → Lane Detection
    # =========================================================================
    if args.ipm:
        ipm_x_range = (args.ipm_x_min, args.ipm_x_max)
        ipm_y_range = (args.ipm_y_min, args.ipm_y_max)
        
        print('Pipeline: SegFormer → Road Mask → Separate Road → IPM → Lane Detection')
        print(f'IPM range: X=[{args.ipm_x_min}, {args.ipm_x_max}]m, Y=[{args.ipm_y_min}, {args.ipm_y_max}]m')
        print(f'Resolution: {args.pixels_per_meter} pixels/meter')
        
        # Run the pipeline
        road_only, ipm_road_image, ipm_lane_mask, original_road_mask, ipm_road_mask, ipm_size = extractor.extract_lanes_ipm(
            image_rgb,
            fov=args.fov,
            pitch=args.pitch,
            camera_height=args.camera_height,
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
        
        # Save original road mask (from SegFormer on perspective image)
        if args.save_mask:
            orig_mask_path = out_dir / f'{base_name}_road_mask.png'
            cv2.imwrite(str(orig_mask_path), original_road_mask)
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
    # Extract road mask
    road_mask = extractor.extract_road_mask(image_rgb)
    
    # Save binary mask
    if args.save_mask:
        mask_path = out_dir / f'{base_name}_road_mask.png'
        cv2.imwrite(str(mask_path), road_mask)
        print(f'Saved road mask: {mask_path}')

    # Save colored road (optional)
    if args.colored:
        colored = extractor.visualize_road(image_rgb, mode='colored')
        colored_bgr = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
        colored_path = out_dir / f'{base_name}_road_colored.png'
        cv2.imwrite(str(colored_path), colored_bgr)
        print(f'Saved colored road: {colored_path}')
    # Save overlay (optional)
    if args.overlay:
        overlay = extractor.visualize_road(image_rgb, mode='overlay', alpha=args.alpha)
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
        print(f'Error: No images found in: {folder}')
        return 1
    
    print(f'Found {len(images)} images to process')
    
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
            road_rgb = extractor.extract_road_only(image_rgb)
            road_gray = cv2.cvtColor(road_rgb, cv2.COLOR_RGB2GRAY)
            road_gray_path = out_dir / f'{base_name}_road_grayscale.png'
            cv2.imwrite(str(road_gray_path), road_gray)
    
    print(f'Done! Results saved to: {out_dir}')
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
    # Inverse Perspective Mapping (IPM) Options
    # =========================================================================
    parser.add_argument('--ipm', action='store_true', default=True, dest='ipm',
                        help='Enable Inverse Perspective Mapping mode (default: enabled)')
    parser.add_argument('--no-ipm', action='store_false', dest='ipm',
                        help='To run it without taking IPM')
    parser.add_argument('--fov', type=float, default=None,
                        help='Camera horizontal field of view in degrees (default: 110 or from config)')
    parser.add_argument('--pitch', type=float, default=None,
                        help='Camera pitch angle in degrees (default: 0 or from config)')
    parser.add_argument('--camera-height', type=float, default=None, dest='camera_height',
                        help='Camera height above ground in meters (default: 1.2 or from config)')
    parser.add_argument('--ipm-x-min', type=float, default=CONFIG.ipm.x_range[0], dest='ipm_x_min',
                        help=f'IPM left boundary in meters (default: {CONFIG.ipm.x_range[0]})')
    parser.add_argument('--ipm-x-max', type=float, default=CONFIG.ipm.x_range[1], dest='ipm_x_max',
                        help=f'IPM right boundary in meters (default: {CONFIG.ipm.x_range[1]})')
    parser.add_argument('--ipm-y-min', type=float, default=CONFIG.ipm.y_range[0], dest='ipm_y_min',
                        help=f'IPM near boundary in meters (default: {CONFIG.ipm.y_range[0]})')
    parser.add_argument('--ipm-y-max', type=float, default=CONFIG.ipm.y_range[1], dest='ipm_y_max',
                        help=f'IPM far boundary in meters (default: {CONFIG.ipm.y_range[1]})')
    parser.add_argument('--pixels-per-meter', type=float, default=CONFIG.ipm.pixels_per_meter, dest='pixels_per_meter',
                        help=f'IPM resolution in pixels per meter (default: {CONFIG.ipm.pixels_per_meter}). Higher = more detail.')
    
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
        print("No output format specified. Defaulting to --save-mask and --overlay.")
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

