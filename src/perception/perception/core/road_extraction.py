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
from pathlib import Path
from typing import Optional, Tuple

from perception.core.segformer import SegFormer, CITYSCAPES_CLASSES, CITYSCAPES_PALETTE


# ============================================================================
# Bird's Eye View (BEV) Transform Functions
# ============================================================================

def compute_bev_transform(
    width: int, 
    height: int, 
    fov: float = 110.0, 
    pitch: float = 0.0, 
    camera_height: float = 1.2,
    bev_width: int = 800,
    bev_height: int = 800,
    bev_x_range: Tuple[float, float] = (-5.0, 5.0),
    bev_y_range: Tuple[float, float] = (3.0, 20.0)
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Compute perspective transform matrix for Bird's Eye View.
    
    Args:
        width: Input image width
        height: Input image height
        fov: Camera horizontal field of view in degrees
        pitch: Camera pitch angle in degrees (positive = looking down)
        camera_height: Camera height above ground in meters
        bev_width: Output BEV image width
        bev_height: Output BEV image height
        bev_x_range: (x_min, x_max) range in meters (left/right)
        bev_y_range: (y_min, y_max) range in meters (forward distance)
        
    Returns:
        M: 3x3 perspective transform matrix
        bev_size: (width, height) of output BEV image
    """
    # Compute camera intrinsics
    fov_rad = np.radians(fov)
    fx = width / (2.0 * np.tan(fov_rad / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0

    pitch_rad = np.radians(pitch)
    
    x_min, x_max = bev_x_range
    y_min, y_max = bev_y_range
    
    # Ground plane points: (X, Y, Z) where Z=0 (ground level)
    ground_points_3d = np.array([
        [x_min, y_max, 0],  # Far left
        [x_max, y_max, 0],  # Far right
        [x_max, y_min, 0],  # Near right
        [x_min, y_min, 0],  # Near left
    ], dtype=np.float64)
    
    # Rotation: pitch around x-axis
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)]
    ], dtype=np.float64)
    
    src_pts = []
    for gp in ground_points_3d:
        # Convert vehicle coords (X_v=right, Y_v=forward, Z_v=up) to 
        # camera coords (X_c=right, Y_c=down, Z_c=forward)
        
        # Point relative to camera position (camera at height h)
        p_vehicle = np.array([gp[0], gp[1], gp[2] - camera_height])
        
        # Transform to camera frame
        # X_c = X_v, Y_c = -Z_v, Z_c = Y_v
        p_cam_unrotated = np.array([p_vehicle[0], -p_vehicle[2], p_vehicle[1]])
        
        p_cam = R_pitch @ p_cam_unrotated
        
        # Project to image using pinhole model
        if p_cam[2] > 0:  # Only if in front of camera
            u = fx * p_cam[0] / p_cam[2] + cx
            v = fy * p_cam[1] / p_cam[2] + cy
            src_pts.append([u, v])
        else:
            src_pts.append([cx, 0])
    
    src_pts = np.float32(src_pts)
    
    # Destination points in BEV image
    dst_pts = np.float32([
        [0, 0],                           # Far left -> top-left
        [bev_width, 0],                   # Far right -> top-right
        [bev_width, bev_height],          # Near right -> bottom-right
        [0, bev_height],                  # Near left -> bottom-left
    ])
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return M, (bev_width, bev_height)


def convert_to_bev(
    image: np.ndarray,
    fov: float = 110.0,
    pitch: float = 0.0,
    camera_height: float = 1.2,
    bev_width: int = 800,
    bev_height: int = 800,
    bev_x_range: Tuple[float, float] = (-5.0, 5.0),
    bev_y_range: Tuple[float, float] = (3.0, 20.0)
) -> np.ndarray:
    """
    Convert an image to Bird's Eye View.
    
    Args:
        image: Input image (H, W, 3) or (H, W)
        fov: Camera horizontal field of view in degrees
        pitch: Camera pitch angle in degrees
        camera_height: Camera height above ground in meters
        bev_width: Output BEV image width
        bev_height: Output BEV image height
        bev_x_range: (x_min, x_max) range in meters
        bev_y_range: (y_min, y_max) range in meters
        
    Returns:
        BEV transformed image
    """
    height, width = image.shape[:2]
    
    M, bev_size = compute_bev_transform(
        width, height, fov, pitch, camera_height,
        bev_width, bev_height, bev_x_range, bev_y_range
    )
    
    bev_img = cv2.warpPerspective(image, M, bev_size, flags=cv2.INTER_LINEAR)
    
    return bev_img


# Road class index in Cityscapes
ROAD_CLASS_ID = 0

# Default output directory
DEFAULT_OUTPUT_DIR = Path('/home/autoiq/nakul/Auto-IQ/data/rosbag_images_15jan/inference')


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
        # To make adaptive thresholding work correctly near mask boundaries, 
        # we fill the non-road area with the mean road intensity.
        mean_road_val = cv2.mean(gray, mask=road_mask)[0]
        filled_gray = gray.copy()
        filled_gray[road_mask == 0] = int(mean_road_val)
    
    # =========================================================================
    # Bird's Eye View (BEV) Methods
    # =========================================================================
    
    def extract_lanes_bev(
        self, 
        image: np.ndarray,
        fov: float = 110.0,
        pitch: float = 0.0,
        camera_height: float = 1.2,
        bev_width: int = 800,
        bev_height: int = 800
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract lane markings using the correct pipeline:
        Image → SegFormer → Road Mask → BEV → Lane Detection
        
        This method:
        1. Runs road segmentation on ORIGINAL image (SegFormer trained on perspective)
        2. Converts both image AND road mask to BEV
        3. Extracts lane markings from BEV road region
        
        Args:
            image: RGB image (H, W, 3)
            fov: Camera horizontal field of view in degrees
            pitch: Camera pitch angle in degrees
            camera_height: Camera height above ground in meters
            bev_width: Output BEV image width
            bev_height: Output BEV image height
            
        Returns:
            bev_image: Bird's Eye View transformed image
            bev_road_mask: Road mask in BEV space
            bev_lane_mask: Lane markings mask in BEV space
            original_road_mask: Road mask from original image (for reference)
        """
        # 1. Run SegFormer on ORIGINAL image (it's trained on perspective images)
        original_road_mask = self.extract_road_mask(image)
        
        # 2. Convert original image to BEV
        bev_image = convert_to_bev(
            image, fov, pitch, camera_height,
            bev_width, bev_height
        )
        
        # 3. Convert road mask to BEV (use NEAREST interpolation for binary mask)
        height, width = image.shape[:2]
        M, bev_size = compute_bev_transform(
            width, height, fov, pitch, camera_height,
            bev_width, bev_height
        )
        bev_road_mask = cv2.warpPerspective(
            original_road_mask, M, bev_size, 
            flags=cv2.INTER_NEAREST  # Nearest neighbor for binary mask
        )
        
        # 4. Extract lanes from BEV image using adaptive thresholding
        bev_lane_mask = self._extract_lanes_from_bev(bev_image, bev_road_mask)
        
        return bev_image, bev_road_mask, bev_lane_mask, original_road_mask
    
    def _extract_lanes_from_bev(
        self, 
        bev_image: np.ndarray, 
        bev_road_mask: np.ndarray
    ) -> np.ndarray:
        """
        Extract lane markings from BEV image using adaptive thresholding.
        
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
        
        # Fill non-road areas with mean road intensity
        mean_val = cv2.mean(gray, mask=bev_road_mask)[0]
        filled_gray = gray.copy()
        filled_gray[bev_road_mask == 0] = int(mean_val)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(filled_gray, (5, 5), 0)
        
        # Method 1: Adaptive thresholding for lane detection
        # Use smaller block size for better local contrast detection
        lane_mask_adaptive = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,  # Block size
            -15  # C value: pixels brighter than local mean - C
        )
        
        # Method 2: Simple threshold based on brightness (white lanes)
        # Calculate road mean and std
        road_pixels = gray[bev_road_mask == 255]
        if len(road_pixels) > 0:
            road_mean = np.mean(road_pixels)
            road_std = np.std(road_pixels)
            # Lanes are typically brighter than road surface
            threshold = road_mean + 1.5 * road_std
            _, lane_mask_simple = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        else:
            lane_mask_simple = np.zeros_like(gray)
        
        # Combine both methods (OR)
        lane_mask = cv2.bitwise_or(lane_mask_adaptive, lane_mask_simple)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel)
        
        # Remove small noise blobs
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel_close)
        
        # Enforce road mask
        lane_mask = cv2.bitwise_and(lane_mask, bev_road_mask)
        
        return lane_mask
    
    def extract_road_bev(
        self, 
        image: np.ndarray,
        fov: float = 110.0,
        pitch: float = 0.0,
        camera_height: float = 1.2,
        bev_width: int = 800,
        bev_height: int = 800
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract road from Bird's Eye View transformed image.
        
        Args:
            image: RGB image (H, W, 3)
            fov: Camera horizontal field of view in degrees
            pitch: Camera pitch angle in degrees
            camera_height: Camera height above ground in meters
            bev_width: Output BEV image width
            bev_height: Output BEV image height
            
        Returns:
            bev_image: Bird's Eye View transformed image
            bev_road_mask: Road mask in BEV space
        """
        # Convert to Bird's Eye View
        bev_image = convert_to_bev(
            image, fov, pitch, camera_height,
            bev_width, bev_height
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
    # Pipeline: Image → SegFormer → Road Mask → BEV → Lane Detection
    # =========================================================================
    if args.bev:
        print(f'Pipeline: Image → SegFormer → Road Mask → BEV → Lane Detection')
        print(f'BEV params: FOV={args.fov}, pitch={args.pitch}, height={args.camera_height}m')
        
        # Run the corrected pipeline
        bev_image, bev_road_mask, bev_lane_mask, original_road_mask = extractor.extract_lanes_bev(
            image_rgb,
            fov=args.fov,
            pitch=args.pitch,
            camera_height=args.camera_height,
            bev_width=args.bev_width,
            bev_height=args.bev_height
        )
        
        # Save BEV image
        bev_bgr = cv2.cvtColor(bev_image, cv2.COLOR_RGB2BGR)
        bev_path = out_dir / f'{base_name}_bev.png'
        cv2.imwrite(str(bev_path), bev_bgr)
        print(f'Saved BEV image: {bev_path}')
        
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
            bev_lane_overlay = bev_image.copy()
            bev_lane_overlay[bev_lane_mask == 255] = (255, 0, 0)  # Red
            
            # Blend
            bev_lane_overlay_blended = cv2.addWeighted(
                bev_image, 1 - args.alpha, 
                bev_lane_overlay, args.alpha, 0
            )
            
            bev_lane_overlay_bgr = cv2.cvtColor(bev_lane_overlay_blended, cv2.COLOR_RGB2BGR)
            bev_lane_overlay_path = out_dir / f'{base_name}_bev_lanes_overlay.png'
            cv2.imwrite(str(bev_lane_overlay_path), bev_lane_overlay_bgr)
            print(f'Saved BEV lane overlay: {bev_lane_overlay_path}')
        
        # Save BEV road overlay
        if args.overlay:
            bev_road_overlay = bev_image.copy()
            bev_road_overlay[bev_road_mask == 255] = (128, 64, 128)  # Purple
            
            bev_road_overlay_blended = cv2.addWeighted(
                bev_image, 1 - args.alpha, 
                bev_road_overlay, args.alpha, 0
            )
            
            bev_road_overlay_bgr = cv2.cvtColor(bev_road_overlay_blended, cv2.COLOR_RGB2BGR)
            bev_road_overlay_path = out_dir / f'{base_name}_bev_road_overlay.png'
            cv2.imwrite(str(bev_road_overlay_path), bev_road_overlay_bgr)
            print(f'Saved BEV road overlay: {bev_road_overlay_path}')
        
        # Save BEV grayscale road texture (road pixels only in grayscale)
        if args.save_road_gray:
            # Extract road-only pixels from BEV
            bev_road_only = cv2.bitwise_and(bev_image, bev_image, mask=bev_road_mask)
            bev_road_gray = cv2.cvtColor(bev_road_only, cv2.COLOR_RGB2GRAY)
            bev_road_gray_path = out_dir / f'{base_name}_bev_road_grayscale.png'
            cv2.imwrite(str(bev_road_gray_path), bev_road_gray)
            print(f'Saved BEV grayscale road: {bev_road_gray_path}')
        
        # Display BEV results if requested
        if args.show:
            # Create side-by-side comparison
            # Resize original to match BEV height for comparison
            scale = args.bev_height / image_bgr.shape[0]
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
    parser.add_argument('--fov', type=float, default=110.0,
                        help='Camera horizontal field of view in degrees (default: 110)')
    parser.add_argument('--pitch', type=float, default=0.0,
                        help='Camera pitch angle in degrees (default: 0)')
    parser.add_argument('--camera-height', type=float, default=1.2, dest='camera_height',
                        help='Camera height above ground in meters (default: 1.2)')
    parser.add_argument('--bev-width', type=int, default=800, dest='bev_width',
                        help='BEV output width in pixels (default: 800)')
    parser.add_argument('--bev-height', type=int, default=800, dest='bev_height',
                        help='BEV output height in pixels (default: 800)')
    
    # Model options
    parser.add_argument('--model', type=str, default='b0', 
                        choices=['b0', 'b1', 'b2', 'b3', 'b4', 'b5'],
                        help='SegFormer model variant (default: b0)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                        help='Device to use (default: auto-detect)')
    
    # Display
    parser.add_argument('--show', action='store_true', help='Display results')
    args = parser.parse_args()
    
    # Handle --all flag
    if args.all:
        args.save_mask = True
        args.save_lanes = True
        args.save_lane_overlay = True
        args.save_road_gray = True
        args.colored = True
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