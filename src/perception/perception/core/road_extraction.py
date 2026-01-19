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

Examples:
    # Single image - extract road mask
    python -m perception.core.road_extraction --image input.jpg --output ./output/
    
    # Single image with all outputs
    python -m perception.core.road_extraction --image input.jpg --colored --overlay --road-only
    
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
    
    # Extract road mask
    road_mask = extractor.extract_road_mask(image_rgb)
    
    # Determine output paths
    if args.output:
        out_dir = Path(args.output)
    else:
        out_dir = DEFAULT_OUTPUT_DIR
        
    out_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = img_path.stem
    
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