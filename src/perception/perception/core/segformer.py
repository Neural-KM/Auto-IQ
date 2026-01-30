#!/usr/bin/env python3
"""
SegFormer Semantic Segmentation Core
=====================================

Core inference class for SegFormer semantic segmentation using HuggingFace's
pretrained models on the Cityscapes dataset.

This module is ROS-agnostic and can be used standalone or integrated into ROS nodes.

Cityscapes Classes (19 classes):
    0: road, 1: sidewalk, 2: building, 3: wall, 4: fence, 5: pole,
    6: traffic light, 7: traffic sign, 8: vegetation, 9: terrain,
    10: sky, 11: person, 12: rider, 13: car, 14: truck, 15: bus,
    16: train, 17: motorcycle, 18: bicycle

Output file:
/Auto-IQ/data/rosbag_images_15jan/inference

Input data:
/Auto-IQ/data/rosbag_images_15jan/Images_15jan_bot_transit


Usage:
    Single Image:
        python -m perception.core.segformer --image path/to/image.jpg --output path/to/output.png
    
    Batch Processing (Folder):
        python -m perception.core.segformer --folder /path/to/images/
        
    Examples:
        # Single image with overlay
        python -m perception.core.segformer --image input.jpg --overlay --alpha 0.6
        
        # Batch process entire folder with B2 model
        python -m perception.core.segformer --folder /data/images/ --model b2
        
        # Batch with stats and GPU
        python -m perception.core.segformer --folder /data/images/ --stats --device cuda
"""

import numpy as np
import cv2
import torch
import time
from pathlib import Path
from typing import Optional, Tuple, Union

from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from perception.data_classes.semantic_labels import get_class_name


# Cityscapes color palette (19 classes)
CITYSCAPES_PALETTE = np.array([
    [128, 64, 128],    # 0: road (purple)
    [244, 35, 232],    # 1: sidewalk (pink)
    [70, 70, 70],      # 2: building (dark gray)
    [102, 102, 156],   # 3: wall (blue-gray)
    [190, 153, 153],   # 4: fence (beige)
    [153, 153, 153],   # 5: pole (gray)
    [250, 170, 30],    # 6: traffic light (orange)
    [220, 220, 0],     # 7: traffic sign (yellow)
    [107, 142, 35],    # 8: vegetation (olive)
    [152, 251, 152],   # 9: terrain (light green)
    [70, 130, 180],    # 10: sky (steel blue)
    [220, 20, 60],     # 11: person (crimson)
    [255, 0, 0],       # 12: rider (red)
    [0, 0, 142],       # 13: car (dark blue)
    [0, 0, 70],        # 14: truck (navy)
    [0, 60, 100],      # 15: bus (dark cyan)
    [0, 80, 100],      # 16: train (teal)
    [0, 0, 230],       # 17: motorcycle (blue)
    [119, 11, 32],     # 18: bicycle (maroon)
], dtype=np.uint8)

CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck', 'bus',
    'train', 'motorcycle', 'bicycle'
]

# Available model variants (sorted by size/accuracy)
SEGFORMER_MODELS = {
    'b0': 'nvidia/segformer-b0-finetuned-cityscapes-1024-1024',
    'b1': 'nvidia/segformer-b1-finetuned-cityscapes-1024-1024',
    'b2': 'nvidia/segformer-b2-finetuned-cityscapes-1024-1024',
    'b3': 'nvidia/segformer-b3-finetuned-cityscapes-1024-1024',
    'b4': 'nvidia/segformer-b4-finetuned-cityscapes-1024-1024',
    'b5': 'nvidia/segformer-b5-finetuned-cityscapes-1024-1024',
}

class SegFormer:
    """
    SegFormer semantic segmentation inference class.
    
    Supports multiple model variants (b0-b5) with increasing accuracy/compute trade-off.
    
    Attributes:
        model: The SegFormer model
        processor: The image processor/tokenizer
        device: torch device (cuda/cpu)
        num_classes: Number of output classes (19 for Cityscapes)
    """
    
    def __init__(
        self, 
        model_name: str = 'nvidia/segformer-b0-finetuned-cityscapes-1024-1024',
        device: Optional[str] = None
    ):
        """
        Initialize SegFormer model.
        
        Args:
            model_name: HuggingFace model name or shorthand (b0, b1, ..., b5)
            device: 'cuda', 'cpu', or None for auto-detection
        """
        # Handle shorthand model names
        if model_name in SEGFORMER_MODELS:
            model_name = SEGFORMER_MODELS[model_name]
        
        self.model_name = model_name
        
        # Device setup
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if device == 'cuda' and not torch.cuda.is_available():
            print('CUDA not available, falling back to CPU')
            device = 'cpu'
            
        self.device = torch.device(device)
        
        # Load model
        print(f'Loading SegFormer model: {model_name}')
        print(f'Device: {self.device}')
        
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self.num_classes = self.model.config.num_labels
        print(f'Model loaded successfully! ({self.num_classes} classes)')
        
        # Warmup inference
        self._warmup()
    
    def _warmup(self):
        """Run a warmup inference to initialize CUDA kernels."""
        dummy = np.zeros((256, 256, 3), dtype=np.uint8)
        _ = self.predict(dummy)
        print('Warmup complete')
    
    def predict(
        self, 
        image: np.ndarray,
        return_logits: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Run semantic segmentation on an image.
        
        Args:
            image: RGB image as numpy array (H, W, 3), uint8
            return_logits: If True, also return raw logits
            
        Returns:
            segmentation: Class indices (H, W) as numpy array
            logits (optional): Raw logits (num_classes, H, W) if return_logits=True
        """
        original_size = image.shape[:2]  # (H, W)
        
        # Preprocess
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # (1, num_classes, H/4, W/4)
        
        # Upsample to original resolution
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=original_size,
            mode='bilinear',
            align_corners=False
        )
        
        # Get class predictions
        segmentation = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()
        
        if return_logits:
            logits_np = upsampled_logits.squeeze().cpu().numpy()
            return segmentation, logits_np
        
        return segmentation
    
    def predict_with_confidence(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run segmentation and return confidence scores.
        
        Args:
            image: RGB image (H, W, 3)
            
        Returns:
            segmentation: Class indices (H, W)
            confidence: Confidence scores (H, W) in [0, 1]
        """
        segmentation, logits = self.predict(image, return_logits=True)
        
        # Softmax to get probabilities
        probs = np.exp(logits) / np.exp(logits).sum(axis=0, keepdims=True)
        confidence = probs.max(axis=0)
        
        return segmentation, confidence
    
    @staticmethod
    def colorize(mask: np.ndarray) -> np.ndarray:
        """
        Convert class indices to RGB colors using Cityscapes palette.
        
        Args:
            mask: Segmentation mask (H, W) with class indices 0-18
            
        Returns:
            RGB image (H, W, 3)
        """
        mask = np.clip(mask, 0, len(CITYSCAPES_PALETTE) - 1)
        return CITYSCAPES_PALETTE[mask]
    
    @staticmethod
    def overlay(
        image: np.ndarray, 
        mask: np.ndarray, 
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Overlay colored segmentation on original image.
        
        Args:
            image: Original RGB image (H, W, 3)
            mask: Segmentation mask (H, W) with class indices
            alpha: Blend factor (0 = only image, 1 = only segmentation)
            
        Returns:
            Blended RGB image (H, W, 3)
        """
        colored = SegFormer.colorize(mask)
        blended = cv2.addWeighted(image, 1 - alpha, colored, alpha, 0)
        return blended
    
    def get_class_stats(self, mask: np.ndarray) -> dict:
        """
        Get pixel statistics for each class in the segmentation.
        
        Args:
            mask: Segmentation mask (H, W)
            
        Returns:
            Dict mapping class names to pixel counts and percentages
        """
        total_pixels = mask.size
        unique, counts = np.unique(mask, return_counts=True)
        
        stats = {}
        for class_id, count in zip(unique, counts):
            class_name = get_class_name(class_id) or 'unknown'
            stats[class_name] = {
                'class_id': int(class_id),
                'pixel_count': int(count),
                'percentage': float(count / total_pixels * 100)
            }
        
        return stats


def main():
    """Semantic segmentation inference - supports single image or batch folder processing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='SegFormer Semantic Segmentation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', '-i', type=str,
                             help='Single image file path')
    input_group.add_argument('--folder', '-f', type=str,
                             help='Folder containing multiple images')
    
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file path for single image (default: input_name_segmented.png)')
    parser.add_argument('--model', '-m', type=str, default='b0',
                        choices=['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'all'],
                        help='Model variant (b0=fastest, b5=most accurate, all=run all models, default: b0)')
    parser.add_argument('--device', '-d', type=str, default=None,
                        choices=['cuda', 'cpu'],
                        help='Device for inference (default: auto-detect)')
    parser.add_argument('--overlay', action='store_true',
                        help='Save overlay (original + segmentation) instead of just mask')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Overlay blend factor: 0=original, 1=mask (default: 0.5)')
    parser.add_argument('--stats', action='store_true',
                        help='Print class statistics')
    parser.add_argument('--show', action='store_true',
                        help='Display result in window (ESC to close)')
    
    args = parser.parse_args()
    
    # Handle "all models" mode
    if args.model == 'all':
        models_to_run = ['b0', 'b1', 'b2', 'b3', 'b4', 'b5']
        print(f'Running inference on all {len(models_to_run)} models: {", ".join(models_to_run)}\n')
        
        if args.image:
            for model_variant in models_to_run:
                print(f'\n{"="*70}')
                print(f'Processing with SegFormer-{model_variant}')
                print(f'{"="*70}')
                args.model = model_variant
                segformer = SegFormer(model_name=model_variant, device=args.device)
                process_single_image(segformer, args)
        else:  # args.folder
            for model_variant in models_to_run:
                print(f'\n{"="*70}')
                print(f'Processing with SegFormer-{model_variant}')
                print(f'{"="*70}')
                args.model = model_variant
                segformer = SegFormer(model_name=model_variant, device=args.device)
                process_batch_folder(segformer, args)
    else:
        # Single model mode
        print(f'Loading SegFormer-{args.model}...')
        segformer = SegFormer(model_name=args.model, device=args.device)
        print()
        
        # Mode 1: Single Image Processing
        if args.image:
            process_single_image(segformer, args)
        
        # Mode 2: Batch Folder Processing
        else:  # args.folder
            process_batch_folder(segformer, args)


def process_single_image(segformer, args):
    """Process a single image file."""
    img_path = Path(args.image)
    
    if not img_path.exists():
        print(f'ERROR: Image file not found: {img_path}')
        return 1
    
    if not img_path.is_file():
        print(f'ERROR: Not a file: {img_path}')
        return 1
    
    print(f'Processing: {img_path.name}')
    
    # Load image
    image_bgr = cv2.imread(str(img_path))
    if image_bgr is None:
        print(f'ERROR: Could not load image: {img_path}')
        return 1
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    
    # Run inference
    start = time.perf_counter()
    mask = segformer.predict(image_rgb)
    elapsed = (time.perf_counter() - start) * 1000
    
    print(f'Inference time: {elapsed:.1f}ms | Image size: {w}x{h}')
    
    # Print stats if requested
    if args.stats:
        stats = segformer.get_class_stats(mask)
        print('Class distribution:')
        for name, info in sorted(stats.items(), key=lambda x: -x[1]['percentage']):
            print(f'  {name}: {info["percentage"]:.1f}%')
    
    # Generate output
    if args.overlay:
        result = segformer.overlay(image_rgb, mask, alpha=args.alpha)
        default_suffix = '_overlay'
    else:
        result = segformer.colorize(mask)
        default_suffix = '_segmented'
    
    # Determine output path with model name
    if args.output:
        out_path = Path(args.output)
        # If output is a directory, create filename inside it
        if out_path.is_dir() or not out_path.suffix:
            out_path.mkdir(parents=True, exist_ok=True)
            model_name = args.model
            out_file = out_path / f'{img_path.stem}_inference_{model_name}{default_suffix}.png'
        else:
            out_file = out_path
    else:
        # Create inference folder and add model name to output
        inference_folder = img_path.parent / 'inference'
        model_name = args.model
        out_file = inference_folder / f'{img_path.stem}_inference_{model_name}{default_suffix}.png'
    
    # Save result
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_file), result_bgr)
    print(f'Saved: {out_file}\n')
    
    # Display if requested
    if args.show:
        combined = np.zeros((h, w * 2, 3), dtype=np.uint8)
        combined[:, :w] = image_bgr
        combined[:, w:] = result_bgr
        
        cv2.imshow('SegFormer: Original (left) | Segmentation (right) | Press any key to close', combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return 0


def process_batch_folder(segformer, args):
    """Process all images in a folder."""
    input_folder = Path(args.folder).resolve()
    
    if not input_folder.exists():
        print(f'ERROR: Folder does not exist: {input_folder}')
        return 1
    
    if not input_folder.is_dir():
        print(f'ERROR: Not a directory: {input_folder}')
        return 1
    
    # Create inference output subfolder
    inference_folder = input_folder / 'inference'
    inference_folder.mkdir(exist_ok=True)
    
    # Collect all image files
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG')
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_folder.glob(ext))
    
    # Filter out files already in inference folder
    image_files = [f for f in image_files if f.parent != inference_folder]
    image_files = sorted(image_files)
    
    if not image_files:
        print(f'ERROR: No images found in {input_folder}')
        print(f'Supported formats: jpg, jpeg, png, bmp')
        return 1
    
    print('=' * 70)
    print(f'Batch Segmentation')
    print('=' * 70)
    print(f'Input folder:  {input_folder}')
    print(f'Output folder: {inference_folder}')
    print(f'Images found:  {len(image_files)}')
    print('=' * 70 + '\n')
    
    # Process each image
    total_time = 0.0
    successful = 0
    failed = 0
    
    for idx, img_path in enumerate(image_files, 1):
        try:
            print(f'[{idx}/{len(image_files)}] {img_path.name}', end=' | ')
            
            # Load image
            image_bgr = cv2.imread(str(img_path))
            if image_bgr is None:
                print('ERROR: Could not load')
                failed += 1
                continue
            
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            h, w = image_rgb.shape[:2]
            
            # Run inference
            start = time.perf_counter()
            mask = segformer.predict(image_rgb)
            elapsed = (time.perf_counter() - start) * 1000
            total_time += elapsed
            
            print(f'Time: {elapsed:.1f}ms | Size: {w}x{h}', end='')
            
            # Print top classes if stats requested
            if args.stats:
                stats = segformer.get_class_stats(mask)
                top_class = list(sorted(stats.items(), key=lambda x: -x[1]['percentage'])[0])
                print(f' | Top: {top_class[0]} ({top_class[1]["percentage"]:.1f}%)', end='')
            
            # Generate output
            if args.overlay:
                result = segformer.overlay(image_rgb, mask, alpha=args.alpha)
                suffix = '_overlay'
            else:
                result = segformer.colorize(mask)
                suffix = '_segmented'
            
            # Save with model name in filename
            model_name = args.model
            out_file = inference_folder / f'{img_path.stem}_inference_{model_name}{suffix}.png'
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_file), result_bgr)
            
            print(' | ✓ Saved')
            successful += 1
            
        except Exception as e:
            print(f'ERROR: {str(e)}')
            failed += 1
            continue
    
    # Summary
    print()
    print('=' * 70)
    print(f'Processed: {successful}/{len(image_files)} images')
    if failed > 0:
        print(f'Failed:    {failed} images')
    if successful > 0:
        avg_time = total_time / successful
        fps = 1000.0 / avg_time
        print(f'Speed:     {avg_time:.1f}ms per image ({fps:.1f} FPS)')
    print(f'Output:    {inference_folder}')
    print('=' * 70)
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    main()
