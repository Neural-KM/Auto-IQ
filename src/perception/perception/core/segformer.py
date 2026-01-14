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

Usage:
    python -m perception.core.segformer --image path/to/image.jpg --output path/to/output.png
"""

import numpy as np
import cv2
import torch
import time
from pathlib import Path
from typing import Optional, Tuple, Union

from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


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
    
    @staticmethod
    def get_class_name(class_id: int) -> str:
        """Get class name from class ID."""
        if 0 <= class_id < len(CITYSCAPES_CLASSES):
            return CITYSCAPES_CLASSES[class_id]
        return 'unknown'
    
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
            class_name = self.get_class_name(class_id)
            stats[class_name] = {
                'class_id': int(class_id),
                'pixel_count': int(count),
                'percentage': float(count / total_pixels * 100)
            }
        
        return stats


def main():
    """Test SegFormer on stored images."""
    import argparse
    
    # Default directories relative to this script
    SCRIPT_DIR = Path(__file__).resolve().parent.parent  # perception/perception/
    DATA_DIR = SCRIPT_DIR / 'data'
    INFERENCE_DIR = DATA_DIR / 'inference'
    
    parser = argparse.ArgumentParser(
        description='SegFormer Semantic Segmentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f'''
Default paths:
  Input:  {DATA_DIR}/
  Output: {INFERENCE_DIR}/

Examples:
  # Single image (looks in data/ directory)
  python -m perception.core.segformer --image snapshot.png
  
  # Full path override
  python -m perception.core.segformer --image /path/to/image.jpg
  
  # Different model variant
  python -m perception.core.segformer --image snapshot.png --model b2
  
  # Show overlay instead of colored mask
  python -m perception.core.segformer --image snapshot.png --overlay --alpha 0.6
        '''
    )
    
    parser.add_argument('--image', '-i', type=str, required=True,
                        help='Input image name (in data/) or full path')
    parser.add_argument('--model', '-m', type=str, default='b0',
                        choices=['b0', 'b1', 'b2', 'b3', 'b4', 'b5'],
                        help='Model variant (b0=fastest, b5=most accurate)')
    parser.add_argument('--device', '-d', type=str, default=None,
                        choices=['cuda', 'cpu'],
                        help='Device for inference (default: auto)')
    parser.add_argument('--overlay', action='store_true',
                        help='Output overlay instead of colored mask')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Overlay blend factor (0-1)')
    parser.add_argument('--stats', action='store_true',
                        help='Print class statistics')
    parser.add_argument('--show', action='store_true',
                        help='Display result in window')
    
    args = parser.parse_args()
    
    # Resolve input path - check if it's just a filename or a full path
    input_path = Path(args.image)
    if not input_path.is_absolute() and not input_path.exists():
        # Try looking in the data directory
        input_path = DATA_DIR / args.image
    
    # Initialize model
    segformer = SegFormer(model_name=args.model, device=args.device)
    
    # Get input files
    if input_path.is_dir():
        image_files = list(input_path.glob('*.jpg')) + \
                      list(input_path.glob('*.jpeg')) + \
                      list(input_path.glob('*.png'))
    else:
        image_files = [input_path]
    
    if not image_files or (len(image_files) == 1 and not image_files[0].exists()):
        print(f'No images found at {input_path}')
        print(f'Tip: Place images in {DATA_DIR}/ or provide full path')
        return
    
    # Ensure inference output directory exists
    INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f'Processing {len(image_files)} image(s)...')
    print(f'Output directory: {INFERENCE_DIR}\n')
    
    # Process each image
    total_time = 0.0
    
    for img_path in image_files:
        print(f'Processing: {img_path.name}')
        
        # Load image
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f'  ERROR: Could not load image')
            continue
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Run inference
        start = time.perf_counter()
        mask = segformer.predict(image_rgb)
        elapsed = (time.perf_counter() - start) * 1000
        total_time += elapsed
        
        print(f'  Inference: {elapsed:.1f}ms')
        
        # Print stats if requested
        if args.stats:
            stats = segformer.get_class_stats(mask)
            print('  Class distribution:')
            for name, info in sorted(stats.items(), key=lambda x: -x[1]['percentage']):
                print(f'    {name}: {info["percentage"]:.1f}%')
        
        # Generate output
        if args.overlay:
            result = segformer.overlay(image_rgb, mask, alpha=args.alpha)
        else:
            result = segformer.colorize(mask)
        
        # Output path is always in inference directory
        out_file = INFERENCE_DIR / f'{img_path.stem}_segmented.png'
        
        # Save result (convert RGB to BGR for cv2)
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_file), result_bgr)
        print(f'  Saved: {out_file}')
        
        # Show if requested
        if args.show:
            # Create side-by-side view
            h, w = image_rgb.shape[:2]
            combined = np.zeros((h, w * 2, 3), dtype=np.uint8)
            combined[:, :w] = image_bgr
            combined[:, w:] = result_bgr
            
            cv2.imshow('SegFormer: Original | Segmentation', combined)
            key = cv2.waitKey(0)
            if key == 27:  # ESC
                break
    
    if args.show:
        cv2.destroyAllWindows()
    
    # Summary
    if len(image_files) > 1:
        avg_time = total_time / len(image_files)
        print(f'\nProcessed {len(image_files)} images')
        print(f'Average inference time: {avg_time:.1f}ms ({1000/avg_time:.1f} FPS)')


if __name__ == '__main__':
    main()
