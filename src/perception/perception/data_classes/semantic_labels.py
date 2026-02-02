"""
CARLA Semantic Segmentation Labels
===================================

Defines semantic class IDs, names, and colors for CARLA ground truth segmentation.
"""

import numpy as np
from typing import Tuple, Dict, Optional


# (id, name, color) tuples - name is derived automatically
_CLASSES = [
    (0,  'unlabeled',     (0, 0, 0)),
    (1,  'road',          (128, 64, 128)),
    (2,  'sidewalk',      (244, 35, 232)),
    (3,  'building',      (70, 70, 70)),
    (4,  'wall',          (102, 102, 156)),
    (5,  'fence',         (190, 153, 153)),
    (6,  'pole',          (153, 153, 153)),
    (7,  'traffic_light', (250, 170, 30)),
    (8,  'traffic_sign',  (220, 220, 0)),
    (9,  'vegetation',    (107, 142, 35)),
    (10, 'terrain',       (152, 251, 152)),
    (11, 'sky',           (70, 130, 180)),
    (12, 'pedestrian',    (220, 20, 60)),
    (13, 'rider',         (255, 0, 0)),
    (14, 'vehicle',       (0, 0, 142)),
    (15, 'truck',         (0, 0, 70)),
    (16, 'bus',           (0, 60, 100)),
    (17, 'train',         (0, 80, 100)),
    (18, 'motorcycle',    (0, 0, 230)),
    (19, 'bicycle',       (119, 11, 32)),
    (20, 'static',        (110, 190, 160)),
    (21, 'dynamic',       (170, 120, 50)),
    (22, 'other',         (55, 90, 80)),
    (23, 'water',         (45, 60, 150)),
    (24, 'road_line',     (157, 234, 50)),
    (25, 'ground',        (81, 0, 81)),
    (26, 'bridge',        (150, 100, 100)),
    (27, 'rail_track',    (230, 150, 140)),
    (28, 'guard_rail',    (180, 165, 180)),
]

# Lookup by ID: {0: ('unlabeled', (0,0,0)), 1: ('road', (128,64,128)), ...}
SEMANTIC_CLASSES: Dict[int, Tuple[str, Tuple[int, int, int]]] = {
    id: (name, color) for id, name, color in _CLASSES
}

# Lookup by name
SEMANTIC_BY_NAME: Dict[str, int] = {name: id for id, name, _ in _CLASSES}


# Color lookup table (256 entries for fast numpy indexing)
SEMANTIC_COLOR_LUT = np.zeros((256, 3), dtype=np.uint8)
for id, (_, color) in SEMANTIC_CLASSES.items():
    SEMANTIC_COLOR_LUT[id] = color


def get_class_name(class_id: int) -> Optional[str]:
    """Get class name by ID."""
    entry = SEMANTIC_CLASSES.get(class_id)
    return entry[0] if entry else None


def get_class_color(class_id: int) -> Optional[Tuple[int, int, int]]:
    """Get class color by ID."""
    entry = SEMANTIC_CLASSES.get(class_id)
    return entry[1] if entry else None


def get_class_id(name: str) -> Optional[int]:
    """Get class ID by name."""
    return SEMANTIC_BY_NAME.get(name.lower())


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Convert semantic mask (class IDs) to RGB image.
    
    Args:
        mask: 2D array of class IDs (H, W)
        
    Returns:
        RGB image (H, W, 3)
    """
    return SEMANTIC_COLOR_LUT[np.clip(mask, 0, 255)]


def get_class_stats(mask: np.ndarray) -> dict:
    """
    Get pixel statistics for each class in the segmentation mask.
    
    Useful for analyzing scene composition (e.g., percentage of road, 
    buildings, vegetation in an image).
    
    Args:
        mask: Segmentation mask (H, W) with class IDs
        
    Returns:
        Dict mapping class names to pixel counts and percentages
        Example: {'road': {'class_id': 1, 'pixel_count': 50000, 'percentage': 45.5}, ...}
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
