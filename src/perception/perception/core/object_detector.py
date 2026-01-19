"""
Object Detector
===============

YOLO-based 2D object detector.


YOLO comparison data::
Benchmark Results:
+-------------+----------------------+-------+----------------------+
| Model       |   Avg Inference (ms) |   FPS |   Avg Detections/Img |
+=============+======================+=======+======================+
| yolov5nu.pt |                13.32 | 75.1  |                  0.4 |
+-------------+----------------------+-------+----------------------+
| yolov5su.pt |                12.87 | 77.72 |                  0.7 |
+-------------+----------------------+-------+----------------------+
| yolov5mu.pt |                15.75 | 63.5  |                  0.7 |
+-------------+----------------------+-------+----------------------+
| yolov5lu.pt |                19.84 | 50.41 |                  0.7 |
+-------------+----------------------+-------+----------------------+
| yolov5xu.pt |                27.53 | 36.32 |                  0.8 |
+-------------+----------------------+-------+----------------------+
| yolov8n.pt  |                11.14 | 89.79 |                  0.7 |
+-------------+----------------------+-------+----------------------+
| yolov8s.pt  |                11.21 | 89.19 |                  0.7 |
+-------------+----------------------+-------+----------------------+
| yolov8m.pt  |                16.28 | 61.43 |                  0.8 |
+-------------+----------------------+-------+----------------------+
| yolov8l.pt  |                19.59 | 51.05 |                  0.8 |
+-------------+----------------------+-------+----------------------+
| yolov8x.pt  |                25.72 | 38.88 |                  0.6 |
+-------------+----------------------+-------+----------------------+
| yolov9t.pt  |                15.7  | 63.69 |                  0.8 |
+-------------+----------------------+-------+----------------------+
| yolov9c.pt  |                17.5  | 57.16 |                  0.6 |
+-------------+----------------------+-------+----------------------+
| yolov9e.pt  |                27.77 | 36.01 |                  0.7 |
+-------------+----------------------+-------+----------------------+
| yolov10n.pt |                10.18 | 98.19 |                  0.6 |
+-------------+----------------------+-------+----------------------+
| yolov10s.pt |                11.37 | 87.98 |                  0.4 |
+-------------+----------------------+-------+----------------------+
| yolov10m.pt |                13.1  | 76.35 |                  0.6 |
+-------------+----------------------+-------+----------------------+
| yolov10b.pt |                14.28 | 70.04 |                  0.6 |
+-------------+----------------------+-------+----------------------+
| yolov10l.pt |                15.72 | 63.61 |                  0.6 |
+-------------+----------------------+-------+----------------------+
| yolov10x.pt |                20.78 | 48.12 |                  0.7 |
+-------------+----------------------+-------+----------------------+
| yolo11n.pt  |                12.02 | 83.23 |                  0.6 |
+-------------+----------------------+-------+----------------------+
| yolo11s.pt  |                12.3  | 81.31 |                  0.6 |
+-------------+----------------------+-------+----------------------+
| yolo11m.pt  |                14.14 | 70.71 |                  0.6 |
+-------------+----------------------+-------+----------------------+
| yolo11l.pt  |                17.53 | 57.04 |                  0.6 |
+-------------+----------------------+-------+----------------------+
| yolo11x.pt  |                24.53 | 40.77 |                  0.7 |
+-------------+----------------------+-------+----------------------+

Run on images taken from phone inside lab with chairs etc.

"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch


class ObjectDetector:
    """
    YOLO-based object detector.
    """
    
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.5, hood_mask_config=None):
        """
        Initialize object detector.
        
        Args:
            model_path: Path to YOLO model file or name (e.g. 'yolov8n.pt')
            conf_threshold: Confidence threshold for detections
            hood_mask_config: Dict with hood mask settings (enabled, top_y, etc.)
        """
        self.conf_threshold = conf_threshold
        
        # Check for GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[ObjectDetector] Loading model {model_path} on {self.device}...")
        
        try:
            self.model = YOLO(model_path)
            print(f"[ObjectDetector] Model loaded successfully")
        except Exception as e:
            print(f"[ObjectDetector] Failed to load model: {e}")
            self.model = None
            
        # COCO classes mapping (we care about vehicles and pedestrians)
        # 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
        self.target_classes = {
            0: 'pedestrian',
            1: 'vehicle', # bicycle treated as vehicle
            2: 'vehicle',
            3: 'vehicle',
            5: 'vehicle',
            7: 'vehicle'
        }
        
        # Ego vehicle hood mask (trapezoid to black out hood region)
        if hood_mask_config and hood_mask_config.get('enabled', False):
            self._hood_mask_enabled = True
            self._hood_mask_points = {
                'top_y': hood_mask_config.get('top_y', 0.85),
                'top_left_x': hood_mask_config.get('top_left_x', 0.25),
                'top_right_x': hood_mask_config.get('top_right_x', 0.75),
                'bottom_y': hood_mask_config.get('bottom_y', 1.0),
                'bottom_left_x': hood_mask_config.get('bottom_left_x', 0.08),
                'bottom_right_x': hood_mask_config.get('bottom_right_x', 0.92),
            }
        else:
            self._hood_mask_enabled = False
            self._hood_mask_points = {}
        
        # Store last masked image for visualization
        self._last_masked_image = None
    
    def _apply_hood_mask(self, image):
        """
        Apply trapezoid mask to black out ego vehicle hood region.
        
        Args:
            image: RGB numpy array (H, W, 3)
            
        Returns:
            Masked image with hood region blacked out
        """
        if not self._hood_mask_enabled:
            return image
        
        h, w = image.shape[:2]
        p = self._hood_mask_points
        
        # Convert normalized coordinates to pixel coordinates
        # Trapezoid: top-left, top-right, bottom-right, bottom-left
        pts = np.array([
            [int(p['top_left_x'] * w), int(p['top_y'] * h)],
            [int(p['top_right_x'] * w), int(p['top_y'] * h)],
            [int(p['bottom_right_x'] * w), int(p['bottom_y'] * h)],
            [int(p['bottom_left_x'] * w), int(p['bottom_y'] * h)],
        ], dtype=np.int32)
        
        # Create masked copy
        masked = image.copy()
        cv2.fillPoly(masked, [pts], (0, 0, 0))
        
        return masked
    
    def detect(self, image):
        """
        Detect objects in an image.
        
        Args:
            image: RGB numpy array (H, W, 3)
            
        Returns:
            List of detections: [{'bbox': [x1, y1, x2, y2], 'class': str, 'score': float}, ...]
        """
        if self.model is None or image is None:
            return []
        
        # Apply hood mask to prevent detecting ego vehicle
        masked_image = self._apply_hood_mask(image)
        self._last_masked_image = masked_image
            
        # Run inference
        # verbose=False to suppress stdout
        results = self.model(masked_image, verbose=False, conf=self.conf_threshold, device=self.device)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                
                # Filter for target classes
                if cls_id in self.target_classes:
                    class_name = self.target_classes[cls_id]
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy() # [x1, y1, x2, y2]
                    
                    detections.append({
                        'bbox': xyxy.tolist(),
                        'class': class_name,
                        'score': conf
                    })
                    
        return detections
    
    def get_masked_image(self):
        """Get the last masked image used for detection."""
        return self._last_masked_image
