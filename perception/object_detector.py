"""
Object Detector
===============

YOLO-based 2D object detector.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch

from config.config import PERCEPTION


class ObjectDetector:
    """
    YOLO-based object detector.
    """
    
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.5):
        """
        Initialize object detector.
        
        Args:
            model_path: Path to YOLO model file or name (e.g. 'yolov8n.pt')
            conf_threshold: Confidence threshold for detections
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
        hood_mask = PERCEPTION['ego_hood_mask']
        self._hood_mask_enabled = hood_mask['enabled']
        self._hood_mask_points = {
            'top_y': hood_mask['top_y'],
            'top_left_x': hood_mask['top_left_x'],
            'top_right_x': hood_mask['top_right_x'],
            'bottom_y': hood_mask['bottom_y'],
            'bottom_left_x': hood_mask['bottom_left_x'],
            'bottom_right_x': hood_mask['bottom_right_x'],
        }
        
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
