"""
Object Detector
===============

YOLO-based 2D object detector.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch

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
            
        # Run inference
        # verbose=False to suppress stdout
        results = self.model(image, verbose=False, conf=self.conf_threshold, device=self.device)
        
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
