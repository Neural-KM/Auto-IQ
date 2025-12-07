"""
Lane Detector
=============

Detects lanes from camera input.
"""

import numpy as np
from config import Settings
from sensors.camera_semantic import SemanticLabels


class LaneDetector:
    """
    Lane detection using semantic segmentation.
    
    This implementation uses CARLA's semantic segmentation camera.
    Can be extended with actual CV/ML-based detection later.
    """
    
    def __init__(self, image_width=None, image_height=None):
        """
        Initialize lane detector.
        
        Args:
            image_width: Camera image width
            image_height: Camera image height
        """
        self.width = image_width or Settings.CAMERA_WIDTH
        self.height = image_height or Settings.CAMERA_HEIGHT
        
        # Region of interest (bottom portion of image)
        self.roi_top = int(self.height * 0.5)  # Top 50% ignored
        self.roi_bottom = self.height
        
        # Lane center tracking
        self.lane_center = self.width // 2
        self.image_center = self.width // 2
    
    def detect(self, semantic_labels):
        """
        Detect lane center from semantic labels.
        
        Args:
            semantic_labels: 2D array of semantic labels from camera
            
        Returns:
            Dictionary with detection results
        """
        if semantic_labels is None:
            return {'valid': False, 'lane_center': self.image_center, 'error': 0}
        
        # Get road pixels in ROI
        roi = semantic_labels[self.roi_top:self.roi_bottom, :]
        road_mask = roi == SemanticLabels.ROAD
        
        # Find road boundaries by scanning horizontally
        # at different heights in the ROI
        lane_centers = []
        
        for row_offset in [0.3, 0.5, 0.7]:  # Sample at different heights
            row = int(road_mask.shape[0] * row_offset)
            road_row = road_mask[row, :]
            
            if road_row.any():
                # Find leftmost and rightmost road pixels
                road_indices = np.where(road_row)[0]
                left_edge = road_indices[0]
                right_edge = road_indices[-1]
                center = (left_edge + right_edge) // 2
                lane_centers.append(center)
        
        if lane_centers:
            # Average lane center with temporal smoothing
            detected_center = int(np.mean(lane_centers))
            self.lane_center = int(0.7 * self.lane_center + 0.3 * detected_center)
        
        # Compute cross-track error (positive = vehicle is left of center)
        error = (self.image_center - self.lane_center) / self.image_center
        
        return {
            'valid': True,
            'lane_center': self.lane_center,
            'error': error,  # Normalized error [-1, 1]
            'error_pixels': self.image_center - self.lane_center
        }
    
    def visualize(self, image, detection_result):
        """
        Draw lane detection visualization on image.
        
        Args:
            image: RGB image (numpy array)
            detection_result: Result from detect()
            
        Returns:
            Image with visualization
        """
        if image is None:
            return None
        
        vis_image = image.copy()
        
        # Draw lane center line
        lane_center = detection_result.get('lane_center', self.image_center)
        
        # Green line for detected lane center
        vis_image[self.roi_top:, lane_center-2:lane_center+2] = [0, 255, 0]
        
        # Red line for image center
        vis_image[self.roi_top:, self.image_center-1:self.image_center+1] = [255, 0, 0]
        
        return vis_image
