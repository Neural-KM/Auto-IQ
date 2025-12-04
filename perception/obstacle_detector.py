"""
Obstacle Detector
=================

Detects static obstacles on the road ahead.
"""

import numpy as np
from config import Settings


class ObstacleDetector:
    """
    Obstacle detection using semantic segmentation and depth.
    
    Detects objects on the road that aren't road surface.
    """
    
    def __init__(self, image_width=None, image_height=None):
        """
        Initialize obstacle detector.
        
        Args:
            image_width: Camera image width
            image_height: Camera image height
        """
        self.width = image_width or Settings.CAMERA_WIDTH
        self.height = image_height or Settings.CAMERA_HEIGHT
        
        # Detection region (center of image, lower portion)
        self.roi_left = int(self.width * 0.3)
        self.roi_right = int(self.width * 0.7)
        self.roi_top = int(self.height * 0.4)
        self.roi_bottom = int(self.height * 0.8)
        
        # Detection thresholds
        self.min_obstacle_pixels = 100  # Minimum pixels to consider as obstacle
        self.detection_distance = Settings.OBSTACLE_DETECTION_DISTANCE
    
    def detect(self, semantic_labels, depth_meters):
        """
        Detect obstacles ahead.
        
        Args:
            semantic_labels: 2D array of semantic labels
            depth_meters: 2D array of depth in meters
            
        Returns:
            Dictionary with detection results
        """
        if semantic_labels is None or depth_meters is None:
            return {
                'obstacle_detected': False,
                'distance': float('inf'),
                'position': None
            }
        
        # Get ROI
        sem_roi = semantic_labels[self.roi_top:self.roi_bottom, 
                                   self.roi_left:self.roi_right]
        depth_roi = depth_meters[self.roi_top:self.roi_bottom,
                                  self.roi_left:self.roi_right]
        
        # Road and lane lines are not obstacles
        road_mask = sem_roi == Settings.SemanticLabels.ROAD
        lane_mask = sem_roi == Settings.SemanticLabels.ROAD_LINE
        sidewalk_mask = sem_roi == Settings.SemanticLabels.SIDEWALK
        
        drivable_mask = road_mask | lane_mask | sidewalk_mask
        
        # Potential obstacles: anything on road that isn't road surface
        # and is within detection distance
        close_mask = depth_roi < self.detection_distance
        obstacle_mask = ~drivable_mask & close_mask
        
        # Check if enough obstacle pixels
        obstacle_count = np.sum(obstacle_mask)
        
        if obstacle_count < self.min_obstacle_pixels:
            return {
                'obstacle_detected': False,
                'distance': float('inf'),
                'position': None,
                'pixel_count': obstacle_count
            }
        
        # Get obstacle distance (minimum depth in obstacle region)
        obstacle_distances = depth_roi[obstacle_mask]
        min_distance = float(np.min(obstacle_distances))
        
        # Get obstacle position (center of obstacle pixels)
        obstacle_indices = np.where(obstacle_mask)
        center_y = int(np.mean(obstacle_indices[0])) + self.roi_top
        center_x = int(np.mean(obstacle_indices[1])) + self.roi_left
        
        # Determine if obstacle is left, center, or right
        image_center = self.width // 2
        if center_x < image_center - self.width * 0.1:
            position = 'left'
        elif center_x > image_center + self.width * 0.1:
            position = 'right'
        else:
            position = 'center'
        
        return {
            'obstacle_detected': True,
            'distance': min_distance,
            'position': position,
            'center': (center_x, center_y),
            'pixel_count': obstacle_count
        }
    
    def should_trigger_lane_change(self, detection_result):
        """
        Determine if obstacle requires lane change.
        
        Args:
            detection_result: Result from detect()
            
        Returns:
            Boolean indicating if lane change should be triggered
        """
        if not detection_result['obstacle_detected']:
            return False
        
        # Trigger lane change if obstacle is ahead and close enough
        distance = detection_result['distance']
        position = detection_result['position']
        
        return (position == 'center' and 
                distance < Settings.LANE_CHANGE_TRIGGER_DISTANCE)
    
    def get_avoidance_direction(self, detection_result):
        """
        Get recommended direction to avoid obstacle.
        
        Args:
            detection_result: Result from detect()
            
        Returns:
            'left' or 'right'
        """
        position = detection_result.get('position', 'center')
        
        # If obstacle is on one side, go to the other
        if position == 'left':
            return 'right'
        elif position == 'right':
            return 'left'
        else:
            # Default to right for center obstacles (right-hand traffic)
            return 'left'
