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
        
        # Detection region - narrower to focus on the lane ahead
        # Only center 30% of image width (was 40%)
        self.roi_left = int(self.width * 0.35)
        self.roi_right = int(self.width * 0.65)
        self.roi_top = int(self.height * 0.4)
        self.roi_bottom = int(self.height * 0.75)  # Don't go too low (catches own car hood)
        
        # Detection thresholds
        self.min_obstacle_pixels = 150  # Increased from 100 for fewer false positives
        self.detection_distance = Settings.OBSTACLE_DETECTION_DISTANCE
        
        # Debug visualization data (updated each detect() call)
        self.debug_data = {
            'roi_image': None,          # The ROI region extracted
            'vehicle_mask': None,       # Raw vehicle detections
            'pedestrian_mask': None,    # Raw pedestrian detections  
            'road_mask': None,          # Road pixels in ROI
            'close_mask': None,         # Pixels within detection distance
            'raw_obstacle_mask': None,  # Before road filtering
            'filtered_mask': None,      # After road filtering (final)
            'roi_bounds': (self.roi_left, self.roi_top, self.roi_right, self.roi_bottom)
        }
    
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
        
        # Store ROI for debug visualization
        self.debug_data['roi_image'] = sem_roi.copy()
        
        # Only detect actual obstacles: vehicles and pedestrians
        vehicle_mask = sem_roi == Settings.SemanticLabels.VEHICLE
        pedestrian_mask = sem_roi == Settings.SemanticLabels.PEDESTRIAN
        
        # Check if obstacles are near road surface (not on sidewalk/roadside)
        road_mask = sem_roi == Settings.SemanticLabels.ROAD
        
        # Store masks for debug
        self.debug_data['vehicle_mask'] = vehicle_mask.copy()
        self.debug_data['pedestrian_mask'] = pedestrian_mask.copy()
        self.debug_data['road_mask'] = road_mask.copy()
        
        # Potential obstacles: vehicles or pedestrians within detection distance
        close_mask = depth_roi < self.detection_distance
        self.debug_data['close_mask'] = close_mask.copy()
        
        obstacle_mask = (vehicle_mask | pedestrian_mask) & close_mask
        self.debug_data['raw_obstacle_mask'] = obstacle_mask.copy()
        
        # Additional filter: only count obstacles that have road pixels nearby (vertically below)
        # This filters out roadside parked cars, barriers, etc.
        filtered_obstacle_mask = np.zeros_like(obstacle_mask)
        obstacle_indices = np.where(obstacle_mask)
        
        if len(obstacle_indices[0]) > 0:
            for i in range(len(obstacle_indices[0])):
                y, x = obstacle_indices[0][i], obstacle_indices[1][i]
                # Check if there's road below this obstacle pixel (within a few pixels)
                check_below = min(y + 20, sem_roi.shape[0])
                if np.any(road_mask[y:check_below, max(0,x-5):min(x+5, sem_roi.shape[1])]):
                    filtered_obstacle_mask[y, x] = True
        
        obstacle_mask = filtered_obstacle_mask
        self.debug_data['filtered_mask'] = obstacle_mask.copy()
        
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
        
        # Get bounding box of obstacle pixels (in full image coordinates)
        min_y = int(np.min(obstacle_indices[0])) + self.roi_top
        max_y = int(np.max(obstacle_indices[0])) + self.roi_top
        min_x = int(np.min(obstacle_indices[1])) + self.roi_left
        max_x = int(np.max(obstacle_indices[1])) + self.roi_left
        bbox = (min_x, min_y, max_x, max_y)  # (x1, y1, x2, y2)
        
        # Determine obstacle type (vehicle or pedestrian)
        vehicle_count = np.sum(vehicle_mask & close_mask)
        pedestrian_count = np.sum(pedestrian_mask & close_mask)
        if vehicle_count > pedestrian_count:
            obstacle_type = 'vehicle'
        else:
            obstacle_type = 'pedestrian'
        
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
            'bbox': bbox,
            'obstacle_type': obstacle_type,
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
    
    def get_debug_images(self):
        """
        Get colored debug images showing intermediate detection steps.
        
        Returns:
            Dict with colored numpy arrays for visualization (RGB format):
            - 'roi_semantic': Colored semantic ROI
            - 'vehicle_mask': Vehicle detections (blue)
            - 'road_mask': Road pixels (purple)
            - 'raw_obstacles': Before filtering (yellow)
            - 'filtered_obstacles': After filtering (red/green)
            - 'combined': All masks overlaid
        """
        debug_images = {}
        
        # ROI semantic colored (RGB format)
        if self.debug_data['roi_image'] is not None:
            roi = self.debug_data['roi_image']
            h, w = roi.shape
            
            # Color the semantic labels (RGB)
            colored_roi = np.zeros((h, w, 3), dtype=np.uint8)
            colored_roi[roi == Settings.SemanticLabels.ROAD] = [128, 64, 128]  # Purple - road
            colored_roi[roi == Settings.SemanticLabels.ROAD_LINE] = [157, 234, 50]  # Green/yellow - road markings
            colored_roi[roi == Settings.SemanticLabels.VEHICLE] = [0, 0, 142]  # Dark blue - vehicle (CARLA color)
            colored_roi[roi == Settings.SemanticLabels.PEDESTRIAN] = [220, 20, 60]  # Crimson - pedestrian
            colored_roi[roi == Settings.SemanticLabels.SIDEWALK] = [244, 35, 232]  # Pink - sidewalk
            colored_roi[roi == Settings.SemanticLabels.VEGETATION] = [107, 142, 35]  # Olive - vegetation
            colored_roi[roi == Settings.SemanticLabels.TERRAIN] = [152, 251, 152]  # Light green - terrain
            debug_images['roi_semantic'] = colored_roi
        
        # Vehicle mask (blue on black) - RGB
        if self.debug_data['vehicle_mask'] is not None:
            mask = self.debug_data['vehicle_mask']
            h, w = mask.shape
            img = np.zeros((h, w, 3), dtype=np.uint8)
            img[mask] = [0, 100, 255]  # Blue
            debug_images['vehicle_mask'] = img
        
        # Road mask (purple on black)
        if self.debug_data['road_mask'] is not None:
            mask = self.debug_data['road_mask']
            h, w = mask.shape
            img = np.zeros((h, w, 3), dtype=np.uint8)
            img[mask] = [128, 64, 128]  # Purple
            debug_images['road_mask'] = img
        
        # Raw obstacles before filtering (yellow) - RGB
        if self.debug_data['raw_obstacle_mask'] is not None:
            mask = self.debug_data['raw_obstacle_mask']
            h, w = mask.shape
            img = np.zeros((h, w, 3), dtype=np.uint8)
            img[mask] = [255, 255, 0]  # Yellow
            debug_images['raw_obstacles'] = img
        
        # Filtered obstacles (green = kept, red = filtered out) - RGB
        if self.debug_data['raw_obstacle_mask'] is not None and self.debug_data['filtered_mask'] is not None:
            raw = self.debug_data['raw_obstacle_mask']
            filtered = self.debug_data['filtered_mask']
            h, w = raw.shape
            img = np.zeros((h, w, 3), dtype=np.uint8)
            # Red = was obstacle but filtered out (RGB)
            img[raw & ~filtered] = [255, 0, 0]
            # Green = kept as valid obstacle (RGB)
            img[filtered] = [0, 255, 0]
            debug_images['filtered_obstacles'] = img
        
        # Combined overlay (RGB format)
        if self.debug_data['roi_image'] is not None:
            roi = self.debug_data['roi_image']
            h, w = roi.shape
            combined = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Base: road in dark purple (RGB)
            if self.debug_data['road_mask'] is not None:
                combined[self.debug_data['road_mask']] = [60, 30, 60]
            
            # Vehicles in blue (RGB)
            if self.debug_data['vehicle_mask'] is not None:
                combined[self.debug_data['vehicle_mask']] = [0, 100, 255]
            
            # Final detections highlighted in bright green (RGB)
            if self.debug_data['filtered_mask'] is not None:
                combined[self.debug_data['filtered_mask']] = [0, 255, 0]
            
            debug_images['combined'] = combined
        
        return debug_images
    
    def get_roi_bounds(self):
        """Get the ROI boundaries."""
        return self.debug_data['roi_bounds']
