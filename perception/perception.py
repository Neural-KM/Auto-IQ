"""
Real Perception
===============

Obstacle detection from raw sensor data (RGB + Depth cameras).
Uses YOLO for 2D object detection, then projects to 3D using depth.
"""

import math
import time
import numpy as np
import cv2
from typing import List, Optional

from state.obstacle import Obstacle
from perception.object_detector import ObjectDetector


class Perception:
    """
    Real perception using camera sensor data.
    
    Pipeline:
    1. Get RGB image from sensor suite
    2. Run YOLO object detection -> 2D bounding boxes
    3. Get depth at each detection center
    4. Project 2D+depth to 3D coordinates (camera frame)
    5. Transform to vehicle frame
    6. Return list of Obstacle instances
    """
    
    def __init__(self, vehicle, world, sensor_suite, model_path='yolov8n.pt', conf_threshold=0.5):
        """
        Initialize perception.
        
        Args:
            vehicle: CARLA vehicle actor
            world: CARLA world
            sensor_suite: SensorSuite instance with rgb and depth cameras
            model_path: Path to YOLO model
            conf_threshold: Detection confidence threshold
        """
        self.vehicle = vehicle
        self.world = world
        self.sensor_suite = sensor_suite
        
        # Initialize object detector
        self.detector = ObjectDetector(model_path=model_path, conf_threshold=conf_threshold)
        
        # Cache camera intrinsics (computed once when we get first image)
        self._intrinsics_computed = False
        self._fx = None
        self._fy = None
        self._cx = None
        self._cy = None
        
        # Camera transform relative to vehicle (will be set from sensor position)
        self._camera_transform = None
        
        # Store last detection results for visualization
        self._last_rgb_image = None
        self._last_detections = None
        
        print("[Perception] Real perception initialized")
    
    def _compute_intrinsics(self, width: int, height: int, fov: float) -> None:
        """
        Compute camera intrinsic parameters.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            fov: Horizontal field of view in degrees
        """
        # Focal length from horizontal FOV
        # fov = 2 * atan(width / (2 * fx))
        # fx = width / (2 * tan(fov/2))
        fov_rad = math.radians(fov)
        self._fx = width / (2.0 * math.tan(fov_rad / 2.0))
        self._fy = self._fx  # Square pixels
        self._cx = width / 2.0
        self._cy = height / 2.0
        self._intrinsics_computed = True
        
        print(f"[Perception] Camera intrinsics: fx={self._fx:.1f}, fy={self._fy:.1f}, cx={self._cx:.1f}, cy={self._cy:.1f}")
    
    def _pixel_to_camera_frame(self, u: float, v: float, depth: float) -> tuple:
        """
        Convert pixel coordinates + depth to 3D point in camera frame.
        
        Camera frame convention (looking from behind camera):
        - X: right
        - Y: down
        - Z: forward (into the scene)
        
        Args:
            u: Pixel x coordinate
            v: Pixel y coordinate
            depth: Depth in meters
            
        Returns:
            (x, y, z) in camera frame (meters)
        """
        # Standard pinhole camera model
        z = depth
        x = (u - self._cx) * z / self._fx
        y = (v - self._cy) * z / self._fy
        
        return (x, y, z)
    
    def _camera_to_vehicle_frame(self, x_cam: float, y_cam: float, z_cam: float,
                                  camera_position: dict) -> tuple:
        """
        Transform point from camera frame to vehicle frame.
        
        Vehicle frame convention:
        - X: forward
        - Y: right
        - Z: up
        
        Args:
            x_cam, y_cam, z_cam: Point in camera frame
            camera_position: Dict with x, y, z, pitch, yaw, roll of camera relative to vehicle
            
        Returns:
            (x, y, z) in vehicle frame (meters)
        """
        # Camera frame to vehicle frame rotation
        # Camera: X-right, Y-down, Z-forward
        # Vehicle: X-forward, Y-right, Z-up
        
        # First, convert camera frame to standard vehicle frame (no rotation)
        # Camera Z (forward) -> Vehicle X (forward)
        # Camera X (right) -> Vehicle Y (right)
        # Camera Y (down) -> Vehicle -Z (up)
        x_veh = z_cam
        y_veh = x_cam
        z_veh = -y_cam
        
        # Apply camera pitch rotation (around Y axis in vehicle frame)
        pitch_rad = math.radians(camera_position.get('pitch', 0.0))
        cos_p = math.cos(pitch_rad)
        sin_p = math.sin(pitch_rad)
        
        x_rot = x_veh * cos_p + z_veh * sin_p
        z_rot = -x_veh * sin_p + z_veh * cos_p
        
        # Add camera offset
        x_final = x_rot + camera_position.get('x', 0.0)
        y_final = y_veh + camera_position.get('y', 0.0)
        z_final = z_rot + camera_position.get('z', 0.0)
        
        return (x_final, y_final, z_final)
    
    def _estimate_dimensions(self, class_name: str) -> tuple:
        """
        Estimate object dimensions based on class.
        
        Args:
            class_name: 'vehicle' or 'pedestrian'
            
        Returns:
            (length, width, height) in meters
        """
        if class_name == 'vehicle':
            # Average car dimensions
            return (4.5, 1.8, 1.5)
        else:
            # Pedestrian dimensions
            return (0.5, 0.5, 1.7)
    
    def detect_obstacles(self, radius: float = 50.0) -> List[Obstacle]:
        """
        Detect obstacles from sensor data.
        
        Args:
            radius: Maximum detection radius in meters
            
        Returns:
            List of Obstacle instances, sorted by distance
        """
        if not self.vehicle or not self.vehicle.is_alive:
            return []
        
        # Get RGB camera
        rgb_sensor = self.sensor_suite.get_sensor('front_rgb')
        if rgb_sensor is None:
            print("[Perception] Warning: No RGB sensor found")
            return []
        
        rgb_image = rgb_sensor.get_data()
        if rgb_image is None:
            return []
        
        # Get Depth camera
        depth_sensor = self.sensor_suite.get_sensor('depth')
        if depth_sensor is None:
            print("[Perception] Warning: No depth sensor found")
            return []
        
        depth_image = depth_sensor.get_depth_meters()
        if depth_image is None:
            return []
        
        # Compute intrinsics if not done yet
        if not self._intrinsics_computed:
            self._compute_intrinsics(rgb_sensor.width, rgb_sensor.height, rgb_sensor.fov)
        
        # Run 2D detection with timing
        start_time = time.time()
        detections = self.detector.detect(rgb_image)
        inference_time = time.time() - start_time
        print(f"[Perception] YOLO inference time: {inference_time:.3f}s, detections: {len(detections)}")
        
        # Store for visualization
        self._last_rgb_image = rgb_image.copy()
        self._last_detections = detections
        
        if not detections:
            return []
        
        obstacles = []
        obstacle_id = 1000  # Start IDs at 1000 to distinguish from CARLA actor IDs
        
        for det in detections:
            bbox = det['bbox']  # [x1, y1, x2, y2]
            class_name = det['class']
            score = det['score']
            
            # Get center of bounding box
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            
            # Get depth at center (use median of center region for robustness)
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            # Sample depth from center region (middle 50%)
            margin_x = int((x2 - x1) * 0.25)
            margin_y = int((y2 - y1) * 0.25)
            
            roi_x1 = max(0, x1 + margin_x)
            roi_x2 = min(depth_image.shape[1], x2 - margin_x)
            roi_y1 = max(0, y1 + margin_y)
            roi_y2 = min(depth_image.shape[0], y2 - margin_y)
            
            if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
                # Fallback to center point
                depth = depth_image[int(cy), int(cx)]
            else:
                depth_roi = depth_image[roi_y1:roi_y2, roi_x1:roi_x2]
                # Use median to reject outliers (sky, ground)
                depth = np.median(depth_roi)
            
            # Skip if depth is invalid or too far
            if depth <= 0 or depth > radius:
                continue
            
            # Project to camera frame
            x_cam, y_cam, z_cam = self._pixel_to_camera_frame(cx, cy, depth)
            
            # Transform to vehicle frame
            camera_pos = rgb_sensor.position
            x_veh, y_veh, z_veh = self._camera_to_vehicle_frame(x_cam, y_cam, z_cam, camera_pos)
            
            # Estimate dimensions
            length, width, height = self._estimate_dimensions(class_name)
            
            # Create obstacle
            obstacle = Obstacle(
                id=obstacle_id,
                obstacle_type=class_name,
                x=x_veh,
                y=y_veh,
                z=z_veh,
                vx=0.0,  # Velocity estimation requires tracking (future work)
                vy=0.0,
                vz=0.0,
                length=length,
                width=width,
                height=height,
            )
            obstacles.append(obstacle)
            obstacle_id += 1
        
        # Sort by distance
        obstacles.sort(key=lambda o: (o.x**2 + o.y**2 + o.z**2)**0.5)
        
        return obstacles
    
    def get_annotated_image(self) -> Optional[np.ndarray]:
        """
        Get the last RGB image with YOLO detection bounding boxes drawn.
        
        Returns:
            Annotated RGB image as numpy array, or None if no image available
        """
        if self._last_rgb_image is None:
            return None
        
        # Make a copy to draw on
        annotated = self._last_rgb_image.copy()
        
        if self._last_detections:
            for det in self._last_detections:
                bbox = det['bbox']
                class_name = det['class']
                score = det['score']
                
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                
                # Color based on class
                color = (255, 0, 0) if class_name == 'vehicle' else (0, 255, 0)
                
                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{class_name} {score:.2f}"
                (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
                cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated
    
    def get_masked_image(self) -> Optional[np.ndarray]:
        """Get the masked image used for YOLO detection (with hood blacked out)."""
        return self.detector.get_masked_image()
