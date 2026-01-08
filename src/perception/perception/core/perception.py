"""
Real Perception (ROS2 Version)
==============================

Obstacle detection from RGB camera + LiDAR using frustum filtering.
Uses YOLO for 2D object detection, then projects LiDAR points to 3D.

This is the ROS2 adaptation of the original Perception class.
Instead of using sensor_suite.get_sensor(), it accepts images directly.
"""

import math
import time
from lark import logger
import numpy as np
import cv2
from typing import List, Optional, Dict, Any

from perception.core.object_detector import ObjectDetector
from perception.data_classes.obstacle import Obstacle


class Perception:
    """
    Real perception using RGB camera + LiDAR frustum filtering.
    
    Pipeline:
    1. Get RGB image from sensor data
    2. Run YOLO object detection -> 2D bounding boxes
    3. Get LiDAR point cloud
    4. Project LiDAR points to image and filter by bounding boxes
    5. Cluster points and fit oriented bounding boxes
    6. Return list of Obstacle instances in vehicle frame
    """

    def __init__(self, model_path: str = 'yolov8n.pt', conf_threshold: float = 0.5,
                 method: int = 0, hood_mask_config: Optional[Dict] = None):
        """
        Initialize perception.
        
        Args:
            model_path: Path to YOLO model
            conf_threshold: Detection confidence threshold
            method: Detection method (must be 0 for LiDAR frustum filtering)
            hood_mask_config: Optional hood mask configuration for ObjectDetector
        """
        # Validate method
        if method != 0:
            raise ValueError(f"Only method=0 (LiDAR frustum filtering) is supported. Got method={method}")
        
        self.method = method
        
        # Initialize object detector
        self.detector = ObjectDetector(
            model_path=model_path, 
            conf_threshold=conf_threshold,
            hood_mask_config=hood_mask_config
        )
        
        # Cache camera intrinsics (computed once when we get first image)
        self._intrinsics_computed = False
        self._fx = None
        self._fy = None
        self._cx = None
        self._cy = None
        
        # Camera position relative to vehicle (set externally)
        self._camera_position = {'x': 0.0, 'y': 0.0, 'z': 1.5, 'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}
        
        # LiDAR position relative to vehicle (set externally)
        self._lidar_position = {'x': 0.0, 'y': 0.0, 'z': 2.5}
        
        # Store last detection results for visualization
        self._last_rgb_image = None
        self._last_detections = None
        
        print("[Perception] Real perception initialized")
    
    def set_intrinsics(self, fx: float, fy: float, cx: float, cy: float) -> None:
        """
        Set camera intrinsic parameters directly.
        
        Args:
            fx: Focal length x
            fy: Focal length y
            cx: Principal point x
            cy: Principal point y
        """
        self._fx = fx
        self._fy = fy
        self._cx = cx
        self._cy = cy
        self._intrinsics_computed = True
        print(f"[Perception] Camera intrinsics set: fx={self._fx:.1f}, fy={self._fy:.1f}, cx={self._cx:.1f}, cy={self._cy:.1f}")
    
    def set_camera_position(self, x: float, y: float, z: float, pitch: float = 0.0,
                            yaw: float = 0.0, roll: float = 0.0) -> None:
        """
        Set camera position relative to vehicle.
        
        Args:
            x, y, z: Position in vehicle frame
            pitch, yaw, roll: Orientation in degrees
        """
        self._camera_position = {
            'x': x, 'y': y, 'z': z,
            'pitch': pitch, 'yaw': yaw, 'roll': roll
        }
        print(f"[Perception] Camera position set: x={x:.2f}, y={y:.2f}, z={z:.2f}, pitch={pitch:.1f}")
    
    def set_lidar_position(self, x: float, y: float, z: float) -> None:
        """
        Set LiDAR position relative to vehicle.
        
        Args:
            x, y, z: Position in vehicle frame
        """
        self._lidar_position = {'x': x, 'y': y, 'z': z}
        print(f"[Perception] LiDAR position set: x={x:.2f}, y={y:.2f}, z={z:.2f}")
    
    def _compute_intrinsics(self, width: int, height: int, fov: float) -> None:
        """
        Compute camera intrinsic parameters from FOV.
        
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
        
        print(f"[Perception] Camera intrinsics computed: fx={self._fx:.1f}, fy={self._fy:.1f}, cx={self._cx:.1f}, cy={self._cy:.1f}")
    
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
    
    def detect_obstacles(self, rgb_image: np.ndarray,
                         lidar_points: Optional[np.ndarray] = None,
                         radius: float = 50.0) -> List[Obstacle]:
        """
        Detect obstacles from RGB camera + LiDAR using frustum filtering.
        
        Args:
            rgb_image: RGB image as numpy array (H, W, 3)
            lidar_points: LiDAR points as Nx3 or Nx4 numpy array (x, y, z, [intensity])
            radius: Maximum detection radius in meters
            
        Returns:
            List of Obstacle instances in vehicle frame, sorted by distance
        """
        if rgb_image is None or lidar_points is None or len(lidar_points) == 0:
            return []
        
        return self.detect_obstacles_lidar_frustum_cluster(rgb_image, lidar_points, radius)

    def detect_obstacles_lidar_frustum_cluster(self, rgb_image: np.ndarray,
                                               lidar_points: np.ndarray,
                                               radius: float = 50.0) -> List[Obstacle]:
        """
        Detect obstacles using Frustum Filtering with LiDAR.
        
        Algorithm:
        1. Frustum Filtering: Project LiDAR points to image and select those inside YOLO boxes
        2. Coordinate Transform: Convert points to Vehicle Frame
        3. Clustering: Remove outliers using statistical filtering
        4. Oriented Bounding Box Fitting:
           - Use PCA on XY plane to find principal axes
           - Compute oriented bounding box dimensions (length, width, height)
           - Extract yaw angle from principal axis
        
        Args:
            rgb_image: RGB image as numpy array (H, W, 3)
            lidar_points: LiDAR points as Nx3 or Nx4 numpy array (x, y, z, [intensity])
            radius: Maximum detection radius in meters
            
        Returns:
            List of Obstacle instances in vehicle frame, sorted by distance
        """
        if rgb_image is None:
            return []
        
        if lidar_points is None or len(lidar_points) == 0:
            return []
        
        # Compute intrinsics if not done yet
        if not self._intrinsics_computed:
            print("[Perception] Warning: Camera intrinsics not set. Call set_intrinsics() first.")
            return []
        
        # Take only x, y, z (handle both Nx3 and Nx4 arrays)
        lidar_points = lidar_points[:, :3] if lidar_points.shape[1] >= 3 else lidar_points
        
        # Run 2D detection
        start_time = time.time()
        detections = self.detector.detect(rgb_image)
        inference_time = time.time() - start_time
        # print(f"[Perception] YOLO inference time: {inference_time:.3f}s, detections: {len(detections)}")
        
        # Store for visualization
        self._last_rgb_image = rgb_image.copy()
        self._last_detections = detections
        
        if not detections:
            return []
        
        obstacles = []
        obstacle_id = 1000
        
        # 1. Transform LiDAR points to Vehicle Frame
        # LiDAR sensor usually outputs in its own frame (x-forward, y-right, z-up)
        lidar_pos = self._lidar_position
        lidar_points_veh = lidar_points + np.array([lidar_pos['x'], lidar_pos['y'], lidar_pos['z']])
        
        # 2. Transform Vehicle Frame points to Camera Frame for projection
        camera_pos = self._camera_position
        
        # Translate to camera position
        pts_rel = lidar_points_veh - np.array([camera_pos['x'], camera_pos['y'], camera_pos['z']])
        
        # Rotate to camera frame
        # Cam X (right) = Veh Y
        # Cam Y (down) = -Veh Z
        # Cam Z (forward) = Veh X
        lidar_points_cam = np.zeros_like(pts_rel)
        lidar_points_cam[:, 0] = pts_rel[:, 1]   # x = y_veh
        lidar_points_cam[:, 1] = -pts_rel[:, 2]  # y = -z_veh
        lidar_points_cam[:, 2] = pts_rel[:, 0]   # z = x_veh
        
        # Filter points behind camera (create mask before filtering)
        front_mask = lidar_points_cam[:, 2] > 0
        lidar_points_cam = lidar_points_cam[front_mask]
        lidar_points_veh = lidar_points_veh[front_mask]
        
        # Project to Image Plane
        u = (lidar_points_cam[:, 0] * self._fx / lidar_points_cam[:, 2]) + self._cx
        v = (lidar_points_cam[:, 1] * self._fy / lidar_points_cam[:, 2]) + self._cy
        
        # Stack u, v with vehicle frame points (for final box fitting)
        # projected_points: [u, v, x_veh, y_veh, z_veh]
        projected_points = np.column_stack((u, v, lidar_points_veh))
        
        for det in detections:
            bbox = det['bbox']  # [x1, y1, x2, y2]
            class_name = det['class']
            
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            
            # Frustum Filtering: Select points inside the 2D bounding box
            mask = (projected_points[:, 0] >= x1) & (projected_points[:, 0] <= x2) & \
                   (projected_points[:, 1] >= y1) & (projected_points[:, 1] <= y2)
            
            frustum_points = projected_points[mask]
            
            if len(frustum_points) < 5:
                continue
                
            # Extract 3D points (in Vehicle Frame)
            points_veh = frustum_points[:, 2:5]
            
            # --- Clustering / Outlier Removal ---
            # Simple statistical filtering to remove background/foreground noise
            # Calculate median center of the cluster
            center_median = np.median(points_veh, axis=0)
            
            # Calculate distance of each point to the median center
            dists = np.linalg.norm(points_veh - center_median, axis=1)
            
            # Keep points within a reasonable radius (e.g., 3.0m covers a large car)
            # This removes points that might be far background captured in the frustum
            cluster_mask = dists < 3.0
            points_veh = points_veh[cluster_mask]
            
            if len(points_veh) < 3:
                continue
            
            # --- Oriented Bounding Box via PCA ---
            # Compute an oriented bounding box (OBB) using PCA on the XY plane.
            # Steps:
            #  - perform PCA on the XY points to find principal axes
            #  - project points to PCA frame to get length/width extents
            #  - compute min/max Z for height
            #  - reconstruct OBB center in vehicle frame and compute yaw

            # Extract XY and Z
            xy = points_veh[:, :2]
            z_vals = points_veh[:, 2]

            # Centroid in XY
            centroid_xy = np.mean(xy, axis=0)
            centered = xy - centroid_xy

            # If we have enough points, do PCA
            if centered.shape[0] >= 2:
                cov = np.cov(centered, rowvar=False)
                eigvals, eigvecs = np.linalg.eigh(cov)
                # Order eigenvectors by descending eigenvalue
                order = eigvals.argsort()[::-1]
                eigvecs = eigvecs[:, order]

                # Project into PCA axes
                proj = centered.dot(eigvecs)
                min_proj = np.min(proj, axis=0)
                max_proj = np.max(proj, axis=0)
                extents = max_proj - min_proj

                est_length = float(extents[0]) if extents[0] > 0 else 0.0
                est_width = float(extents[1]) if extents[1] > 0 else 0.0

                # center in PCA frame -> back to vehicle XY
                center_pca = (min_proj + max_proj) / 2.0
                center_xy = centroid_xy + eigvecs.dot(center_pca)

                # Height and Z center
                min_z = np.min(z_vals)
                max_z = np.max(z_vals)
                est_height = float(max_z - min_z)
                center_z = float((min_z + max_z) / 2.0)

                center_x = float(center_xy[0])
                center_y = float(center_xy[1])

                # Yaw: angle of principal axis (first eigenvector) in vehicle XY
                principal = eigvecs[:, 0]
                yaw = math.atan2(float(principal[1]), float(principal[0]))
            else:
                # Fallback to axis-aligned bounding box
                min_x = np.min(points_veh[:, 0])
                max_x = np.max(points_veh[:, 0])
                min_y = np.min(points_veh[:, 1])
                max_y = np.max(points_veh[:, 1])
                min_z = np.min(points_veh[:, 2])
                max_z = np.max(points_veh[:, 2])

                est_length = max_x - min_x
                est_width = max_y - min_y
                est_height = max_z - min_z

                center_x = (min_x + max_x) / 2.0
                center_y = (min_y + max_y) / 2.0
                center_z = (min_z + max_z) / 2.0
                yaw = 0.0

            obstacle = Obstacle(
                id=obstacle_id,
                obstacle_type=class_name,
                x=center_x,
                y=center_y,
                z=center_z,
                vx=0.0,
                vy=0.0,
                vz=0.0,
                length=est_length,
                width=est_width,
                height=est_height,
                yaw=yaw,
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
