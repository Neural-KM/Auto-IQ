"""
Lane Detector
=============

Detects lane boundaries from CARLA semantic segmentation images.
Uses bird's eye view transform for better lane line fitting.
Outputs polynomial fits and lane geometry for path planning.
"""

import numpy as np
import cv2
from sensors.camera_semantic import SemanticLabels
from state.lane import LaneDetection


class LaneDetector:
    """
    Lane detection using semantic segmentation with bird's eye view transform.
    
    Pipeline:
    1. Extract lane markings (RoadLine) from semantic image
    2. Transform to bird's eye view (top-down perspective)
    3. Use sliding window to find lane pixels
    4. Fit polynomial curves to left and right lanes
    5. Return lane geometry and polynomial fits for path planning
    """
    
    def __init__(self, image_width, image_height, camera_fov, camera_pitch, camera_height):
        """
        Initialize lane detector.
        
        Args:
            image_width: Camera image width in pixels
            image_height: Camera image height in pixels
            camera_fov: Camera horizontal field of view in degrees
            camera_pitch: Camera pitch angle in degrees (negative = looking down)
            camera_height: Camera height above ground in meters
        """
        self.width = image_width
        self.height = image_height
        
        # Camera parameters
        self.fov = camera_fov
        self.pitch = camera_pitch  # degrees, negative = looking down
        self.camera_height = camera_height  # meters
        
        # Bird's eye view output size and real-world coverage
        self.bev_width = 400
        self.bev_height = 400
        
        # BEV real-world dimensions (meters)
        # Looking ahead: 5m to 30m from vehicle, 10m lane width total
        self.bev_x_range = (-5.0, 5.0)    # Left/right from vehicle center (meters)
        self.bev_y_range = (5.0, 35.0)    # Forward distance range (meters)
        
        # Compute camera intrinsic matrix
        self._compute_camera_intrinsics()
        
        # Compute perspective transform matrices
        self._compute_perspective_transform()
        
        # Temporal smoothing factor
        self.smoothing = 0.7
        
        # State tracking
        self.left_lane_x = None
        self.right_lane_x = None
        self.lane_center = self.width // 2
        self.lane_width_pixels = None
        self.avg_lane_width = None  # Track average lane width for stability
        
        # Previous polynomial fits for temporal smoothing
        self.prev_left_fit = None
        self.prev_right_fit = None
        
        # Vehicle is at image center
        self.vehicle_x = self.width // 2
        
        # Debug images for visualization
        self.debug_images = {
            'bev_lane_mask': None,
            'bev_road_mask': None,
            'bev_combined': None,
            'lane_fit': None
        }
    
    def _compute_camera_intrinsics(self):
        """Compute camera intrinsic matrix from FOV and image dimensions."""
        # Focal length from horizontal FOV
        # fov = 2 * atan(width / (2 * fx))
        # fx = width / (2 * tan(fov/2))
        fov_rad = np.radians(self.fov)
        self.fx = self.width / (2.0 * np.tan(fov_rad / 2.0))
        self.fy = self.fx  # Square pixels
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0
        
        # Intrinsic matrix
        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)
    
    def _compute_perspective_transform(self):
        """
        Compute perspective transform for bird's eye view using camera geometry.
        
        Uses the pinhole camera model to project ground plane points to image,
        then computes homography from image trapezoid to BEV rectangle.
        """
        pitch_rad = np.radians(self.pitch)
        
        # Define 4 ground plane points in vehicle coordinates (X=right, Y=forward, Z=up)
        # These form a rectangle on the ground that we want to see in BEV
        x_min, x_max = self.bev_x_range
        y_min, y_max = self.bev_y_range
        
        # Ground plane points: (X, Y, Z) where Z=0 (ground level)
        # Vehicle is at origin, camera is at (0, 0, camera_height)
        ground_points_3d = np.array([
            [x_min, y_max, 0],  # Far left
            [x_max, y_max, 0],  # Far right
            [x_max, y_min, 0],  # Near right
            [x_min, y_min, 0],  # Near left
        ], dtype=np.float64)
        
        # Transform to camera coordinates
        # Camera frame: x=right, y=down, z=forward
        # Rotation: pitch around x-axis (positive pitch = looking up)
        R_pitch = np.array([
            [1, 0, 0],
            [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
            [0, np.sin(pitch_rad), np.cos(pitch_rad)]
        ], dtype=np.float64)
        
        # Translation: camera is at height h above ground
        # In camera coords: ground is at y = +h (below camera)
        src_pts = []
        for gp in ground_points_3d:
            # Convert vehicle coords (X_v=right, Y_v=forward, Z_v=up) to 
            # camera coords (X_c=right, Y_c=down, Z_c=forward)
            # Point relative to camera position
            p_vehicle = np.array([gp[0], gp[1], gp[2] - self.camera_height])
            
            # Transform to camera frame
            # X_c = X_v, Y_c = -Z_v, Z_c = Y_v
            p_cam_unrotated = np.array([p_vehicle[0], -p_vehicle[2], p_vehicle[1]])
            
            # Apply pitch rotation
            p_cam = R_pitch @ p_cam_unrotated
            
            # Project to image using pinhole model
            if p_cam[2] > 0:  # Only if in front of camera
                u = self.fx * p_cam[0] / p_cam[2] + self.cx
                v = self.fy * p_cam[1] / p_cam[2] + self.cy
                src_pts.append([u, v])
            else:
                # Point behind camera, use a fallback
                src_pts.append([self.cx, 0])
        
        src_pts = np.float32(src_pts)
        
        # Destination points in BEV image
        # Map ground rectangle to BEV image rectangle
        dst_pts = np.float32([
            [0, 0],                           # Far left -> top-left
            [self.bev_width, 0],              # Far right -> top-right
            [self.bev_width, self.bev_height], # Near right -> bottom-right
            [0, self.bev_height],             # Near left -> bottom-left
        ])
        
        self.M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
        self.src_points = src_pts  # Store for visualization
        
        # Store meters-per-pixel for real-world measurements
        self.meters_per_pixel_x = (x_max - x_min) / self.bev_width
        self.meters_per_pixel_y = (y_max - y_min) / self.bev_height
    
    def to_birds_eye_view(self, image):
        """Transform image to bird's eye view."""
        if image is None:
            return None
        return cv2.warpPerspective(image, self.M, (self.bev_width, self.bev_height),
                                   flags=cv2.INTER_NEAREST)
    
    def detect(self, semantic_image):
        """
        Detect lanes from semantic segmentation image.
        
        Args:
            semantic_image: 2D numpy array of semantic labels (H x W)
            
        Returns:
            dict with detection results including CTE, heading error, confidence
        """
        if semantic_image is None:
            return self._invalid_result()
        
        h, w = semantic_image.shape[:2]
        
        # Extract binary masks
        lane_mask = (semantic_image == SemanticLabels.ROAD_LINE).astype(np.uint8) * 255
        road_mask = (semantic_image == SemanticLabels.ROAD).astype(np.uint8) * 255
        
        # Transform to bird's eye view
        bev_lane = self.to_birds_eye_view(lane_mask)
        bev_road = self.to_birds_eye_view(road_mask)
        
        # Store debug images
        self.debug_images['bev_lane_mask'] = bev_lane
        self.debug_images['bev_road_mask'] = bev_road
        
        # Combined BEV visualization
        bev_combined = np.zeros((self.bev_height, self.bev_width, 3), dtype=np.uint8)
        bev_combined[bev_road > 127] = [80, 40, 80]  # Purple for road
        bev_combined[bev_lane > 127] = [0, 255, 255]  # Yellow for lane markings
        self.debug_images['bev_combined'] = bev_combined
        
        # Fit lane polynomials using sliding window
        left_fit, right_fit, lane_fit_img = self._fit_lanes_sliding_window(bev_lane)
        self.debug_images['lane_fit'] = lane_fit_img
        
        # Apply temporal smoothing to fits
        if left_fit is not None:
            if self.prev_left_fit is not None:
                left_fit = self.smoothing * self.prev_left_fit + (1 - self.smoothing) * left_fit
            self.prev_left_fit = left_fit
        
        if right_fit is not None:
            if self.prev_right_fit is not None:
                right_fit = self.smoothing * self.prev_right_fit + (1 - self.smoothing) * right_fit
            self.prev_right_fit = right_fit
        
        # Calculate outputs
        return self._compute_lane_metrics(left_fit, right_fit, w)
    
    def _fit_lanes_sliding_window(self, bev_lane_mask):
        """
        Fit polynomial curves to lane lines using sliding window search.
        
        Returns:
            (left_fit, right_fit, visualization_image)
        """
        # Visualization image
        vis_img = np.zeros((self.bev_height, self.bev_width, 3), dtype=np.uint8)
        vis_img[bev_lane_mask > 127] = [255, 255, 0]
        
        # Histogram of bottom half to find lane bases
        histogram = np.sum(bev_lane_mask[self.bev_height//2:, :], axis=0)
        
        midpoint = self.bev_width // 2
        left_base = np.argmax(histogram[:midpoint]) if np.max(histogram[:midpoint]) > 100 else None
        right_base = np.argmax(histogram[midpoint:]) + midpoint if np.max(histogram[midpoint:]) > 100 else None
        
        # Sliding window params
        n_windows = 9
        window_height = self.bev_height // n_windows
        margin = 50
        min_pixels = 30
        
        # Get nonzero pixels
        nonzero = bev_lane_mask.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])
        
        left_current = left_base
        right_current = right_base
        left_lane_inds = []
        right_lane_inds = []
        
        # Slide windows
        for window in range(n_windows):
            win_y_low = self.bev_height - (window + 1) * window_height
            win_y_high = self.bev_height - window * window_height
            
            # Left window
            if left_current is not None:
                win_x_low = max(0, left_current - margin)
                win_x_high = min(self.bev_width, left_current + margin)
                
                cv2.rectangle(vis_img, (win_x_low, win_y_low), 
                             (win_x_high, win_y_high), (0, 255, 0), 2)
                
                good_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                            (nonzero_x >= win_x_low) & (nonzero_x < win_x_high)).nonzero()[0]
                left_lane_inds.append(good_inds)
                
                if len(good_inds) > min_pixels:
                    left_current = int(np.mean(nonzero_x[good_inds]))
            
            # Right window
            if right_current is not None:
                win_x_low = max(0, right_current - margin)
                win_x_high = min(self.bev_width, right_current + margin)
                
                cv2.rectangle(vis_img, (win_x_low, win_y_low),
                             (win_x_high, win_y_high), (0, 255, 0), 2)
                
                good_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                            (nonzero_x >= win_x_low) & (nonzero_x < win_x_high)).nonzero()[0]
                right_lane_inds.append(good_inds)
                
                if len(good_inds) > min_pixels:
                    right_current = int(np.mean(nonzero_x[good_inds]))
        
        # Fit polynomials
        left_fit = None
        right_fit = None
        
        if left_lane_inds:
            left_lane_inds = np.concatenate(left_lane_inds)
            if len(left_lane_inds) > 50:
                left_x = nonzero_x[left_lane_inds]
                left_y = nonzero_y[left_lane_inds]
                left_fit = np.polyfit(left_y, left_x, 2)
                
                # Draw fitted curve
                plot_y = np.linspace(0, self.bev_height - 1, self.bev_height)
                plot_x = np.polyval(left_fit, plot_y)
                pts = np.column_stack((plot_x, plot_y)).astype(np.int32)
                cv2.polylines(vis_img, [pts], False, (255, 0, 0), 3)  # Blue
        
        if right_lane_inds:
            right_lane_inds = np.concatenate(right_lane_inds)
            if len(right_lane_inds) > 50:
                right_x = nonzero_x[right_lane_inds]
                right_y = nonzero_y[right_lane_inds]
                right_fit = np.polyfit(right_y, right_x, 2)
                
                plot_y = np.linspace(0, self.bev_height - 1, self.bev_height)
                plot_x = np.polyval(right_fit, plot_y)
                pts = np.column_stack((plot_x, plot_y)).astype(np.int32)
                cv2.polylines(vis_img, [pts], False, (0, 0, 255), 3)  # Red
        
        # Draw lane center
        if left_fit is not None and right_fit is not None:
            plot_y = np.linspace(0, self.bev_height - 1, self.bev_height)
            center_x = (np.polyval(left_fit, plot_y) + np.polyval(right_fit, plot_y)) / 2
            pts = np.column_stack((center_x, plot_y)).astype(np.int32)
            cv2.polylines(vis_img, [pts], False, (0, 255, 255), 2)  # Cyan
        
        # Draw vehicle position (center line)
        cv2.line(vis_img, (self.bev_width // 2, 0), 
                (self.bev_width // 2, self.bev_height), (255, 255, 255), 1)
        
        return left_fit, right_fit, vis_img
    
    def _compute_lane_metrics(self, left_fit, right_fit, orig_width):
        """Compute lane metrics from polynomial fits."""
        confidence = 0.0
        y_eval = self.bev_height - 10  # Near bottom (closest to vehicle)
        
        left_x = None
        right_x = None
        
        if left_fit is not None:
            left_x = int(np.polyval(left_fit, y_eval))
            confidence += 0.4
        
        if right_fit is not None:
            right_x = int(np.polyval(right_fit, y_eval))
            confidence += 0.4
        
        # Use tracked average width if available, else default to 40% of BEV width
        current_lane_width = self.avg_lane_width if self.avg_lane_width is not None else self.bev_width * 0.4
        
        if left_x is not None and right_x is not None:
            lane_width_bev = right_x - left_x
            
            # Update average lane width (with heavy smoothing to reject noise)
            if self.avg_lane_width is None:
                self.avg_lane_width = lane_width_bev
            else:
                self.avg_lane_width = 0.95 * self.avg_lane_width + 0.05 * lane_width_bev
                
            confidence += 0.2
        elif left_x is not None:
            lane_width_bev = current_lane_width
        elif right_x is not None:
            lane_width_bev = current_lane_width
        else:
            lane_width_bev = None
        
        # Map to original image coords (approximate)
        if left_x is not None:
            self.left_lane_x = int((left_x / self.bev_width) * orig_width)
        if right_x is not None:
            self.right_lane_x = int((right_x / self.bev_width) * orig_width)
        if lane_width_bev is not None:
            self.lane_width_pixels = int(lane_width_bev * (orig_width / self.bev_width))
        
        return LaneDetection(
            valid=confidence > 0.2,
            confidence=min(confidence, 1.0),
            lane_center=self.lane_center,
            left_lane=self.left_lane_x,
            right_lane=self.right_lane_x,
            lane_width=self.lane_width_pixels,
            left_fit=left_fit,
            right_fit=right_fit,
            lane_width_bev=lane_width_bev,
            bev_params={
                'width': self.bev_width,
                'height': self.bev_height,
                'x_range': self.bev_x_range,
                'y_range': self.bev_y_range
            }
        )
    
    def _invalid_result(self):
        """Return result when detection fails."""
        return LaneDetection(
            valid=False,
            confidence=0.0,
            lane_center=self.width // 2,
            left_lane=None,
            right_lane=None,
            lane_width=None,
            left_fit=None,
            right_fit=None
        )
    
    def reset(self):
        """Reset detector state."""
        self.left_lane_x = None
        self.right_lane_x = None
        self.lane_center = self.width // 2
        self.lane_width_pixels = None
        self.prev_left_fit = None
        self.prev_right_fit = None
        self.avg_lane_width = None
    
    def get_debug_images(self):
        """
        Get intermediate debug images for visualization.
        
        Returns:
            dict with:
                - 'bev_lane_mask': Binary lane markings in bird's eye view
                - 'bev_road_mask': Binary road in bird's eye view  
                - 'bev_combined': Combined colored BEV
                - 'lane_fit': BEV with sliding windows and fitted polynomials
        """
        return self.debug_images
    
    def get_source_points(self):
        """Get the perspective transform source points for visualization."""
        return self.src_points


class LaneDetectorDebugWindow:
    """
    Separate OpenCV window for visualizing lane detection intermediate images.
    
    Shows a 2x2 grid of:
    - Original semantic (with ROI trapezoid)
    - BEV combined (road + lane markings)
    - BEV lane mask
    - Lane fit visualization (sliding windows + polynomials)
    """
    
    def __init__(self, window_name):
        """
        Initialize debug window.
        
        Args:
            window_name: Name of the OpenCV window
        """
        self.window_name = window_name
        self.window_created = False
        self.panel_size = 400  # Each panel is 400x400
    
    def show(self, lane_detector, semantic_image=None, detection_result=None):
        """
        Update and show the debug window.
        
        Args:
            lane_detector: LaneDetector instance (to get debug images)
            semantic_image: Original semantic segmentation image (optional)
            detection_result: Result from lane_detector.detect() (optional)
        """
        debug = lane_detector.get_debug_images()
        
        # Create 2x2 grid
        grid = np.zeros((self.panel_size * 2, self.panel_size * 2, 3), dtype=np.uint8)
        
        # Panel 1 (top-left): Original semantic with ROI overlay
        if semantic_image is not None:
            panel1 = self._create_semantic_panel(semantic_image, lane_detector)
        else:
            panel1 = self._create_placeholder("Semantic Image")
        grid[0:self.panel_size, 0:self.panel_size] = panel1
        
        # Panel 2 (top-right): BEV combined
        if debug['bev_combined'] is not None:
            panel2 = debug['bev_combined'].copy()
            self._add_label(panel2, "Bird's Eye View")
        else:
            panel2 = self._create_placeholder("BEV Combined")
        grid[0:self.panel_size, self.panel_size:self.panel_size*2] = panel2
        
        # Panel 3 (bottom-left): BEV lane mask
        if debug['bev_lane_mask'] is not None:
            panel3 = cv2.cvtColor(debug['bev_lane_mask'], cv2.COLOR_GRAY2BGR)
            self._add_label(panel3, "Lane Mask (BEV)")
        else:
            panel3 = self._create_placeholder("Lane Mask")
        grid[self.panel_size:self.panel_size*2, 0:self.panel_size] = panel3
        
        # Panel 4 (bottom-right): Lane fit visualization
        if debug['lane_fit'] is not None:
            panel4 = debug['lane_fit'].copy()
            self._add_label(panel4, "Lane Fitting")
            # Add detection info
            if detection_result is not None:
                self._add_detection_info(panel4, detection_result)
        else:
            panel4 = self._create_placeholder("Lane Fit")
        grid[self.panel_size:self.panel_size*2, self.panel_size:self.panel_size*2] = panel4
        
        # Create window if not exists
        if not self.window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.panel_size * 2, self.panel_size * 2)
            self.window_created = True
        
        # Show the grid
        cv2.imshow(self.window_name, grid)
    
    def _create_semantic_panel(self, semantic_image, lane_detector):
        """Create colored semantic image with ROI trapezoid overlay."""
        h, w = semantic_image.shape[:2]
        
        # Color the semantic labels
        panel = np.zeros((h, w, 3), dtype=np.uint8)
        panel[semantic_image == SemanticLabels.ROAD] = [128, 64, 128]
        panel[semantic_image == SemanticLabels.ROAD_LINE] = [50, 234, 157]
        panel[semantic_image == SemanticLabels.SIDEWALK] = [232, 35, 244]
        panel[semantic_image == SemanticLabels.VEHICLE] = [142, 0, 0]
        panel[semantic_image == SemanticLabels.VEGETATION] = [35, 142, 107]
        panel[semantic_image == SemanticLabels.SKY] = [180, 130, 70]
        
        # Draw ROI trapezoid
        src_pts = lane_detector.get_source_points().astype(np.int32)
        cv2.polylines(panel, [src_pts], True, (0, 255, 0), 2)
        
        # Resize to panel size
        panel = cv2.resize(panel, (self.panel_size, self.panel_size))
        self._add_label(panel, "Semantic + ROI")
        
        return panel
    
    def _create_placeholder(self, text):
        """Create a placeholder panel with text."""
        panel = np.zeros((self.panel_size, self.panel_size, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)
        cv2.putText(panel, text, (50, self.panel_size // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
        return panel
    
    def _add_label(self, panel, text):
        """Add label text to top of panel."""
        cv2.rectangle(panel, (0, 0), (len(text) * 12 + 10, 25), (0, 0, 0), -1)
        cv2.putText(panel, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _add_detection_info(self, panel, result):
        """Add detection metrics to panel."""
        y = 50
        info_lines = [
            f"Confidence: {result.confidence:.2f}",
            f"Valid: {result.valid}"
        ]
        
        for line in info_lines:
            cv2.putText(panel, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, (255, 255, 255), 1)
            y += 18
    
    def close(self):
        """Close the debug window."""
        if self.window_created:
            cv2.destroyWindow(self.window_name)
            self.window_created = False
    
    @staticmethod
    def wait_key(delay=1):
        """
        Wait for key press. Call this in your main loop.
        
        Args:
            delay: Wait time in ms (1 = minimal wait, 0 = wait indefinitely)
            
        Returns:
            Key code pressed, or -1 if no key
        """
        return cv2.waitKey(delay)
