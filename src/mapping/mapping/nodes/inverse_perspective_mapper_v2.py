#!/usr/bin/env python3
"""
Inverse Perspective Mapper V2 - Reverse Projection
===================================================

ROS2 node that generates a 2D occupancy grid from semantically segmented
camera images using REVERSE Inverse Perspective Mapping.

Forward projection (pixel-to-grid) creates holes and aliasing at distance.
Reverse projection (grid-to-pixel) guarantees dense, hole-free maps by
iterating over every grid cell and sampling the camera image.

Algorithm:
1. For each grid cell (r, c), compute world coordinates (x, y, z=0)
2. Transform world point to camera frame using base_link->camera TF
3. Project 3D camera point to 2D pixel using intrinsics
4. Sample semantic image at that pixel
5. Mark grid cell based on semantic label

Subscribes:
- Semantic segmentation image (sensor_msgs/Image)
- Camera intrinsics (sensor_msgs/CameraInfo)

Publishes:
- Occupancy grid (nav_msgs/OccupancyGrid)
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import OccupancyGrid
import tf2_ros
from tf2_ros import TransformException
from scipy import ndimage
import time

from cv_bridge import CvBridge

# Import semantic labels from perception module (CARLA convention)
from perception.data_classes.semantic_labels import SEMANTIC_BY_NAME, colorize_mask
from mapping.utils.ipm_funcs import (
    reverse_project_grid, 
    apply_column_drop_skirt)
from mapping.utils.geometry import mark_ego_footprint, mark_circular_clear_zone
# ================= CONFIGURATION =================
ROAD_LABEL_ID = SEMANTIC_BY_NAME['road']           # 1 in CARLA
ROAD_LINE_LABEL_ID = SEMANTIC_BY_NAME['road_line'] # 24 in CARLA

VEHICLE_LABEL_IDS = {SEMANTIC_BY_NAME['vehicle'],
                     SEMANTIC_BY_NAME['rider'],
                     SEMANTIC_BY_NAME['truck'],
                     SEMANTIC_BY_NAME['bus']}
# Grid cell values
UNKNOWN = -1
FREE = 0
OCCUPIED = 100

class InversePerspectiveMapperNode2(Node):
    """
    ROS2 Node for generating occupancy grid using Reverse Projection IPM.
    
    Unlike forward projection (pixel-to-grid), reverse projection (grid-to-pixel)
    iterates over every grid cell and samples the camera image, guaranteeing
    a dense, hole-free occupancy map.
    Problems: 
    1) The road below a car is seen as drivable because 
        semantics and reverse IPM are doing "too well".
        Handling this by vertically diluting the obstacle before IPM
    2) Marking the space blocked by hood as free which should be actually be handled by a another camera in the front

    """
    
    def __init__(self):
        super().__init__('inverse_perspective_mapper_node_v2')
        
        # ==================== Parameters ====================
        self.declare_parameter('semantic_source', 'fused')  # 'fused' or 'ground_truth'
        self.declare_parameter('costmap_topic', '/mapping/ipm_costmap')
        self.declare_parameter('base_frame', 'ego_vehicle')
        
        # Grid parameters
        self.declare_parameter('grid_size', 40.0)    # meters (square grid)
        self.declare_parameter('resolution', 0.4)   # meters per cell
        
        # Morphology parameters (in cells)
        self.declare_parameter('morphology_iterations', 2)  # Dilation then equal erosion (closing)
        
        # Ego vehicle footprint parameters (meters)
        self.declare_parameter('vehicle_length', 3)
        self.declare_parameter('vehicle_width', 1.5)
        self.declare_parameter('rear_to_base_link', 1.7)  # Distance from base_link to vehicle center
        self.declare_parameter('clear_zone_radius', 5.0)  # Circular free zone around vehicle (meters)
        
        # Resolve semantic source → only the image topic changes
        # Camera intrinsics and frame are shared (same physical camera)
        semantic_source = self.get_parameter('semantic_source').value
        camera_info_topic = '/carla/ego_vehicle/sensor/camera/rgb/front/camera_info'
        self.camera_frame = 'ego_vehicle/sensor/camera/rgb/front'
        
        if semantic_source == 'ground_truth':
            semantic_topic = '/carla/ego_vehicle/ground_truth/camera/semantic/front/image'
        else:
            semantic_topic = '/perception/semantic_map/fused'
        
        costmap_topic = self.get_parameter('costmap_topic').value
        self.base_frame = self.get_parameter('base_frame').value
        
        self.grid_size = self.get_parameter('grid_size').value
        self.resolution = self.get_parameter('resolution').value
        self.morphology_iterations = self.get_parameter('morphology_iterations').value
        self.vehicle_length = self.get_parameter('vehicle_length').value
        self.vehicle_width = self.get_parameter('vehicle_width').value
        self.rear_to_base_link = self.get_parameter('rear_to_base_link').value
        self.clear_zone_radius = self.get_parameter('clear_zone_radius').value
        
        # Derive grid cells from size and resolution
        self.grid_cells = int(self.grid_size / self.resolution)
        
        # Grid origin in base_link frame (centered on vehicle)
        self.origin_x = -self.grid_size / 2.0
        self.origin_y = -self.grid_size / 2.0
        
        # ==================== TF2 ====================
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=1.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # ==================== State ====================
        self.K = None  # Camera intrinsics matrix [3x3]
        self.img_height = None
        self.img_width = None
        
        # Debug counters
        self._tf_log_count = 0
        self._img_log_count = 0
        
        # ==================== CV Bridge ====================
        self.cv_bridge = CvBridge()
        
        # ==================== QoS ====================
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # ==================== Subscribers ====================
        self.camera_info_sub = self.create_subscription(
            CameraInfo, camera_info_topic,
            self.camera_info_callback, sensor_qos)
        
        self.semantic_sub = self.create_subscription(
            Image, semantic_topic,
            self.semantic_callback, sensor_qos)
        
        # ==================== Publisher ====================
        self.costmap_pub = self.create_publisher(OccupancyGrid, costmap_topic, 1)
        self.debug_image_pub = self.create_publisher(Image, '/mapping/ipm_debug_image', 1)
        self.get_logger().info('Inverse Perspective Mapper V2 (Reverse Projection) started')
        self.get_logger().info(f'  Semantic Image: {semantic_topic}')
        self.get_logger().info(f'  Camera Info: {camera_info_topic}')
        self.get_logger().info(f'  Output: {costmap_topic}')
        self.get_logger().info(f'  Base Frame: {self.base_frame}')
        self.get_logger().info(f'  Camera Frame: {self.camera_frame}')
        self.get_logger().info(f'  Grid: {self.grid_cells}x{self.grid_cells} cells, {self.grid_size}m, {self.resolution}m/cell')
        self.get_logger().info(f'  Morphology: closing with {self.morphology_iterations} iterations')
        self.get_logger().info(f'  Road labels: road={ROAD_LABEL_ID}, road_line={ROAD_LINE_LABEL_ID}')

    def camera_info_callback(self, msg: CameraInfo):
        """Store camera intrinsics from CameraInfo (called once)."""
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.img_height = msg.height
            self.img_width = msg.width
            self.get_logger().info(f'CameraInfo received. Image: {self.img_width}x{self.img_height}')
            self.get_logger().info(f'  Intrinsics: fx={self.K[0,0]:.1f}, fy={self.K[1,1]:.1f}, '
                                   f'cx={self.K[0,2]:.1f}, cy={self.K[1,2]:.1f}')
            # Unsubscribe after getting info once
            self.destroy_subscription(self.camera_info_sub)

    def _get_base_to_camera_transform(self, stamp):
        """
        Get transform from base_link to camera_link frame.
        
        For reverse projection, we need base->camera (inverse of usual camera->base).
        
        Returns:
            Tuple of (R, t) or (None, None) if lookup fails
        """
        try:
            # Lookup: base_link -> camera (we want to transform points FROM base TO camera)
            transform = self.tf_buffer.lookup_transform(
                self.camera_frame,  # Target frame
                self.base_frame,    # Source frame
                stamp
            )
            
            # Extract Translation
            t = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ], dtype=np.float64)
            
            # Extract Rotation (Quaternion -> Matrix)
            q = transform.transform.rotation
            x, y, z, w = q.x, q.y, q.z, q.w
            
            R = np.array([
                [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
                [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
                [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
            ], dtype=np.float64)
            
            # Debug: Log transform for first few frames
            self._tf_log_count += 1
            if self._tf_log_count <= 3:
                self.get_logger().info(f'TF base->cam: t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]')
                self.get_logger().info(f'  R row0 (cam X in base): [{R[0,0]:.3f}, {R[0,1]:.3f}, {R[0,2]:.3f}]')
                self.get_logger().info(f'  R row1 (cam Y in base): [{R[1,0]:.3f}, {R[1,1]:.3f}, {R[1,2]:.3f}]')
                self.get_logger().info(f'  R row2 (cam Z in base): [{R[2,0]:.3f}, {R[2,1]:.3f}, {R[2,2]:.3f}]')
            
            return R, t
            
        except TransformException as e:
            self.get_logger().warn(f'TF lookup failed: {e}')
            return None, None

    def semantic_callback(self, msg: Image):
        """Process semantic segmentation image and generate occupancy grid."""
        t_start = time.perf_counter()
        
        if self.K is None:
            self.get_logger().warn('No camera info yet, skipping frame')
            return
        
        # 1. Get Transform: base_link -> camera_link
        R, t = self._get_base_to_camera_transform(msg.header.stamp)
        if R is None:
            return
        
        # 2. Parse Semantic Image
        height = msg.height
        width = msg.width
        
        try:
            semantic_img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge conversion failed: {e}')
            return
        
        # Debug: Log unique labels and road pixel count
        self._img_log_count += 1
        if self._img_log_count <= 3:
            unique = np.unique(semantic_img)
            road_count = np.sum((semantic_img == ROAD_LABEL_ID) | (semantic_img == ROAD_LINE_LABEL_ID))
            self.get_logger().info(f'Image: {width}x{height}, encoding: {msg.encoding}')
            self.get_logger().info(f'  Unique labels: {unique[:15]}...')
            self.get_logger().info(f'  Road pixels: {road_count}')
        
        # 3. Extract camera intrinsics
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        
        # 4. Apply column drop skirt to semantic image to avoid detecting drivable area under the car
        hood_mask_rows = 135  # Rows to mask at bottom of image (camera+position-specific)
        semantic_img = apply_column_drop_skirt(semantic_img, obstacle_ids=list(VEHICLE_LABEL_IDS), drop_pixels=5, hood_mask_rows=hood_mask_rows)
        
        # Create debug image after drop skirt
        colorized_dropped = colorize_mask(semantic_img)
        colorized_dropped[-hood_mask_rows:, :] = [128, 128, 128]  # Gray out hood-masked area
        debug_msg = self.cv_bridge.cv2_to_imgmsg(colorized_dropped, encoding='rgb8')
        debug_msg.header = msg.header
        self.debug_image_pub.publish(debug_msg)
        # 5. Run Reverse Projection (Numba JIT with parallel)
        grid = reverse_project_grid(
            self.grid_cells,
            self.grid_cells,
            self.origin_x,
            self.origin_y,
            self.resolution,
            R,
            t,
            fx, fy, cx, cy,
            height, width,
            semantic_img,
            ROAD_LABEL_ID,
            ROAD_LINE_LABEL_ID,
            UNKNOWN,
            FREE,
            OCCUPIED
        )
        
        # 5. Apply morphological operations on obstacles
        grid = self._apply_morphology(grid)
        
        # # 6. Mark ego vehicle footprint as FREE
        # grid = mark_ego_footprint(
        #     grid=grid,
        #     resolution=self.resolution,
        #     vehicle_length=self.vehicle_length,
        #     vehicle_width=self.vehicle_width,
        #     rear_to_base=self.rear_to_base_link,
        #     FREE=FREE
        # )
        
        # 7. Mark circular clear zone around vehicle
        if self.clear_zone_radius > 0:
            grid = mark_circular_clear_zone(
                grid=grid,
                radius=self.clear_zone_radius,
                resolution=self.resolution,
                FREE=FREE
            )
        
        # 8. Publish OccupancyGrid
        self._publish_grid(grid=grid, stamp=msg.header.stamp)
        
        t_end = time.perf_counter()
        
        # Log stats
        n_free = np.sum(grid == FREE)
        n_occupied = np.sum(grid == OCCUPIED)
        n_unknown = np.sum(grid == UNKNOWN)
        self.get_logger().info(
            f'IPM V2: {(t_end - t_start)*1000:.1f}ms | '
            f'Free: {n_free}, Occupied: {n_occupied}, Unknown: {n_unknown}', throttle_duration_sec=5.0
        )

    def _apply_morphology(self, grid: np.ndarray) -> np.ndarray:
        """
        Apply morphological closing (dilation + erosion) to obstacle layer.
        
        Closing fills small holes and smooths boundaries without changing
        the overall obstacle size.
        """
        if self.morphology_iterations == 0:
            return grid
        
        # Create binary obstacle mask
        obstacle_mask = (grid == OCCUPIED)
        
        # Morphological closing: dilate then erode with same iterations
        struct = ndimage.generate_binary_structure(2, 1)  # 4-connected
        obstacle_mask = ndimage.binary_dilation(
            obstacle_mask, structure=struct, iterations=self.morphology_iterations
        )
        obstacle_mask = ndimage.binary_erosion(
            obstacle_mask, structure=struct, iterations=self.morphology_iterations
        )
        
        # Apply back to grid
        result = grid.copy()
        result[obstacle_mask & (grid == FREE)] = OCCUPIED
        
        return result

    def _publish_grid(self, grid: np.ndarray, stamp):
        """Publish occupancy grid message in base_link frame."""
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = self.base_frame  # Grid centered on vehicle
        
        msg.info.resolution = float(self.resolution)
        msg.info.width = self.grid_cells
        msg.info.height = self.grid_cells
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0  # No rotation
        
        msg.data = grid.flatten().tolist()
        self.costmap_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = InversePerspectiveMapperNode2()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
