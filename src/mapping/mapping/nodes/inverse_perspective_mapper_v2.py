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
from numba import jit, prange
import time

# Import semantic labels from perception module (CARLA convention)
from perception.data_classes.semantic_labels import SEMANTIC_BY_NAME

# ================= CONFIGURATION =================
ROAD_LABEL_ID = SEMANTIC_BY_NAME['road']           # 1 in CARLA
ROAD_LINE_LABEL_ID = SEMANTIC_BY_NAME['road_line'] # 24 in CARLA

# Grid cell values
UNKNOWN = -1
FREE = 0
OCCUPIED = 100


@jit(nopython=True, parallel=True, cache=True)
def reverse_project_grid(
    grid_rows: int,
    grid_cols: int,
    origin_x: float,
    origin_y: float,
    resolution: float,
    R_base2cam: np.ndarray,
    t_base2cam: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_height: int,
    img_width: int,
    semantic_img: np.ndarray,
    road_label: int,
    road_line_label: int
) -> np.ndarray:
    """
    Reverse projection: iterate over grid cells and sample camera image.
    
    CARLA camera_link frame: X=forward, Y=left, Z=up
    Optical frame (for pinhole): X=right, Y=down, Z=forward
    
    Conversion: optical = [link_Y negated, link_Z negated, link_X]
                optical_x = -link_y
                optical_y = -link_z  
                optical_z = link_x
    
    For each grid cell:
    1. Grid to Base: Convert (r, c) to vehicle coordinates
    2. Base to Camera Link: Transform point to camera_link frame
    3. Camera Link to Optical: Convert frame conventions
    4. Check Z: If Z_optical <= 0, point is behind camera -> UNKNOWN
    5. Project to Pixel: Use intrinsics to get (u, v)
    6. Bounds Check & Sample: If in bounds, sample semantic image
    
    Args:
        grid_rows: Number of rows in grid
        grid_cols: Number of columns in grid
        origin_x: Grid origin X in base frame (meters)
        origin_y: Grid origin Y in base frame (meters)
        resolution: Grid resolution (meters per cell)
        R_base2cam: Rotation matrix from base_link to camera_link [3x3]
        t_base2cam: Translation vector from base_link to camera_link [3]
        fx, fy: Focal lengths
        cx, cy: Principal point
        img_height, img_width: Image dimensions
        semantic_img: Semantic label image [H, W]
        road_label: Label ID for road
        road_line_label: Label ID for road line
        
    Returns:
        grid: Occupancy grid [grid_rows, grid_cols] with values UNKNOWN/FREE/OCCUPIED
    """
    grid = np.full((grid_rows, grid_cols), UNKNOWN, dtype=np.int8)
    
    # Use prange for parallel execution over rows
    for r in prange(grid_rows):
        for c in range(grid_cols):
            # 1. Grid to Base Frame
            # Grid cell (r, c) -> base_link coordinates (x, y, z=0)
            # c is along X (forward), r is along Y (left)
            x_base = origin_x + c * resolution
            y_base = origin_y + r * resolution
            z_base = 0.0  # Ground plane
            
            # 2. Base to Camera Link Frame
            # P_link = R * P_base + t
            x_link = R_base2cam[0, 0] * x_base + R_base2cam[0, 1] * y_base + R_base2cam[0, 2] * z_base + t_base2cam[0]
            y_link = R_base2cam[1, 0] * x_base + R_base2cam[1, 1] * y_base + R_base2cam[1, 2] * z_base + t_base2cam[1]
            z_link = R_base2cam[2, 0] * x_base + R_base2cam[2, 1] * y_base + R_base2cam[2, 2] * z_base + t_base2cam[2]
            
            # 3. Camera Link to Optical Frame conversion
            # Link: X=forward, Y=left, Z=up
            # Optical: X=right, Y=down, Z=forward
            # optical_x = -link_y (right = -left)
            # optical_y = -link_z (down = -up)
            # optical_z = link_x  (forward = forward)
            x_optical = -y_link
            y_optical = -z_link
            z_optical = x_link
            
            # 4. Check Z: Point must be in front of camera (Z_optical > 0)
            if z_optical <= 0.0:
                # Point is behind camera
                continue  # Already UNKNOWN
            
            # 5. Project to Pixel using pinhole model
            # u = fx * (x/z) + cx
            # v = fy * (y/z) + cy
            u = fx * (x_optical / z_optical) + cx
            v = fy * (y_optical / z_optical) + cy
            
            # 6. Bounds Check
            u_int = int(u + 0.5)  # Round to nearest
            v_int = int(v + 0.5)
            
            if u_int < 0 or u_int >= img_width or v_int < 0 or v_int >= img_height:
                # Outside image bounds
                continue  # Already UNKNOWN
            
            # 7. Sample semantic label
            label = semantic_img[v_int, u_int]
            
            if label == road_label or label == road_line_label:
                grid[r, c] = FREE
            else:
                grid[r, c] = OCCUPIED
    
    return grid


class InversePerspectiveMapperNode2(Node):
    """
    ROS2 Node for generating occupancy grid using Reverse Projection IPM.
    
    Unlike forward projection (pixel-to-grid), reverse projection (grid-to-pixel)
    iterates over every grid cell and samples the camera image, guaranteeing
    a dense, hole-free occupancy map.
    """
    
    def __init__(self):
        super().__init__('inverse_perspective_mapper_node_v2')
        
        # ==================== Parameters ====================
        self.declare_parameter('semantic_topic', '/carla/ego_vehicle/ground_truth/camera/semantic/front/image')
        self.declare_parameter('camera_info_topic', '/carla/ego_vehicle/ground_truth/camera/semantic/front/camera_info')
        self.declare_parameter('costmap_topic', '/mapping/ipm_costmap')
        self.declare_parameter('base_frame', 'ego_vehicle')
        self.declare_parameter('camera_frame', 'ego_vehicle/ground_truth/camera/semantic/front')
        
        # Grid parameters
        self.declare_parameter('grid_size', 40.0)    # meters (square grid)
        self.declare_parameter('resolution', 0.4)   # meters per cell
        
        # Get parameter values
        semantic_topic = self.get_parameter('semantic_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        costmap_topic = self.get_parameter('costmap_topic').value
        self.base_frame = self.get_parameter('base_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        
        self.grid_size = self.get_parameter('grid_size').value
        self.resolution = self.get_parameter('resolution').value
        
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
        
        self.get_logger().info('Inverse Perspective Mapper V2 (Reverse Projection) started')
        self.get_logger().info(f'  Semantic Image: {semantic_topic}')
        self.get_logger().info(f'  Camera Info: {camera_info_topic}')
        self.get_logger().info(f'  Output: {costmap_topic}')
        self.get_logger().info(f'  Base Frame: {self.base_frame}')
        self.get_logger().info(f'  Camera Frame: {self.camera_frame}')
        self.get_logger().info(f'  Grid: {self.grid_cells}x{self.grid_cells} cells, {self.grid_size}m, {self.resolution}m/cell')
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
        
        if msg.encoding == 'mono8':
            semantic_img = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width)
        else:
            self.get_logger().warn(f'Unsupported semantic encoding: {msg.encoding} Expecting mono8.')
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
        
        # 4. Run Reverse Projection (Numba JIT with parallel)
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
            ROAD_LINE_LABEL_ID
        )
        
        # 5. Publish OccupancyGrid
        self._publish_grid(grid, msg.header.stamp)
        
        t_end = time.perf_counter()
        
        # Log stats
        n_free = np.sum(grid == FREE)
        n_occupied = np.sum(grid == OCCUPIED)
        n_unknown = np.sum(grid == UNKNOWN)
        self.get_logger().info(
            f'IPM V2: {(t_end - t_start)*1000:.1f}ms | '
            f'Free: {n_free}, Occupied: {n_occupied}, Unknown: {n_unknown}'
        )

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
