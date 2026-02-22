#!/usr/bin/env python3
"""
Inverse Perspective Mapper (IPM) Costmap Node
==============================================

ROS2 node that generates a 2D occupancy grid from semantically segmented
camera images using Inverse Perspective Mapping.

Key Features:
- Projects semantic image pixels to ground plane (Z=0)
- Uses camera intrinsics and extrinsics for accurate projection
- Marks road/road_line as FREE, everything else as OCCUPIED
- Grid is centered on vehicle in base_link frame

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
import time

from cv_bridge import CvBridge
# Import semantic labels from perception module (CARLA convention)
from perception.data_classes.semantic_labels import SEMANTIC_BY_NAME
from mapping.utils.ipm_funcs import (
    project_pixels_to_ground,
    fill_grid_from_points)
# ================= CONFIGURATION =================
# Drivable area - only road and road_line are free space
ROAD_LABEL_ID = SEMANTIC_BY_NAME['road']           # 1 in CARLA
ROAD_LINE_LABEL_ID = SEMANTIC_BY_NAME['road_line'] # 24 in CARLA

# Grid cell values
UNKNOWN = -1
FREE = 0
OCCUPIED = 100

class InversePerspectiveMapperNode(Node):
    """
    ROS2 Node for generating occupancy grid from semantic camera images.
    
    Uses Inverse Perspective Mapping to project road pixels to ground plane.
    Grid is centered on vehicle in base_link frame.

    Problems: Projection logic with a z plane fails if let's say a car is climbing up a exit. Need to 
    rather find projections onto a plane defined by car's attitude and position
    """
    
    def __init__(self):
        super().__init__('inverse_perspective_mapper_node')
        
        # ==================== Parameters ====================
        self.declare_parameter('semantic_topic', '/carla/ego_vehicle/ground_truth/camera/semantic/front/image')
        self.declare_parameter('camera_info_topic', '/carla/ego_vehicle/ground_truth/camera/semantic/front/camera_info')
        self.declare_parameter('costmap_topic', '/mapping/ipm_costmap')
        self.declare_parameter('base_frame', 'ego_vehicle')  # Vehicle footprint frame
        self.declare_parameter('map_frame', 'map')           # World-fixed frame for grid alignment
        
        # Grid parameters
        self.declare_parameter('grid_size', 40.0)   # meters (square grid)
        self.declare_parameter('resolution', 0.4)   # meters per cell
        
        # Get parameter values
        semantic_topic = self.get_parameter('semantic_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        costmap_topic = self.get_parameter('costmap_topic').value
        self.base_frame = self.get_parameter('base_frame').value
        self.map_frame = self.get_parameter('map_frame').value
        
        self.grid_size = self.get_parameter('grid_size').value
        self.resolution = self.get_parameter('resolution').value
        
        # Derive grid cells from size and resolution
        self.grid_cells = int(self.grid_size / self.resolution)
        
        # ==================== TF2 ====================
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=1.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # ==================== State ====================
        self.K = None  # Camera intrinsics matrix [3x3]
        self.camera_frame = None
        
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
        
        self.get_logger().info('Inverse Perspective Mapper Node started')
        self.get_logger().info(f'  Semantic Image: {semantic_topic}')
        self.get_logger().info(f'  Camera Info: {camera_info_topic}')
        self.get_logger().info(f'  Output: {costmap_topic}')
        self.get_logger().info(f'  Base Frame: {self.base_frame}')
        self.get_logger().info(f'  Map Frame: {self.map_frame} (grid aligned to this)')
        self.get_logger().info(f'  Grid: {self.grid_cells}x{self.grid_cells} cells, {self.grid_size}m, {self.resolution}m/cell')
        self.get_logger().info(f'  Road labels: road={ROAD_LABEL_ID}, road_line={ROAD_LINE_LABEL_ID}')

    def camera_info_callback(self, msg: CameraInfo):
        """Store camera intrinsics from CameraInfo (called once)."""
        if self.K is None:
            # msg.k is flattened [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            self.K = np.array(msg.k).reshape(3, 3)
            self.camera_frame = msg.header.frame_id
            self.get_logger().info(f'CameraInfo received. Frame: {self.camera_frame}')
            self.get_logger().info(f'  Intrinsics: fx={self.K[0,0]:.1f}, fy={self.K[1,1]:.1f}, '
                                   f'cx={self.K[0,2]:.1f}, cy={self.K[1,2]:.1f}')
            # Unsubscribe after getting info once (intrinsics are constant)
            self.destroy_subscription(self.camera_info_sub)

    def _get_extrinsics(self, stamp):
        """
        Get camera-to-map transform at exact timestamp.
        
        Projects to map frame so the grid stays world-aligned (no yaw rotation).
        
        Returns:
            Tuple of (R, t) or (None, None) if lookup fails
        """
        try:            
            # Lookup transform from Camera -> Map (world-fixed frame)
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.camera_frame,
                stamp
            )
            
            # Extract Translation
            t = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ], dtype=np.float32)
            
            # Extract Rotation (Quaternion -> Matrix)
            q = transform.transform.rotation
            x, y, z, w = q.x, q.y, q.z, q.w
            
            # Standard quaternion to rotation matrix
            R = np.array([
                [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
                [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
                [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
            ], dtype=np.float32)
            
            return R, t
            
        except TransformException as e:
            self.get_logger().warn(f'TF lookup failed: {e}')
            return None, None
    
    def _get_vehicle_position(self, stamp):
        """
        Get vehicle position in map frame at exact timestamp.
        
        Used to center the grid on the vehicle and get ground Z.
        
        Returns:
            Tuple of (x, y, z) or None if lookup fails
        """
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                stamp
            )
            return (
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            )
        except TransformException as e:
            self.get_logger().warn(f'Vehicle pose lookup failed: {e}')
            return None

    def semantic_callback(self, msg: Image):
        """Process semantic segmentation image and generate occupancy grid."""
        self.get_logger().info('Semantic image received')
        t_start_semantic_cb = time.perf_counter()
        if self.K is None:
            return
        
        # 1. Get Extrinsics for this specific frame time
        R, t = self._get_extrinsics(msg.header.stamp)
        if R is None:
            return
        
        # DEBUG: Log camera transform
        self.get_logger().info(f'Camera transform: t=[{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}]')
        self.get_logger().info(f'Camera R row2 (Z-axis in base): [{R[2,0]:.3f}, {R[2,1]:.3f}, {R[2,2]:.3f}]')
        
        # 2. Parse Image
        
        height = msg.height
        width = msg.width
        self.get_logger().info(f'Image size: {width}x{height}, encoding: {msg.encoding}')
        
        try:
            label_img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge conversion failed: {e}')
            return
           

        # DEBUG: Log unique labels in the image to verify semantic encoding
        unique_labels = np.unique(label_img)
        self.get_logger().info(f'Image encoding: {msg.encoding}, shape: {label_img.shape}')
        self.get_logger().info(f'Unique labels in image: {unique_labels[:20]}...')  # First 20
        self.get_logger().info(f'Looking for road={ROAD_LABEL_ID}, road_line={ROAD_LINE_LABEL_ID}')

        # 3. Filter Road Pixels
        # Get coordinates of all pixels that match ROAD_LABEL_ID or ROAD_LINE_LABEL_ID
        # v_arr (rows), u_arr (cols)
        road_mask = (label_img == ROAD_LABEL_ID) | (label_img == ROAD_LINE_LABEL_ID)
        v_arr, u_arr = np.where(road_mask)
        
        self.get_logger().info(f'Found {len(u_arr)} road pixels')
        
        if len(u_arr) == 0:
            # No road pixels found - publish empty grid
            self.get_logger().warn('No road pixels found in image')
            self._publish_empty_grid(msg.header.stamp)
            return
        
        # Get vehicle position in map frame to center grid and get ground Z
        vehicle_pos = self._get_vehicle_position(msg.header.stamp)
        if vehicle_pos is None:
            return
        vehicle_x, vehicle_y, ground_z = vehicle_pos
        
        # Grid origin in map frame (centered on vehicle)
        origin_x = vehicle_x - self.grid_size / 2.0
        origin_y = vehicle_y - self.grid_size / 2.0

        # 4. Project to Ground (Numba JIT)
        # Use vehicle Z as ground plane (ground in sim wont be at Z=0 always)
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        
        gx, gy, valid_mask = project_pixels_to_ground(
            u_arr.astype(np.float32),
            v_arr.astype(np.float32),
            fx, fy, cx, cy, R, t, ground_z
        )
        
        # DEBUG: Check projection results
        n_valid = np.sum(valid_mask)
        self.get_logger().info(f'Valid projections: {n_valid}/{len(u_arr)}')
        if n_valid > 0:
            valid_gx = gx[valid_mask]
            valid_gy = gy[valid_mask]
            self.get_logger().info(f'Projection X range: [{valid_gx.min():.2f}, {valid_gx.max():.2f}]')
            self.get_logger().info(f'Projection Y range: [{valid_gy.min():.2f}, {valid_gy.max():.2f}]')
            self.get_logger().info(f'Grid origin: ({origin_x:.2f}, {origin_y:.2f})')
            self.get_logger().info(f'Grid extent: X=[{origin_x:.2f}, {origin_x + self.grid_size:.2f}], '
                                   f'Y=[{origin_y:.2f}, {origin_y + self.grid_size:.2f}]')
        
        # 5. Populate Grid
        # Initialize as OCCUPIED (100) - road pixels will be marked FREE (0)
        grid_data = np.full((self.grid_cells, self.grid_cells), OCCUPIED, dtype=np.int8)
        
        fill_grid_from_points(
            gx, gy, valid_mask, grid_data,
            origin_x, origin_y, self.resolution,
            self.grid_cells, self.grid_cells,
            FREE
        )
        
        # DEBUG: Count free cells in grid
        n_free = np.sum(grid_data == 0)
        self.get_logger().info(f'Free cells in grid: {n_free}/{self.grid_cells * self.grid_cells}')
        
        # 6. Publish
        self._publish_grid(grid_data, origin_x, origin_y, msg.header.stamp)
        t_end_semantic_cb = time.perf_counter()
        self.get_logger().info(f'Semantic callback processing time: {(t_end_semantic_cb - t_start_semantic_cb)*1000:.1f} ms')
    def _publish_empty_grid(self, stamp):
        """Publish an empty grid (all unknown) when no road pixels found."""
        # Need vehicle position for origin even for empty grid
        vehicle_pos = self._get_vehicle_position(stamp)
        if vehicle_pos is None:
            return
        origin_x = vehicle_pos[0] - self.grid_size / 2.0
        origin_y = vehicle_pos[1] - self.grid_size / 2.0
        
        grid_data = np.full((self.grid_cells, self.grid_cells), -1, dtype=np.int8)
        self._publish_grid(grid_data, origin_x, origin_y, stamp)
    
    def _publish_grid(self, grid_data: np.ndarray, origin_x: float, origin_y: float, stamp):
        """Publish occupancy grid message in map frame (world-aligned)."""
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = self.map_frame  # World-fixed frame, no yaw rotation
        
        msg.info.resolution = self.resolution
        msg.info.width = self.grid_cells
        msg.info.height = self.grid_cells
        msg.info.origin.position.x = origin_x
        msg.info.origin.position.y = origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0  # No rotation - aligned to map axes
        
        msg.data = grid_data.flatten().tolist()
        self.costmap_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = InversePerspectiveMapperNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()