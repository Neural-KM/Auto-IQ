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
from numba import jit

# Import semantic labels from perception module (CARLA convention)
from perception.data_classes.semantic_labels import SEMANTIC_BY_NAME

# ================= CONFIGURATION =================
# Drivable area - only road and road_line are free space
ROAD_LABEL_ID = SEMANTIC_BY_NAME['road']           # 1 in CARLA
ROAD_LINE_LABEL_ID = SEMANTIC_BY_NAME['road_line'] # 24 in CARLA

@jit(nopython=True, cache=True)
def project_pixels_to_ground(u_arr, v_arr, fx, fy, cx, cy, 
                             R_cam2map, t_cam2map, ground_z):
    """
    Project image pixels to ground plane (Z=ground_z) in map frame.
    
    CARLA uses camera_link frame convention (X=forward, Y=left, Z=up),
    but pixel unprojection assumes optical frame (X=right, Y=down, Z=forward).
    
    We convert: optical [norm_x, norm_y, 1] -> link [1, -norm_x, -norm_y]
    
    Math:
    1. Pixel to Ray (Optical Frame): ray_optical = [(u-cx)/fx, (v-cy)/fy, 1]
    2. Convert to Camera Link Frame: ray_link = [1, -norm_x, -norm_y]
    3. Ray to Map Frame: ray_map = R * ray_link
    4. Intersection with Ground (Z=ground_z):
       s = (ground_z - tz) / dz
    """
    n_points = len(u_arr)
    out_x = np.empty(n_points, dtype=np.float32)
    out_y = np.empty(n_points, dtype=np.float32)
    valid_mask = np.zeros(n_points, dtype=np.bool_)
    
    # Extract translation (camera origin in map frame)
    tx, ty, tz = t_cam2map[0], t_cam2map[1], t_cam2map[2]
    
    for i in range(n_points):
        u = u_arr[i]
        v = v_arr[i]
        
        # 1. Unproject to normalized ray in Camera Optical Frame
        # Optical: X=right, Y=down, Z=forward
        norm_x = (u - cx) / fx
        norm_y = (v - cy) / fy
        
        # 2. Convert from Optical Frame to Camera Link Frame
        # CARLA camera_link: X=forward, Y=left, Z=up
        # Optical X (right) = -Link Y (left)
        # Optical Y (down) = -Link Z (up)  
        # Optical Z (forward) = Link X (forward)
        # So: link = [optical_z, -optical_x, -optical_y] = [1, -norm_x, -norm_y]
        link_x = 1.0
        link_y = -norm_x
        link_z = -norm_y
        
        # 3. Rotate Ray Direction to Map Frame
        # dir = R * [link_x, link_y, link_z]
        dx = R_cam2map[0,0]*link_x + R_cam2map[0,1]*link_y + R_cam2map[0,2]*link_z
        dy = R_cam2map[1,0]*link_x + R_cam2map[1,1]*link_y + R_cam2map[1,2]*link_z
        dz = R_cam2map[2,0]*link_x + R_cam2map[2,1]*link_y + R_cam2map[2,2]*link_z
        
        # 4. Intersect with Ground Plane (Z=ground_z)
        # We need s such that: tz + s * dz = ground_z  =>  s = (ground_z - tz) / dz
        # For rays pointing down (dz < 0) with camera above ground (tz > ground_z), s > 0
        
        if abs(dz) > 1e-4:  # Avoid division by zero (horizon rays)
            s = (ground_z - tz) / dz
            
            # Intersection must be in front of the ray (s > 0)
            if s > 0:
                gx = tx + s * dx
                gy = ty + s * dy
                
                # Filter points extremely far away which cause grid artifacts
                if s < 100.0: 
                    out_x[i] = gx
                    out_y[i] = gy
                    valid_mask[i] = True
                    
    return out_x, out_y, valid_mask

@jit(nopython=True, cache=True)
def fill_grid_from_points(x_arr, y_arr, valid_mask, grid, 
                          origin_x, origin_y, resolution, width, height):
    """
    Mark grid cells as FREE (0) based on ground points.
    Initializes grid with UNKNOWN (-1) or OBSTACLE (100) before filling.
    """
    n_points = len(x_arr)
    
    for i in range(n_points):
        if not valid_mask[i]:
            continue
            
        # Convert World (x, y) to Grid (c, r)
        c = int((x_arr[i] - origin_x) / resolution)
        r = int((y_arr[i] - origin_y) / resolution)
        
        if 0 <= c < width and 0 <= r < height:
            # Mark as FREE
            grid[r, c] = 0
            
            # Optional: Simple hole filling (dilate)
            # If resolution is high, single pixels might leave gaps. 
            # We can mark neighbors too.
            # if c+1 < width: grid[r, c+1] = 0
            # if r+1 < height: grid[r+1, c] = 0

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
        self.declare_parameter('grid_resolution', 0.2)   # meters per cell
        self.declare_parameter('grid_width', 200)        # cells
        self.declare_parameter('grid_height', 200)       # cells
        
        # Get parameter values
        semantic_topic = self.get_parameter('semantic_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        costmap_topic = self.get_parameter('costmap_topic').value
        self.base_frame = self.get_parameter('base_frame').value
        self.map_frame = self.get_parameter('map_frame').value
        
        self.resolution = self.get_parameter('grid_resolution').value
        self.width = self.get_parameter('grid_width').value
        self.height = self.get_parameter('grid_height').value
        
        # Grid size in meters (for centering on vehicle)
        self.grid_size_x = self.width * self.resolution
        self.grid_size_y = self.height * self.resolution
        
        # ==================== TF2 ====================
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=1.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # ==================== State ====================
        self.K = None  # Camera intrinsics matrix [3x3]
        self.camera_frame = None
        
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
        self.get_logger().info(f'  Grid: {self.width}x{self.height} cells, {self.resolution}m resolution')
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
        if self.K is None:
            return
        
        # 1. Get Extrinsics for this specific frame time
        R, t = self._get_extrinsics(msg.header.stamp)
        if R is None:
            return
        
        # DEBUG: Log camera transform
        self.get_logger().info(f'Camera transform: t=[{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}]')
        self.get_logger().info(f'Camera R row2 (Z-axis in base): [{R[2,0]:.3f}, {R[2,1]:.3f}, {R[2,2]:.3f}]')
        
        # Test a single pixel from bottom-center of image (should point at ground)
        test_u = 640.0  # center x
        test_v = 700.0  # near bottom y
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        norm_x = (test_u - cx) / fx
        norm_y = (test_v - cy) / fy
        # Convert from optical to camera_link frame (CARLA convention)
        link_x, link_y, link_z = 1.0, -norm_x, -norm_y
        dx = R[0,0]*link_x + R[0,1]*link_y + R[0,2]*link_z
        dy = R[1,0]*link_x + R[1,1]*link_y + R[1,2]*link_z
        dz = R[2,0]*link_x + R[2,1]*link_y + R[2,2]*link_z
        self.get_logger().info(f'Test pixel ({test_u}, {test_v}): link_ray=[{link_x:.3f}, {link_y:.3f}, {link_z:.3f}], base_ray=[{dx:.3f}, {dy:.3f}, {dz:.3f}]')
        if abs(dz) > 1e-4:
            s = -t[2] / dz
            self.get_logger().info(f'  s={s:.3f} (s>0 required for valid intersection)')
            if s > 0:
                gx = t[0] + s * dx
                gy = t[1] + s * dy
                self.get_logger().info(f'  Ground intersection: ({gx:.2f}, {gy:.2f})')
        
        # 2. Parse Image directly from buffer
        try:
            height = msg.height
            width = msg.width
            self.get_logger().info(f'Image size: {width}x{height}, encoding: {msg.encoding}')
            # CARLA semantic segmentation uses BGRA format (4 bytes per pixel)
            # Label is stored in R channel (index 2 in BGRA order)
            if msg.encoding == 'bgra8':
                # Parse as 4-channel image, extract R channel (index 2)

                img_data = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width, 4)
                label_img = img_data[:, :, 2]  # R channel has the label
            elif msg.encoding == 'mono8':
                self.get_logger().info('Mono8 encoding detected')
                # Direct grayscale label image
                label_img = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width)
            elif msg.encoding == 'rgb8':
                # RGB format - label might be in first channel
                img_data = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width, 3)
                label_img = img_data[:, :, 0]  # R channel
            else:
                self.get_logger().error(f'Unsupported encoding: {msg.encoding}')
                return
        except Exception as e:
            self.get_logger().error(f'Image parsing error: {e}')
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
        origin_x = vehicle_x - self.grid_size_x / 2.0
        origin_y = vehicle_y - self.grid_size_y / 2.0

        # 4. Project to Ground (Numba JIT)
        # Use vehicle Z as ground plane (CARLA ground is not at Z=0)
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
            self.get_logger().info(f'Grid extent: X=[{origin_x:.2f}, {origin_x + self.grid_size_x:.2f}], '
                                   f'Y=[{origin_y:.2f}, {origin_y + self.grid_size_y:.2f}]')
        
        # 5. Populate Grid
        # Initialize as OCCUPIED (100) - road pixels will be marked FREE (0)
        grid_data = np.full((self.height, self.width), 100, dtype=np.int8)
        
        fill_grid_from_points(
            gx, gy, valid_mask, grid_data,
            origin_x, origin_y, self.resolution,
            self.width, self.height
        )
        
        # DEBUG: Count free cells in grid
        n_free = np.sum(grid_data == 0)
        self.get_logger().info(f'Free cells in grid: {n_free}/{self.width * self.height}')
        
        # 6. Publish
        self._publish_grid(grid_data, origin_x, origin_y, msg.header.stamp)
    
    def _publish_empty_grid(self, stamp):
        """Publish an empty grid (all unknown) when no road pixels found."""
        # Need vehicle position for origin even for empty grid
        vehicle_pos = self._get_vehicle_position(stamp)
        if vehicle_pos is None:
            return
        origin_x = vehicle_pos[0] - self.grid_size_x / 2.0
        origin_y = vehicle_pos[1] - self.grid_size_y / 2.0
        
        grid_data = np.full((self.height, self.width), -1, dtype=np.int8)
        self._publish_grid(grid_data, origin_x, origin_y, stamp)
    
    def _publish_grid(self, grid_data: np.ndarray, origin_x: float, origin_y: float, stamp):
        """Publish occupancy grid message in map frame (world-aligned)."""
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = self.map_frame  # World-fixed frame, no yaw rotation
        
        msg.info.resolution = self.resolution
        msg.info.width = self.width
        msg.info.height = self.height
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