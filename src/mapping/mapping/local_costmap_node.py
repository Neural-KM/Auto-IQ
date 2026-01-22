#!/usr/bin/env python3
"""
Rolling Window Binary Bayes Filter Costmap Node
================================================

ROS2 node that generates an odom-aligned rolling window occupancy grid
using a Binary Bayes Filter with log-odds representation.

Key Features:
- Grid stays aligned with odom frame (world-fixed, no rotation)
- Grid "rolls" to keep ego-vehicle centered without smearing
- Log-odds representation with hysteresis clamping
- Bresenham raycasting for free space marking
- Robust ego-motion handling

Math (Log-Odds):
- Initialization: L = 0.0 (Unknown / 0.5 probability)
- Clamping: Clip values between [-5.0, 5.0] (hysteresis filter)
- Hit: Add +2.0 (tunable l_hit) for occupied cells
- Miss: Subtract 0.7 (tunable l_miss) for free cells

Motion Handling (The "Roll" Logic):
- Input: Robot position in odom frame (x_curr, y_curr)
- Shift Calculation: Calculate integer index shift (dx, dy)
- Operation: Use numpy.roll(grid, (-shift_x, -shift_y), axis=(0,1))
- Cleanup: Zero out new rows/columns that wrap around
- Origin Update: Update physical world coordinate of grid's [0,0] index

Subscribes:
- /perception/semantic_pointcloud (PointCloud2) - Labeled XYZ + label
- /carla/ego_vehicle/ground_truth/odometry (Odometry) - Vehicle state

Publishes:
- /local_costmap (OccupancyGrid) - 2D occupancy grid in odom frame
"""

import struct
import time
import numpy as np
from typing import Tuple, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

import tf2_ros
from tf2_ros import TransformException

from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry
from std_msgs.msg import Header

from perception.data_classes.semantic_labels import SEMANTIC_BY_NAME
from numba import jit

# ==================== Semantic Class Sets ====================
# Drivable area - only road and road_line are free space
DRIVABLE_CLASSES = {
    SEMANTIC_BY_NAME['road'],
    SEMANTIC_BY_NAME['road_line'],
}

# Everything else is an obstacle (all classes not in DRIVABLE_CLASSES)

@jit(nopython=True, cache=True)
def bresenham_line_fast(x0, y0, x1, y1):
    """JIT-compiled Bresenham algorithm."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    # Pre-allocate output arrays
    # Max length of a line is dx + dy
    max_len = dx + dy + 1
    x_arr = np.empty(max_len, dtype=np.int32)
    y_arr = np.empty(max_len, dtype=np.int32)
    
    x, y = x0, y0
    idx = 0
    
    while True:
        if x == x1 and y == y1:
            break
            
        x_arr[idx] = x
        y_arr[idx] = y
        idx += 1
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
            
    return x_arr[:idx], y_arr[:idx]

@jit(nopython=True, cache=True)
def compute_raycast_batch(robot_ix, robot_iy, hit_ix_arr, hit_iy_arr, ray_mask, max_ray_len):
    """
    Batch process all raycasts in a single JIT-compiled loop.
    Avoids Python interpreter overhead for thousands of rays.
    """
    n_hits = len(hit_ix_arr)
    
    # Heuristic allocation: assume avg ray is half of max range
    # 5000 hits * 100 cells = 500k ints = ~2MB RAM (Safe to allocate)
    est_size = n_hits * max_ray_len
    
    all_free_x = np.empty(est_size, dtype=np.int32)
    all_free_y = np.empty(est_size, dtype=np.int32)
    
    count = 0
    
    for i in range(n_hits):
        # Skip if mask is false (too far)
        if not ray_mask[i]:
            continue
            
        hx = hit_ix_arr[i]
        hy = hit_iy_arr[i]
        
        # Call JIT bresenham
        xs, ys = bresenham_line_fast(robot_ix, robot_iy, hx, hy)
        n_pts = len(xs)
        
        # Safety check for buffer overflow (rare, but good practice)
        if count + n_pts >= len(all_free_x):
             break # Or handle resize, but with generous est_size this won't happen
            
        all_free_x[count : count + n_pts] = xs
        all_free_y[count : count + n_pts] = ys
        count += n_pts
        
    return all_free_x[:count], all_free_y[:count]
class LocalCostmapNode(Node):
    """
    ROS2 Node for generating odom-aligned rolling window costmap.
    
    Uses Binary Bayes Filter with log-odds representation.
    Grid stays axis-aligned with odom frame and rolls to follow the vehicle.
    """
    
    def __init__(self):
        super().__init__('local_costmap_node')
        
        # ==================== Parameters ====================
        # self.declare_parameter('pointcloud_topic', '/perception/semantic_pointcloud')
        self.declare_parameter('pointcloud_topic', '/carla/ego_vehicle/ground_truth/lidar/top_semantic/point_cloud')
        self.declare_parameter('odom_topic', '/carla/ego_vehicle/ground_truth/odometry')
        self.declare_parameter('costmap_topic', '/mapping/local_costmap')
        self.declare_parameter('odom_frame', 'map')  # Use 'map' since TF tree is map->ego_vehicle
        
        # Grid parameters
        self.declare_parameter('grid_size', 40.0)       # meters (square grid)
        self.declare_parameter('resolution', 0.2)       # meters per cell
        
        # Height filtering
        self.declare_parameter('min_height', -0.3)      # meters relative to vehicle
        self.declare_parameter('max_height', 2.5)       # meters relative to vehicle
        
        # Log-odds parameters
        self.declare_parameter('l_hit', 2.0)            # Log-odds update for occupied
        self.declare_parameter('l_miss', 0.1)           # Log-odds update for free
        self.declare_parameter('l_min', -5.0)           # Minimum log-odds (clamp)
        self.declare_parameter('l_max', 5.0)            # Maximum log-odds (clamp)
        self.declare_parameter('l_occ_threshold', 0.0)  # Threshold for occupied in output
        self.declare_parameter('decay_rate', 0.1)        # Log-odds decay rate per second
        # Raycasting parameters
        self.declare_parameter('max_ray_range', 30.0)   # Max range for raycasting (meters)
        self.declare_parameter('enable_raycasting', True)  # Enable free space raycasting
        
        # Get parameter values
        pointcloud_topic = self.get_parameter('pointcloud_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        costmap_topic = self.get_parameter('costmap_topic').value
        self.odom_frame = self.get_parameter('odom_frame').value
        
        self.grid_size = self.get_parameter('grid_size').value
        self.resolution = self.get_parameter('resolution').value
        
        self.min_height = self.get_parameter('min_height').value
        self.max_height = self.get_parameter('max_height').value
        
        self.l_hit = self.get_parameter('l_hit').value
        self.l_miss = self.get_parameter('l_miss').value
        self.l_min = self.get_parameter('l_min').value
        self.l_max = self.get_parameter('l_max').value
        self.l_occ_threshold = self.get_parameter('l_occ_threshold').value
        self.decay_rate = self.get_parameter('decay_rate').value

        self.max_ray_range = self.get_parameter('max_ray_range').value
        self.enable_raycasting = self.get_parameter('enable_raycasting').value
        
        # ==================== Grid Setup ====================
        # Square grid centered on vehicle
        self.grid_cells = int(self.grid_size / self.resolution)
        
        # Log-odds grid (float32)
        # Initialized to 0.0 (unknown = 0.5 probability)
        self.log_odds_grid = np.zeros((self.grid_cells, self.grid_cells), dtype=np.float32)
        
        # Grid origin in odom frame (world coordinates of cell [0,0])
        # Will be updated as vehicle moves
        self.origin_x = 0.0
        self.origin_y = 0.0
        
        # Last known vehicle position (for calculating roll shifts)
        # These track the position corresponding to current grid origin
        # Updated only by quantized amounts to preserve fractional offsets
        self.last_x = None
        self.last_y = None
        
        # Flag for first update
        self.initialized = False
        
        # Build obstacle label lookup - everything NOT in DRIVABLE_CLASSES is an obstacle
        self.obstacle_labels = np.ones(256, dtype=bool)  # Default: everything is obstacle
        for label in DRIVABLE_CLASSES:
            self.obstacle_labels[label] = False  # Mark drivable classes as NOT obstacle
        
        self.get_logger().info(f'Grid: {self.grid_cells}x{self.grid_cells} cells, '
                               f'{self.grid_size}x{self.grid_size}m, '
                               f'resolution={self.resolution}m')
        self.get_logger().info(f'Drivable classes: road, road_line. All other classes are obstacles.')
        self.get_logger().info(f'Log-odds: hit={self.l_hit}, miss={self.l_miss}, '
                               f'clamp=[{self.l_min}, {self.l_max}]')
        
        # ==================== TF2 ====================
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # ==================== State ====================
        self.current_odom: Optional[Odometry] = None
        
        # ==================== QoS ====================
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # ==================== Subscribers ====================
        self.pointcloud_sub = self.create_subscription(
            PointCloud2, pointcloud_topic,
            self.pointcloud_callback, sensor_qos)
        
        self.odom_sub = self.create_subscription(
            Odometry, odom_topic,
            self.odom_callback, sensor_qos)
        
        # ==================== Publisher ====================
        self.costmap_pub = self.create_publisher(
            OccupancyGrid, costmap_topic, 10)
        
        self.get_logger().info(f'Rolling Window Bayes Filter Costmap Node started')
        self.get_logger().info(f'  Input PointCloud: {pointcloud_topic}')
        self.get_logger().info(f'  Input Odometry: {odom_topic}')
        self.get_logger().info(f'  Output: {costmap_topic}')
        self.get_logger().info(f'  Odom Frame: {self.odom_frame}')
        self.get_logger().info(f'  Raycasting: {self.enable_raycasting}')
    
    def odom_callback(self, msg: Odometry):
        """Store latest odometry for position tracking."""
        self.current_odom = msg
    
    def pointcloud_callback(self, msg: PointCloud2):
        """Process labeled point cloud and update costmap."""
        t_start = time.perf_counter()
        
        if self.current_odom is None:
            self.get_logger().warn('No odometry received yet, skipping pointcloud')
            return
        
        # Get current vehicle position in odom frame
        odom_pos = self.current_odom.pose.pose.position
        x_curr = odom_pos.x
        y_curr = odom_pos.y
        
        # Initialize on first update
        if not self.initialized:
            self._initialize_grid(x_curr, y_curr)
        
        # Handle ego-motion: roll grid to keep vehicle centered
        self._roll_grid_for_motion(x_curr, y_curr)
        
        # Parse point cloud
        points, labels = self._parse_labeled_pointcloud(msg)
        
        if len(points) == 0:
            self._publish_costmap(msg.header.stamp)
            return
        
        # Transform points from sensor frame to odom frame
        points_odom = self._transform_points_to_odom(points, msg.header)
        
        if points_odom is None or len(points_odom) == 0:
            self._publish_costmap(msg.header.stamp)
            return
        
        # Height filter (relative to vehicle z)
        z_vehicle = odom_pos.z
        z = points_odom[:, 2]
        height_mask = (z >= z_vehicle + self.min_height) & (z <= z_vehicle + self.max_height)
        points_odom = points_odom[height_mask]
        labels = labels[height_mask]
        
        if len(points_odom) == 0:
            self._publish_costmap(msg.header.stamp)
            return
        
        # Update costmap with sensor data
        self._update_costmap(points_odom, labels, x_curr, y_curr)
        
        # Publish
        self._publish_costmap(msg.header.stamp)
        
        # Log timing
        t_elapsed = (time.perf_counter() - t_start) * 1000  # ms
        self.get_logger().info(f'Costmap update: {t_elapsed:.1f}ms ({len(points_odom)} points)', throttle_duration_sec=1.0)
    
    def _initialize_grid(self, x_curr: float, y_curr: float):
        """Initialize grid origin centered on vehicle position."""
        # Grid origin = vehicle position - half grid size
        self.origin_x = x_curr - self.grid_size / 2.0
        self.origin_y = y_curr - self.grid_size / 2.0
        
        self.last_x = x_curr
        self.last_y = y_curr
        
        self.initialized = True
        
        self.get_logger().info(f'Grid initialized at origin ({self.origin_x:.2f}, {self.origin_y:.2f})')
    
    def _roll_grid_for_motion(self, x_curr: float, y_curr: float):
        """
        Roll the grid to keep the vehicle centered.
        
        Uses numpy.roll for efficient shifting, then zeros out wrapped edges.
        This prevents "smearing" of obstacles during motion.
        
        """
        if self.last_x is None:
            return
        
        # Calculate displacement in world coordinates
        dx_world = x_curr - self.last_x
        dy_world = y_curr - self.last_y
        
        # Convert to cell shifts (truncate toward zero to get integer cells)
        # Use floor for positive, ceil for negative to get consistent truncation
        shift_x = int(np.floor(dx_world / self.resolution)) if dx_world >= 0 else int(np.ceil(dx_world / self.resolution))
        shift_y = int(np.floor(dy_world / self.resolution)) if dy_world >= 0 else int(np.ceil(dy_world / self.resolution))
        
        # Only roll if there's a significant shift
        if shift_x == 0 and shift_y == 0:
            return
        
        # Roll the grid
        # numpy.roll shifts array elements: positive = shift right/down
        # We want to shift the data in the opposite direction of vehicle motion
        # so the grid "follows" the vehicle
        self.log_odds_grid = np.roll(self.log_odds_grid, (-shift_y, -shift_x), axis=(0, 1))
        
        # Zero out the wrapped edges (these are now unknown/stale)
        if shift_x > 0:
            # Vehicle moved right (positive x), clear left edge (old right wrapped to left)
            self.log_odds_grid[:, :shift_x] = 0.0
        elif shift_x < 0:
            # Vehicle moved left (negative x), clear right edge
            self.log_odds_grid[:, shift_x:] = 0.0
        
        if shift_y > 0:
            # Vehicle moved forward (positive y in grid), clear bottom edge
            self.log_odds_grid[:shift_y, :] = 0.0
        elif shift_y < 0:
            # Vehicle moved backward (negative y in grid), clear top edge
            self.log_odds_grid[shift_y:, :] = 0.0
        
        # Calculate the QUANTIZED displacement (what we actually rolled)
        quantized_dx = shift_x * self.resolution
        quantized_dy = shift_y * self.resolution
        
        # Update origin by the quantized amount
        self.origin_x += quantized_dx
        self.origin_y += quantized_dy
        
        # CRITICAL: Only update last position by the quantized amount!
        # This preserves the fractional offset for the next iteration.
        # If we set last_x = x_curr, we lose the sub-cell movement and
        # the grid origin will drift from the vehicle position over time.
        self.last_x += quantized_dx
        self.last_y += quantized_dy
    
    def _parse_labeled_pointcloud(self, msg: PointCloud2) -> Tuple[np.ndarray, np.ndarray]:
        """Parse PointCloud2 with XYZ + label fields."""
        # Find field offsets
        x_offset = y_offset = z_offset = label_offset = 0
        for field in msg.fields:
            if field.name == 'x':
                x_offset = field.offset
            elif field.name == 'y':
                y_offset = field.offset
            elif field.name == 'z':
                z_offset = field.offset
            elif field.name == 'label':
                label_offset = field.offset
        
        point_step = msg.point_step
        n_points = msg.width * msg.height
        
        if n_points == 0:
            return np.array([]), np.array([])
        
        data = np.frombuffer(msg.data, dtype=np.uint8)
        
        # Vectorized parsing
        points = np.zeros((n_points, 3), dtype=np.float32)
        labels = np.zeros(n_points, dtype=np.uint32)
        
        for i in range(n_points):
            offset = i * point_step
            points[i, 0] = struct.unpack('f', data[offset + x_offset:offset + x_offset + 4])[0]
            points[i, 1] = struct.unpack('f', data[offset + y_offset:offset + y_offset + 4])[0]
            points[i, 2] = struct.unpack('f', data[offset + z_offset:offset + z_offset + 4])[0]
            labels[i] = struct.unpack('I', data[offset + label_offset:offset + label_offset + 4])[0]
        
        return points, labels
    
    def _transform_points_to_odom(self, points: np.ndarray, header: Header) -> Optional[np.ndarray]:
        """
        Transform points from sensor frame to odom frame.
        
        Args:
            points: Nx3 array in sensor frame
            header: PointCloud2 header (contains frame_id and timestamp)
            
        Returns:
            Nx3 array in odom frame, or None if transform fails
        """
        try:
            # Get transform from sensor frame to odom frame
            transform = self.tf_buffer.lookup_transform(
                self.odom_frame,
                header.frame_id,
                rclpy.time.Time(),  # Use latest available
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
        except TransformException as e:
            self.get_logger().warn(f'Transform lookup failed: {e}')
            return None
        
        # Extract rotation and translation
        t = transform.transform.translation
        q = transform.transform.rotation
        
        # Convert quaternion to rotation matrix
        qw, qx, qy, qz = q.w, q.x, q.y, q.z
        
        # Rotation matrix from quaternion
        R = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
        ], dtype=np.float32)
        
        translation = np.array([t.x, t.y, t.z], dtype=np.float32)
        
        # Transform points: p_odom = R @ p_sensor + t
        points_odom = (R @ points.T).T + translation
        
        return points_odom
    
    def _world_to_grid(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert world coordinates to grid indices.
        
        Args:
            x, y: Arrays of world coordinates in odom frame
            
        Returns:
            Tuple of (ix, iy) grid indices
        """
        ix = ((x - self.origin_x) / self.resolution).astype(np.int32)
        iy = ((y - self.origin_y) / self.resolution).astype(np.int32)
        return ix, iy
    
    def _update_costmap(self, points_odom: np.ndarray, labels: np.ndarray,
                        robot_x: float, robot_y: float):
        """
        Update costmap using Binary Bayes Filter.
        
        1. Raycast from robot to each hit point (mark free space with l_miss)
        2. Mark hit cells with l_hit for obstacles
        3. Clamp values to [l_min, l_max]
        
        Args:
            points_odom: Nx3 array of points in odom frame
            labels: N array of semantic labels
            robot_x, robot_y: Robot position in odom frame
        """
        x = points_odom[:, 0]
        y = points_odom[:, 1]
        
        # Convert to grid indices
        hit_ix, hit_iy = self._world_to_grid(x, y)
        
        # Filter points within grid bounds
        valid_mask = (
            (hit_ix >= 0) & (hit_ix < self.grid_cells) &
            (hit_iy >= 0) & (hit_iy < self.grid_cells)
        )
        hit_ix = hit_ix[valid_mask]
        hit_iy = hit_iy[valid_mask]
        labels = labels[valid_mask]
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        if len(hit_ix) == 0:
            return
        
        # Robot position in grid
        robot_ix, robot_iy = self._world_to_grid(
            np.array([robot_x]), np.array([robot_y])
        )
        # robot_ix, and iy with always be around self.grid_cells // 2
        robot_ix = int(robot_ix[0])
        robot_iy = int(robot_iy[0])
        self.get_logger().info(f'Robot grid position: ({robot_ix}, {robot_iy})')
        # Clamp robot position to grid bounds
        robot_ix = np.clip(robot_ix, 0, self.grid_cells - 1)
        robot_iy = np.clip(robot_iy, 0, self.grid_cells - 1)

        # Add decay so that cells not visible anymore go to zero over time
        abs_grid = np.abs(self.log_odds_grid)
        mask_decay = abs_grid > self.decay_rate
        mask_reset = abs_grid <= self.decay_rate
        
        # Apply decay where magnitude is large enough
        self.log_odds_grid[mask_decay] -= np.sign(self.log_odds_grid[mask_decay]) * self.decay_rate
        
        # Snap to zero where magnitude is small
        self.log_odds_grid[mask_reset] = 0.0
        # ==================== Raycasting for Free Space ====================
        if self.enable_raycasting:
            # Calculate distances for range limiting
            distances = np.sqrt((x_valid - robot_x)**2 + (y_valid - robot_y)**2)
            ray_mask = distances <= self.max_ray_range
            
            # Estimate max cells per ray for buffer allocation
            # (e.g. 30m / 0.2m = 150 cells)
            max_ray_cells = int(self.max_ray_range / self.resolution) + 10
            
            # --- NUMBA BATCH CALL ---
            # Replaces the slow Python for-loop
            free_ix_concat, free_iy_concat = compute_raycast_batch(
                robot_ix, robot_iy, hit_ix, hit_iy, ray_mask, max_ray_cells
            )
            
            # Apply free space updates (batch)
            if len(free_ix_concat) > 0:
                # Filter to valid indices
                valid = (
                    (free_ix_concat >= 0) & (free_ix_concat < self.grid_cells) &
                    (free_iy_concat >= 0) & (free_iy_concat < self.grid_cells)
                )
                free_ix_concat = free_ix_concat[valid]
                free_iy_concat = free_iy_concat[valid]
                
                # Apply l_miss to free cells
                np.subtract.at(self.log_odds_grid, (free_iy_concat, free_ix_concat), self.l_miss)
        # ==================== Mark Occupied Cells ====================
        # Only mark obstacle-class points as occupied
        is_obstacle = self.obstacle_labels[np.clip(labels, 0, 255)]
        obstacle_ix = hit_ix[is_obstacle]
        obstacle_iy = hit_iy[is_obstacle]
        
        if len(obstacle_ix) > 0:
            # Apply l_hit to occupied cells
            np.add.at(self.log_odds_grid, (obstacle_iy, obstacle_ix), self.l_hit)
        
        # ==================== Clamp Values ====================
        np.clip(self.log_odds_grid, self.l_min, self.l_max, out=self.log_odds_grid)
    
    def _log_odds_to_probability(self, log_odds: np.ndarray) -> np.ndarray:
        """Convert log-odds to probability [0, 1]."""
        return 1.0 / (1.0 + np.exp(-log_odds))
    
    def _publish_costmap(self, stamp):
        """
        Convert log-odds grid to OccupancyGrid and publish.
        
        Mapping:
        - log_odds < l_occ_threshold -> free (0)
        - log_odds > l_occ_threshold -> occupied (scaled 50-100)
        - log_odds == 0.0 (exactly) -> unknown (-1)
        """
        msg = OccupancyGrid()
        
        # Header - use odom frame since grid is world-aligned
        msg.header = Header()
        msg.header.stamp = stamp
        msg.header.frame_id = self.odom_frame
        
        # Metadata
        msg.info = MapMetaData()
        msg.info.resolution = self.resolution
        msg.info.width = self.grid_cells
        msg.info.height = self.grid_cells
        
        # Origin (position of cell [0,0] in odom frame)
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0  # No rotation - grid aligned with odom
        
        # Convert log-odds to occupancy values
        # OccupancyGrid expects: -1=unknown, 0=free, 100=occupied
        occupancy = np.zeros_like(self.log_odds_grid, dtype=np.int8)
        
        # Free space: log_odds < threshold
        free_mask = self.log_odds_grid < self.l_occ_threshold
        occupancy[free_mask] = 0
        
        # Occupied space: log_odds >= threshold
        # Scale from [threshold, l_max] to [50, 100]
        occ_mask = self.log_odds_grid >= self.l_occ_threshold
        if np.any(occ_mask):
            # Normalize to [0, 1] then scale to [50, 100]
            occ_range = self.l_max - self.l_occ_threshold
            if occ_range > 0:
                normalized = (self.log_odds_grid[occ_mask] - self.l_occ_threshold) / occ_range
                occupancy[occ_mask] = (50 + normalized * 50).astype(np.int8)
            else:
                occupancy[occ_mask] = 100
        
        # Unknown: cells that have never been updated (log_odds very close to 0)
        unknown_mask = np.abs(self.log_odds_grid) < 0.01
        occupancy[unknown_mask] = -1
        
        # Data in row-major order
        msg.data = occupancy.flatten().tolist()
        
        self.costmap_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LocalCostmapNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
