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


@jit(nopython=True, cache=True)
def compute_shadow_buffer(obstacle_x: np.ndarray, obstacle_y: np.ndarray,
                          robot_x: float, robot_y: float, 
                          angular_res: float) -> np.ndarray:
    """
    Build an angular depth buffer (shadow buffer) from obstacle points.
    
    For each angular bin (360°), stores the minimum distance to an obstacle.
    This creates a "shadow" behind each obstacle that will block raycasting.
    
    Args:
        obstacle_x: X coordinates of obstacle points (world frame)
        obstacle_y: Y coordinates of obstacle points (world frame)
        robot_x, robot_y: Robot position (world frame)
        angular_res: Angular resolution in radians (e.g., 1° = 0.0175)
        
    Returns:
        1D array of size (2*pi / angular_res) with minimum distance per bin.
        Bins with no obstacles have value np.inf.
    """
    n_bins = int(2.0 * np.pi / angular_res) + 1
    shadow_buffer = np.full(n_bins, np.inf, dtype=np.float32)
    
    n_obstacles = len(obstacle_x)
    
    for i in range(n_obstacles):
        dx = obstacle_x[i] - robot_x
        dy = obstacle_y[i] - robot_y
        
        # Calculate angle and distance
        angle = np.arctan2(dy, dx)  # [-pi, pi]
        distance = np.sqrt(dx * dx + dy * dy)
        
        # Convert angle to bin index [0, n_bins)
        bin_idx = int((angle + np.pi) / angular_res)
        
        # Keep minimum distance (closest obstacle)
        if distance < shadow_buffer[bin_idx]:
            shadow_buffer[bin_idx] = distance
    
    return shadow_buffer


@jit(nopython=True, cache=True)
def filter_rays_by_shadows(target_x: np.ndarray, target_y: np.ndarray,
                           robot_x: float, robot_y: float,
                           shadow_buffer: np.ndarray, angular_res: float,
                           tolerance: float) -> np.ndarray:
    """
    Filter target points that are occluded by obstacles (behind the shadow).
    
    A point is occluded if its distance is greater than the shadow buffer
    distance for its angular bin (plus a small tolerance).
    
    Args:
        target_x, target_y: Target point coordinates (world frame)
        robot_x, robot_y: Robot position (world frame)
        shadow_buffer: Angular depth buffer from compute_shadow_buffer
        angular_res: Angular resolution in radians (must match shadow buffer)
        tolerance: Distance tolerance in meters (points within tolerance pass through)
        
    Returns:
        Boolean mask - True for points that should be raycasted (not occluded)
    """
    n_points = len(target_x)
    n_bins = len(shadow_buffer)
    valid_mask = np.ones(n_points, dtype=np.bool_)
    
    for i in range(n_points):
        dx = target_x[i] - robot_x
        dy = target_y[i] - robot_y
        
        # Calculate angle and distance
        angle = np.arctan2(dy, dx)  # [-pi, pi]
        distance = np.sqrt(dx * dx + dy * dy)
        
        # Convert angle to bin index
        angle_normalized = angle + np.pi
        bin_idx = int(angle_normalized / angular_res)
        
        # Check if point is behind an obstacle (occluded)
        # If target distance > shadow distance + tolerance, it's behind an obstacle
        if distance > shadow_buffer[bin_idx] + tolerance:
            valid_mask[i] = False
    
    return valid_mask


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
        self.declare_parameter('decay_rate', 0.2)        # Log-odds decay rate per second
        # Raycasting parameters
        self.declare_parameter('max_ray_range', 30.0)   # Max range for raycasting (meters)
        
        # Shadow filter parameters (prevents ray passthrough)
        self.declare_parameter('shadow_angular_res', 1.0)  # Angular resolution in degrees
        self.declare_parameter('shadow_tolerance', 0.5)    # Distance tolerance in meters
        
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
        
        # Shadow filter parameters (convert degrees to radians)
        self.shadow_angular_res = np.radians(self.get_parameter('shadow_angular_res').value)
        self.shadow_tolerance = self.get_parameter('shadow_tolerance').value
        
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
        # Use a 1 second buffer to handle simulation time jumps
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=1.0))
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
            OccupancyGrid, costmap_topic, 1)
        
        self.get_logger().info(f'Rolling Window Bayes Filter Costmap Node started')
        self.get_logger().info(f'  Input PointCloud: {pointcloud_topic}')
        self.get_logger().info(f'  Input Odometry: {odom_topic}')
        self.get_logger().info(f'  Output: {costmap_topic}')
        self.get_logger().info(f'  Odom Frame: {self.odom_frame}')
    
    def odom_callback(self, msg: Odometry):
        """Store latest odometry for position tracking."""
        self.current_odom = msg
    
    def _get_vehicle_pose_at_time(self, stamp) -> Optional[Tuple[float, float, float]]:
        """
        Get vehicle pose (x, y, z) in odom frame at the specified timestamp.
        
        Uses exact timestamp matching - no fallback to latest.
        This ensures the costmap origin is computed at the same time as the
        pointcloud data, preventing lag/snap issues in RViz.
        
        Returns:
            Tuple of (x, y, z) or None if lookup fails
        """
        try:
            transform = self.tf_buffer.lookup_transform(
                self.odom_frame,
                'ego_vehicle',
                stamp
            )
            t = transform.transform.translation
            return (t.x, t.y, t.z)
        except TransformException as e:
            self.get_logger().warn(f'Vehicle pose lookup failed: {e}')
            return None
    
    def pointcloud_callback(self, msg: PointCloud2):
        """Process labeled point cloud and update costmap."""
        t_start = time.perf_counter()
        
        # Get vehicle pose at exact pointcloud timestamp
        result = self._get_vehicle_pose_at_time(msg.header.stamp)
        
        if result is None:
            self.get_logger().warn('No vehicle pose available, skipping pointcloud')
            return
        
        x_curr, y_curr, z_vehicle = result
        
        # Initialize on first update
        if not self.initialized:
            self._initialize_grid(x_curr, y_curr)
        
        # Handle ego-motion: roll grid to keep vehicle centered
        self._roll_grid_for_motion(x_curr, y_curr)
        
        # Parse point cloud
        points, labels = self._parse_labeled_pointcloud(msg)
        
        if len(points) == 0:
            self.get_logger().error('No points for costmap')
            return
        
        # Transform points from sensor frame to odom frame
        points_odom = self._transform_points_to_odom(points, msg.header)
        
        if points_odom is None or len(points_odom) == 0:
            self.get_logger().error('No points for costmap')
            return
        
        # Height filter (relative to vehicle z)
        z = points_odom[:, 2]
        height_mask = (z >= z_vehicle + self.min_height) & (z <= z_vehicle + self.max_height)
        points_odom = points_odom[height_mask]
        labels = labels[height_mask]
        
        if len(points_odom) == 0:
            self.get_logger().error('No points for costmap')
            return
        
        # Update costmap with sensor data
        self._update_costmap(points_odom, labels, x_curr, y_curr)
        
        # Publish with pointcloud timestamp (same as TF lookup timestamp)
        self._publish_costmap(msg.header.stamp)
        
        # Log timing
        t_elapsed = (time.perf_counter() - t_start) * 1000  # ms
        self.get_logger().info(f'Costmap update: {t_elapsed:.1f}ms ({len(points_odom)} points)', throttle_duration_sec=1.0)
    
    def _initialize_grid(self, x_curr: float, y_curr: float):
        """Initialize grid origin centered on vehicle position."""
        # Grid origin = vehicle position - half grid size
        # This places the vehicle at the center of the grid
        self.origin_x = x_curr - self.grid_size / 2.0
        self.origin_y = y_curr - self.grid_size / 2.0
        
        # Track the last position we rolled to (for calculating shift on next update)
        self.last_x = x_curr
        self.last_y = y_curr
        
        self.initialized = True
        
        self.get_logger().info(f'Grid initialized: vehicle at ({x_curr:.2f}, {y_curr:.2f}), '
                               f'origin at ({self.origin_x:.2f}, {self.origin_y:.2f})')
    
    def _roll_grid_for_motion(self, x_curr: float, y_curr: float):
        """
        Roll the grid to keep the vehicle centered.
        
        The key insight: the grid origin should ALWAYS be (vehicle_pos - grid_size/2).
        We use numpy.roll to shift the data so that the obstacles remain in correct
        world positions even though the grid is moving.
        
        Steps:
        1. Calculate how many cells the vehicle moved since last update
        2. Roll the grid data by that amount (opposite direction to vehicle motion)
        3. Clear the wrapped edges (stale data from the other side)
        4. Update origin to keep vehicle centered
        """
        if self.last_x is None:
            return
        
        # Calculate displacement in world coordinates since last roll
        dx_world = x_curr - self.last_x
        dy_world = y_curr - self.last_y
        
        # Convert to integer cell shifts
        # When vehicle moves +1 cell in world, we need to roll grid data -1 cell
        shift_x = int(round(dx_world / self.resolution))
        shift_y = int(round(dy_world / self.resolution))
        
        # Only roll if there's a significant shift (at least 1 cell)
        if shift_x == 0 and shift_y == 0:
            return
        
        # Roll the grid data
        # np.roll with negative shift moves data toward lower indices
        # When vehicle moves +X, we want obstacles to appear at lower X indices (relative to vehicle)
        # So we roll with (-shift_y, -shift_x)
        self.log_odds_grid = np.roll(self.log_odds_grid, (-shift_y, -shift_x), axis=(0, 1))
        
        # Clear the wrapped edges (stale data that wrapped around)
        # The data that "falls off" one edge wraps to the opposite edge - clear it
        if shift_x > 0:
            # Vehicle moved +X, data wrapped to high-X (right) edge
            self.log_odds_grid[:, -shift_x:] = 0.0
        elif shift_x < 0:
            # Vehicle moved -X, data wrapped to low-X (left) edge
            self.log_odds_grid[:, :-shift_x] = 0.0
        
        if shift_y > 0:
            # Vehicle moved +Y, data wrapped to high-Y (bottom) edge
            self.log_odds_grid[-shift_y:, :] = 0.0
        elif shift_y < 0:
            # Vehicle moved -Y, data wrapped to low-Y (top) edge
            self.log_odds_grid[:-shift_y, :] = 0.0
        
        # Update the grid origin to keep vehicle at center
        # Origin = vehicle_position - grid_size/2
        self.origin_x = x_curr - self.grid_size / 2.0
        self.origin_y = y_curr - self.grid_size / 2.0
        
        # Update last position to the current position
        # (We use the actual position, not quantized, to avoid drift)
        self.last_x = x_curr
        self.last_y = y_curr
    
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

        try:
            points_x = np.ndarray(shape=(n_points,), dtype=np.float32, buffer=msg.data, offset=x_offset, strides=(point_step,))
            points_y = np.ndarray(shape=(n_points,), dtype=np.float32, buffer=msg.data, offset=y_offset, strides=(point_step,))
            points_z = np.ndarray(shape=(n_points,), dtype=np.float32, buffer=msg.data, offset=z_offset, strides=(point_step,))
            
            # Stack to create Nx3 array (this creates a copy, but it's contiguous and fast)
            points = np.column_stack((points_x, points_y, points_z))
            
            # Parse labels
            labels = np.ndarray(shape=(n_points,), dtype=np.uint32, buffer=msg.data, offset=label_offset, strides=(point_step,))
            # Create a copy of labels to ensure 1D contiguous array for JIT compatibility
            labels = labels.copy() 
            
            return points, labels
        except Exception as e:
            self.get_logger().error(f'Error parsing pointcloud: {e}')
            return np.array([]), np.array([])
    
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
            # Use exact pointcloud timestamp - no fallback
            if not self.tf_buffer.can_transform(
                self.odom_frame,
                header.frame_id,
                header.stamp,
                timeout=rclpy.duration.Duration(seconds=1.0)
            ):
                self.get_logger().warn(f'TF not available for {header.frame_id} at exact timestamp', throttle_duration_sec=1.0)
                return None
            
            transform = self.tf_buffer.lookup_transform(
                self.odom_frame,
                header.frame_id,
                header.stamp
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
        
        # Robot is always at grid center since we keep origin = vehicle_pos - grid_size/2
        center = self.grid_cells // 2
        robot_ix = center
        robot_iy = center

        # Two seperate masks so that there are no oscillations between positive and negative values
        # Add decay so that cells not visible anymore go to zero over time
        abs_grid = np.abs(self.log_odds_grid)
        mask_decay = abs_grid > self.decay_rate
        mask_reset = abs_grid <= self.decay_rate
        # Apply decay where magnitude of odds is larger than the decay itself
        self.log_odds_grid[mask_decay] -= np.sign(self.log_odds_grid[mask_decay]) * self.decay_rate
        # Snap to zero where magnitude is smaller than decay
        self.log_odds_grid[mask_reset] = 0.0
        
        # ==================== Identify Obstacle Points First ====================
        # We need obstacles BEFORE raycasting to build the shadow buffer
        is_obstacle = self.obstacle_labels[labels]
        obstacle_ix = hit_ix[is_obstacle]
        obstacle_iy = hit_iy[is_obstacle]
        obstacle_x = x_valid[is_obstacle]
        obstacle_y = y_valid[is_obstacle]
        
        # ==================== Raycasting ====================
        if len(hit_ix) > 0:
            # STEP 1: Build shadow buffer from obstacle points
            # This creates an angular depth buffer that prevents rays from passing
            # through obstacles to reach ground points behind them
            shadow_buffer = compute_shadow_buffer(
                obstacle_x.astype(np.float32),
                obstacle_y.astype(np.float32),
                robot_x, robot_y,
                self.shadow_angular_res
            )
            
            # Step 2: remove duplicates hits in grid cell (optimization)
            cell_coords = np.column_stack((hit_ix, hit_iy))
            # self.get_logger().info(f'Raycasting {len(hit_ix)} hits before deduplication')
            unique_cells, unique_indices = np.unique(
                cell_coords.view(dtype=[('x', np.int32), ('y', np.int32)]).reshape(-1),
                return_index=True
            )
            # self.get_logger().info(f'Raycasting {len(unique_indices)} hits after deduplication')
            unique_ix = hit_ix[unique_indices]
            unique_iy = hit_iy[unique_indices]
            unique_x = x_valid[unique_indices]
            unique_y = y_valid[unique_indices]
            
            # STEP 3: Filter rays by shadow buffer
            # Remove points that are behind obstacles (would cause ray passthrough)
            shadow_mask = filter_rays_by_shadows(
                unique_x.astype(np.float32),
                unique_y.astype(np.float32),
                robot_x, robot_y,
                shadow_buffer,
                self.shadow_angular_res,
                self.shadow_tolerance
            )
            
            # Apply shadow filter
            filtered_ix = unique_ix[shadow_mask]
            filtered_iy = unique_iy[shadow_mask]
            filtered_x = unique_x[shadow_mask]
            filtered_y = unique_y[shadow_mask]
            
            # STEP 4: Range filter on remaining points
            distances = np.sqrt((filtered_x - robot_x)**2 + (filtered_y - robot_y)**2)
            ray_mask = distances <= self.max_ray_range
            
            # Estimate max cells per ray for buffer allocation
            max_ray_cells = int(self.max_ray_range / self.resolution) + 10
            
            # STEP 5: Raycast to filtered points only
            free_ix_concat, free_iy_concat = compute_raycast_batch(
                robot_ix, robot_iy, filtered_ix, filtered_iy, ray_mask, max_ray_cells
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
        
        Args:
            stamp: builtin_interfaces.msg.Time from pointcloud header
        
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
