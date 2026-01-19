#!/usr/bin/env python3
"""
Local Costmap Generator Node
============================

ROS2 node that generates an ego-centric rolling window occupancy grid
from semantic labeled point cloud data.

Pipeline:
1. Receive labeled point cloud (XYZ + semantic label)
2. Project 3D points onto 2D local grid centered on ego vehicle
3. Assign costs based on semantic labels:
   - Drivable (road, road_line, ground): Free Space (0)
   - Obstacles (vehicle, pedestrian, etc.): Lethal (100)
   - Unknown: -1
4. Resolve conflicts with obstacle priority for safety
5. Publish as nav_msgs/OccupancyGrid

Subscribes:
- /perception/semantic_pointcloud (PointCloud2) - Labeled XYZ + label

Publishes:
- /mapping/local_costmap (OccupancyGrid) - 2D occupancy grid
"""

import struct
import numpy as np
from scipy import ndimage

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid, MapMetaData
from std_msgs.msg import Header

from perception.data_classes.semantic_labels import SEMANTIC_BY_NAME


# ==================== Cost Values ====================
COST_UNKNOWN = -1
COST_FREE = 0
COST_LETHAL = 100

# ==================== Semantic Class Sets ====================
# Drivable surfaces - mark as free
DRIVABLE_CLASSES = {
    SEMANTIC_BY_NAME['road'],
    SEMANTIC_BY_NAME['road_line'],
    SEMANTIC_BY_NAME['ground'],
    SEMANTIC_BY_NAME['terrain'],
}

# Obstacles - mark as lethal
OBSTACLE_CLASSES = {
    SEMANTIC_BY_NAME['vehicle'],
    SEMANTIC_BY_NAME['truck'],
    SEMANTIC_BY_NAME['bus'],
    SEMANTIC_BY_NAME['motorcycle'],
    SEMANTIC_BY_NAME['bicycle'],
    SEMANTIC_BY_NAME['pedestrian'],
    SEMANTIC_BY_NAME['rider'],
    SEMANTIC_BY_NAME['building'],
    SEMANTIC_BY_NAME['wall'],
    SEMANTIC_BY_NAME['fence'],
    SEMANTIC_BY_NAME['pole'],
    SEMANTIC_BY_NAME['traffic_light'],
    SEMANTIC_BY_NAME['traffic_sign'],
    SEMANTIC_BY_NAME['static'],
    SEMANTIC_BY_NAME['dynamic'],
    SEMANTIC_BY_NAME['guard_rail'],
}


class LocalCostmapNode(Node):
    """
    ROS2 Node for generating ego-centric local costmap from semantic point cloud.
    
    Creates a rolling window occupancy grid centered on the ego vehicle,
    using semantic labels to classify cells as free, occupied, or unknown.
    """
    
    def __init__(self):
        super().__init__('local_costmap_node')
        
        # ==================== Parameters ====================
        self.declare_parameter('pointcloud_topic', '/perception/semantic_pointcloud')
        self.declare_parameter('costmap_topic', '/mapping/local_costmap')
        self.declare_parameter('frame_id', 'ego_vehicle')
        
        # Grid parameters
        self.declare_parameter('grid_width', 40.0)      # meters (lateral, left-right)
        self.declare_parameter('grid_length', 40.0)     # meters (longitudinal, front-back)
        self.declare_parameter('resolution', 0.2)       # meters per cell
        self.declare_parameter('origin_x_offset', 0.5)  # 0.5 = centered, 0.25 = rear-biased
        
        # Height filtering
        self.declare_parameter('min_height', -0.5)      # meters (below ground level)
        self.declare_parameter('max_height', 3.0)       # meters (above ground level)
        
        # Gap filling (for sparse LiDAR scans)
        self.declare_parameter('fill_gaps', True)       # Enable morphological gap filling
        self.declare_parameter('gap_fill_kernel', 3)    # Kernel size for closing operation
        
        # LiDAR blind spot - area directly under/around sensor where ground isn't visible
        # Blind spot radius = lidar_height / tan(abs(lidar_lowest_angle))
        self.declare_parameter('lidar_height', 2.4)              # meters above ground
        self.declare_parameter('lidar_lowest_angle', -30.0)      # degrees (negative = below horizontal)
        
        pointcloud_topic = self.get_parameter('pointcloud_topic').value
        costmap_topic = self.get_parameter('costmap_topic').value
        self.frame_id = self.get_parameter('frame_id').value
        
        self.grid_width = self.get_parameter('grid_width').value
        self.grid_length = self.get_parameter('grid_length').value
        self.resolution = self.get_parameter('resolution').value
        self.origin_x_offset = self.get_parameter('origin_x_offset').value
        
        self.min_height = self.get_parameter('min_height').value
        self.max_height = self.get_parameter('max_height').value
        
        self.fill_gaps = self.get_parameter('fill_gaps').value
        self.gap_fill_kernel = self.get_parameter('gap_fill_kernel').value
        
        lidar_height = self.get_parameter('lidar_height').value
        lidar_lowest_angle = self.get_parameter('lidar_lowest_angle').value
        
        # Calculate blind spot radius: r = h / tan(|angle|)
        # This is the radius inside which the LiDAR cannot see the ground
        angle_rad = np.deg2rad(abs(lidar_lowest_angle))
        if angle_rad > 0.01:  # Avoid division by zero
            self.blind_spot_radius = lidar_height / np.tan(angle_rad)
        else:
            self.blind_spot_radius = 0.0
        
        self.get_logger().info(f'LiDAR blind spot radius: {self.blind_spot_radius:.2f}m '
                               f'(height={lidar_height}m, lowest_angle={lidar_lowest_angle}°)')
        
        # ==================== Grid Setup ====================
        # Grid dimensions in cells
        # OccupancyGrid: width = X direction, height = Y direction
        # Vehicle frame: X = forward, Y = left
        self.size_x = int(self.grid_length / self.resolution)  # forward (X)
        self.size_y = int(self.grid_width / self.resolution)   # lateral (Y)
        
        # Grid origin in vehicle frame (corner at min X, min Y)
        self.origin_x = -self.grid_length * self.origin_x_offset
        self.origin_y = -self.grid_width / 2.0
        
        self.get_logger().info(f'Grid: {self.size_x}x{self.size_y} cells, '
                               f'{self.grid_length}x{self.grid_width}m, '
                               f'resolution={self.resolution}m')
        self.get_logger().info(f'Origin: ({self.origin_x:.1f}, {self.origin_y:.1f})m')
        
        # ==================== Build label lookup arrays ====================
        # For vectorized operations: map label -> cost type
        # 0 = unknown, 1 = free, 2 = obstacle
        self.label_to_type = np.zeros(256, dtype=np.uint8)
        for label in DRIVABLE_CLASSES:
            self.label_to_type[label] = 1  # Free
        for label in OBSTACLE_CLASSES:
            self.label_to_type[label] = 2  # Obstacle
        
        # ==================== QoS ====================
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )
        
        # ==================== Subscriber ====================
        self.pointcloud_sub = self.create_subscription(
            PointCloud2, pointcloud_topic,
            self.pointcloud_callback, sensor_qos)
        
        # ==================== Publisher ====================
        self.costmap_pub = self.create_publisher(
            OccupancyGrid, costmap_topic, 10)
        
        self.get_logger().info(f'Local Costmap Node started')
        self.get_logger().info(f'  Input: {pointcloud_topic}')
        self.get_logger().info(f'  Output: {costmap_topic}')
    
    def pointcloud_callback(self, msg: PointCloud2):
        """Process labeled point cloud and generate occupancy grid."""
        # Parse point cloud
        points, labels = self._parse_labeled_pointcloud(msg)
        
        if len(points) == 0:
            return
        
        # Generate costmap
        costmap = self._generate_costmap(points, labels)
        
        # Mark immediate area around vehicle as free (LiDAR blind spot)
        costmap = self._mark_lidar_spot_free(costmap)
        
        # Fill gaps between LiDAR channels using morphological closing
        # This creates a continuous free space for the corridor generator
        if self.fill_gaps:
            costmap = self._fill_free_space_gaps(costmap)
        
        # Publish
        grid_msg = self._create_occupancy_grid(costmap, msg.header.stamp)
        self.costmap_pub.publish(grid_msg)
    
    def _parse_labeled_pointcloud(self, msg: PointCloud2):
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
        
        # Vectorized parsing using numpy views
        # Create structured array view
        points = np.zeros((n_points, 3), dtype=np.float32)
        labels = np.zeros(n_points, dtype=np.uint32)
        
        for i in range(n_points):
            offset = i * point_step
            points[i, 0] = struct.unpack('f', data[offset + x_offset:offset + x_offset + 4])[0]
            points[i, 1] = struct.unpack('f', data[offset + y_offset:offset + y_offset + 4])[0]
            points[i, 2] = struct.unpack('f', data[offset + z_offset:offset + z_offset + 4])[0]
            labels[i] = struct.unpack('I', data[offset + label_offset:offset + label_offset + 4])[0]
        
        return points, labels
    
    def _generate_costmap(self, points: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Generate 2D costmap from 3D labeled points.
        
        Uses vectorized NumPy operations for efficiency.
        Obstacle points have priority over drivable points for safety.
        
        Args:
            points: Nx3 array of XYZ coordinates in ego vehicle frame
            labels: N array of semantic class IDs
            
        Returns:
            2D costmap array (size_y x size_x), values: -1=unknown, 0=free, 100=lethal
        """
        # Initialize grid as unknown
        # Shape: (height, width) = (size_y, size_x) for row-major storage
        costmap = np.full((self.size_y, self.size_x), COST_UNKNOWN, dtype=np.int8)
        
        if len(points) == 0:
            return costmap
        
        # ==================== Height Filtering ====================
        z = points[:, 2]
        height_mask = (z >= self.min_height) & (z <= self.max_height)
        points = points[height_mask]
        labels = labels[height_mask]
        
        if len(points) == 0:
            return costmap
        
        # ==================== Project to Grid Indices ====================
        # Vehicle frame: X=forward, Y=left
        x = points[:, 0]
        y = points[:, 1]
        
        # Convert to grid indices
        # ix = column index (X direction), iy = row index (Y direction)
        ix = ((x - self.origin_x) / self.resolution).astype(np.int32)
        iy = ((y - self.origin_y) / self.resolution).astype(np.int32)
        
        # Filter points within grid bounds
        valid_mask = (
            (ix >= 0) & (ix < self.size_x) &
            (iy >= 0) & (iy < self.size_y)
        )
        ix = ix[valid_mask]
        iy = iy[valid_mask]
        labels = labels[valid_mask]
        
        if len(ix) == 0:
            return costmap
        
        # ==================== Classify Points by Type ====================
        # Map labels to types: 0=unknown, 1=free, 2=obstacle
        point_types = self.label_to_type[np.clip(labels, 0, 255)]
        
        # Separate free and obstacle points
        free_mask = point_types == 1
        obstacle_mask = point_types == 2
        
        # ==================== Update Grid (Free First, Then Obstacles) ====================
        # This ensures obstacles overwrite free cells for safety
        
        # Mark free cells
        if np.any(free_mask):
            costmap[iy[free_mask], ix[free_mask]] = COST_FREE
        
        # Mark obstacle cells (overwrites free if overlapping)
        if np.any(obstacle_mask):
            costmap[iy[obstacle_mask], ix[obstacle_mask]] = COST_LETHAL
        
        return costmap
    
    def _mark_lidar_spot_free(self, costmap: np.ndarray) -> np.ndarray:
        """
        Mark the LiDAR blind spot area around the vehicle as free space.
        
        The LiDAR cannot see the ground within a radius determined by:
            r = lidar_height / tan(|lowest_beam_angle|)
        
        Since we know the vehicle is on drivable ground, we mark this
        circular region as free (only for unknown cells, not obstacles).
        
        Args:
            costmap: 2D array with -1=unknown, 0=free, 100=obstacle
            
        Returns:
            Costmap with LiDAR blind spot marked as free
        """
        if self.blind_spot_radius <= 0:
            return costmap
        
        # Vehicle/LiDAR is at origin (0, 0) in ego frame
        # Create a circular mask for the blind spot
        
        # Vehicle position in grid coordinates
        vehicle_ix = int((0.0 - self.origin_x) / self.resolution)
        vehicle_iy = int((0.0 - self.origin_y) / self.resolution)
        
        # Blind spot radius in grid cells
        radius_cells = self.blind_spot_radius / self.resolution
        
        # Create coordinate grids
        y_indices, x_indices = np.ogrid[:self.size_y, :self.size_x]
        
        # Compute distance from vehicle for each cell
        dist_sq = (x_indices - vehicle_ix)**2 + (y_indices - vehicle_iy)**2
        
        # Circular mask: cells within blind spot radius
        blind_spot_mask = dist_sq <= radius_cells**2
        
        # Only mark unknown cells as free (don't overwrite detected obstacles)
        fill_mask = blind_spot_mask & (costmap == COST_UNKNOWN)
        costmap[fill_mask] = COST_FREE
        
        return costmap

    def _fill_free_space_gaps(self, costmap: np.ndarray) -> np.ndarray:
        """
        Fill gaps in free space caused by sparse LiDAR scan lines.
        
        Uses distance-adaptive morphological closing - larger kernels for cells
        farther from the vehicle since LiDAR scan lines diverge with distance.
        
        Args:
            costmap: 2D array with -1=unknown, 0=free, 100=obstacle
            
        Returns:
            Costmap with filled free space gaps
        """
        result = costmap.copy()
        
        # Compute distance from vehicle (grid center)
        # Vehicle is at origin, which maps to specific grid indices
        vehicle_ix = int(-self.origin_x / self.resolution)
        vehicle_iy = int(-self.origin_y / self.resolution)
        
        # Create distance map (in grid cells)
        y_indices, x_indices = np.ogrid[:self.size_y, :self.size_x]
        dist_from_vehicle = np.sqrt((x_indices - vehicle_ix)**2 + (y_indices - vehicle_iy)**2)
        
        # Convert to meters
        dist_meters = dist_from_vehicle * self.resolution
        
        # Define distance rings with increasing kernel sizes
        # (max_distance_m, kernel_size)
        rings = [
            (5.0, 3),    # 0-5m: small kernel
            (10.0, 5),   # 5-10m: medium kernel
            (20.0, 7),   # 10-20m: larger kernel
            (float('inf'), 9),  # 20m+: largest kernel
        ]
        
        # Create binary mask of free space
        free_mask = (costmap == COST_FREE)
        
        # Process each ring from inside out
        prev_dist = 0.0
        accumulated_free = free_mask.copy()
        
        for max_dist, kernel_size in rings:
            # Create ring mask
            ring_mask = (dist_meters > prev_dist) & (dist_meters <= max_dist)
            
            # Create kernel for this distance
            kernel = np.ones((kernel_size, kernel_size), dtype=bool)
            
            # Apply closing to the accumulated free space
            closed = ndimage.binary_closing(accumulated_free, structure=kernel, iterations=1)
            
            # Dilate to fill unknown gaps
            dilated = ndimage.binary_dilation(closed, structure=kernel, iterations=1)
            
            # Only update cells in this ring that are unknown
            fill_mask = ring_mask & dilated & (result == COST_UNKNOWN)
            result[fill_mask] = COST_FREE
            
            # Update accumulated free for next ring
            accumulated_free = (result == COST_FREE)
            
            prev_dist = max_dist
        
        return result

    def _create_occupancy_grid(self, costmap: np.ndarray, stamp) -> OccupancyGrid:
        """Create OccupancyGrid message from costmap array."""
        msg = OccupancyGrid()
        
        # Header
        msg.header = Header()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        
        # Metadata
        msg.info = MapMetaData()
        msg.info.resolution = self.resolution
        msg.info.width = self.size_x   # X direction (forward)
        msg.info.height = self.size_y  # Y direction (lateral)
        
        # Origin (position of cell [0,0] in frame_id)
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        
        # Data (row-major order, values clamped to int8 range)
        # OccupancyGrid expects values -1 to 100
        msg.data = costmap.flatten().tolist()
        
        return msg


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
