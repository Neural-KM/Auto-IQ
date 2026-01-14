#!/usr/bin/env python3
"""
Visualization Node
==================

ROS2 node for visualizing Auto-IQ data in both CARLA (3D debug drawings)
and RViz (MarkerArrays).

Subscribes to:
- /carla/ego_vehicle/ground_truth/odometry (nav_msgs/Odometry) - Vehicle state
- /planning/trajectory (autoiq_msgs/Trajectory) - Local trajectory
- /planning/global_route (nav_msgs/Path) - Global route
- /control/predicted_trajectory (nav_msgs/Path) - MPC predicted path
- /perception/obstacles (autoiq_msgs/ObstacleArray) - Detected obstacles
- /perception/ground_truth/obstacles (autoiq_msgs/ObstacleArray) - Ground truth obstacles
- /carla/ego_vehicle/ground_truth/camera/semantic/front/image (Image) - Semantic segmentation
- /perception/semantic_pointcloud (PointCloud2) - Labeled pointcloud (XYZ + label)

Publishes:
- /visualization/trajectory (nav_msgs/Path) - Trajectory for RViz
- /visualization/ground_truth_obstacles (MarkerArray) - Ground truth obstacle markers
- /visualization/detected_obstacles (MarkerArray) - Detected obstacle markers
- /visualization/semantic_image (Image) - Colored semantic segmentation
- /visualization/semantic_pointcloud (PointCloud2) - Colored XYZRGB pointcloud

Connects to CARLA for debug drawing (optional).
"""

import math
import struct
import numpy as np
from dataclasses import replace
from typing import List, Tuple

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import Image, PointCloud2, PointField
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA, Header
from geometry_msgs.msg import PoseStamped, PointStamped

from autoiq_msgs.msg import ObstacleArray as ObstacleArrayMsg, Trajectory as TrajectoryMsg
from autoiq_msgs.msg import VehicleControl

# Import common types and converters
from autoiq_common.types import VehicleState, Trajectory, ObstacleArray, Obstacle
from autoiq_common.converters import (
    vehicle_state_from_odometry,
    trajectory_from_ros,
    obstacle_array_from_ros,
)
from autoiq_common.utils import quaternion_to_yaw

from perception.data_classes.semantic_labels import colorize_mask, SEMANTIC_COLOR_LUT

import carla

from visualization.carla_drawer import CarlaDrawer
from visualization.live_2d_visualizer import Live2DVisualizerROS

class VisualizationNode(Node):
    """
    ROS2 Visualization Node.
    
    Subscribes to planning, control, and perception topics and visualizes
    the data in CARLA (3D) and matplotlib (2D).
    """
    
    def __init__(self):
        super().__init__('visualization_node')
        
        # ==================== Parameters ====================
        self.declare_parameter('carla_host', 'localhost')
        self.declare_parameter('carla_port', 2000)
        self.declare_parameter('carla_timeout', 10.0)
        self.declare_parameter('enable_2d_viz', False)
        self.declare_parameter('viz_rate', 10.0)  # Hz
        self.declare_parameter('draw_lifetime', 0.15)  # seconds
        
        # Comma-separated list of objects to draw in CARLA debug view
        # Options: trajectory, global_route, predicted_traj, ground_truth_obstacles, detected_obstacles, speed, control
        self.declare_parameter('carla_debug_draw', 'trajectory,global_route,predicted_traj,speed,control')
        self.declare_parameter('semantic_topic', '/carla/ego_vehicle/ground_truth/camera/semantic/front/image')
        self.declare_parameter('semantic_output_topic', '/visualization/semantic_image')
        self.declare_parameter('labeled_pointcloud_topic', '/perception/semantic_pointcloud')
        self.declare_parameter('colored_pointcloud_topic', '/visualization/semantic_pointcloud')
        
        carla_host = self.get_parameter('carla_host').value
        carla_port = self.get_parameter('carla_port').value
        carla_timeout = self.get_parameter('carla_timeout').value
        self.enable_2d_viz = self.get_parameter('enable_2d_viz').value
        viz_rate = self.get_parameter('viz_rate').value
        self.draw_lifetime = self.get_parameter('draw_lifetime').value
        
        # Parse carla_debug_draw string into a set of items
        carla_debug_draw_str = self.get_parameter('carla_debug_draw').value
        self.carla_draw_items = set()
        if carla_debug_draw_str:
            self.carla_draw_items = set(item.strip() for item in carla_debug_draw_str.split(',') if item.strip())
        self.enable_carla_viz = len(self.carla_draw_items) > 0
        
        semantic_topic = self.get_parameter('semantic_topic').value
        semantic_output_topic = self.get_parameter('semantic_output_topic').value
        labeled_pointcloud_topic = self.get_parameter('labeled_pointcloud_topic').value
        colored_pointcloud_topic = self.get_parameter('colored_pointcloud_topic').value
        
        # ==================== CARLA Connection ====================
        self.carla_drawer = None
        if self.enable_carla_viz:
            try:
                self.get_logger().info(f'Connecting to CARLA at {carla_host}:{carla_port}...')
                client = carla.Client(carla_host, carla_port)
                client.set_timeout(carla_timeout)
                world = client.get_world()
                self.carla_drawer = CarlaDrawer(world)
                self.get_logger().info('CARLA visualization enabled')
            except Exception as e:
                self.get_logger().warn(f'Failed to connect to CARLA: {e}')
                self.get_logger().warn('CARLA visualization disabled')
                self.enable_carla_viz = False
        
        # ==================== 2D Visualizer ====================
        self.viz_2d = None
        if self.enable_2d_viz:
            self.viz_2d = Live2DVisualizerROS(title="Auto-IQ Live 2D", refresh_hz=viz_rate)
            self.viz_2d.start()
            self.get_logger().info('2D matplotlib visualization enabled')
        
        # ==================== Data Storage ====================
        self.vehicle_state: VehicleState = None
        self.target_speed = 0.0
        
        # Vehicle control (steer, throttle, brake)
        self.current_steer = 0.0
        self.current_throttle = 0.0
        self.current_brake = 0.0
        
        # Paths
        self.trajectory = []  # List of (x, y)
        self.global_route = []  # List of (x, y, z)
        self.predicted_trajectory = []  # List of (x, y, yaw, speed)
        self.current_waypoint = None  # Tuple of (x, y, z) or None
        
        # Obstacles (ROS world coordinates)
        self.ground_truth_obstacles: List[Obstacle] = []  # List of Obstacle dataclasses
        self.detected_obstacles: List[Obstacle] = []  # List of Obstacle dataclasses
        
        # ==================== Publishers for RViz ====================
        # Publish MarkerArrays for obstacles visualization
        self.gt_obstacles_marker_pub = self.create_publisher(
            MarkerArray, '/visualization/ground_truth_obstacles', 10)
        self.detected_obstacles_marker_pub = self.create_publisher(
            MarkerArray, '/visualization/detected_obstacles', 10)
        # Publish trajectory as Path for RViz
        self.trajectory_viz_pub = self.create_publisher(
            Path, '/visualization/trajectory', 10)
        # Publish current waypoint marker for RViz
        self.current_waypoint_marker_pub = self.create_publisher(
            Marker, '/visualization/current_waypoint', 10)
        # Publish colored semantic image
        self.semantic_image_pub = self.create_publisher(
            Image, semantic_output_topic, 10)
        # Publish colored semantic pointcloud
        self.colored_pointcloud_pub = self.create_publisher(
            PointCloud2, colored_pointcloud_topic, 10)
        
        # ==================== Subscribers ====================
        self.odom_sub = self.create_subscription(
            Odometry, '/carla/ego_vehicle/ground_truth/odometry',
            self.odom_callback, 10)
        
        self.trajectory_sub = self.create_subscription(
            TrajectoryMsg, '/planning/trajectory',
            self.trajectory_callback, 10)
        
        self.global_route_sub = self.create_subscription(
            Path, '/planning/global_route',
            self.global_route_callback, 10)
        
        self.predicted_sub = self.create_subscription(
            Path, '/control/predicted_trajectory',
            self.predicted_callback, 10)
        
        self.gt_obstacles_sub = self.create_subscription(
            ObstacleArrayMsg, '/perception/ground_truth/obstacles',
            self.ground_truth_obstacles_callback, 10)
        self.get_logger().info('Subscribed to ground truth obstacles')
        
        self.detected_obstacles_sub = self.create_subscription(
            ObstacleArrayMsg, '/perception/obstacles',
            self.detected_obstacles_callback, 10)
        self.get_logger().info('Subscribed to detected obstacles')
        
        self.semantic_sub = self.create_subscription(
            Image, semantic_topic,
            self.semantic_callback, 10)
        self.get_logger().info(f'Subscribed to semantic image: {semantic_topic}')
        
        self.labeled_pointcloud_sub = self.create_subscription(
            PointCloud2, labeled_pointcloud_topic,
            self.labeled_pointcloud_callback, 10)
        self.get_logger().info(f'Subscribed to labeled pointcloud: {labeled_pointcloud_topic}')
        
        self.current_waypoint_sub = self.create_subscription(
            PointStamped, '/planning/current_waypoint',
            self.current_waypoint_callback, 10)
        
        self.control_sub = self.create_subscription(
            VehicleControl, '/carla/ego_vehicle/control',
            self.control_callback, 10)
        self.get_logger().info('Subscribed to vehicle control')
        
        # ==================== Visualization Timer ====================
        viz_period = 1.0 / viz_rate
        self.viz_timer = self.create_timer(viz_period, self.visualization_loop)
        
        self.get_logger().info(f'Visualization Node initialized ({viz_rate}Hz)')
    
    # ==================== Callbacks ====================
    
    def odom_callback(self, msg: Odometry):
        """
        Update vehicle state from odometry.
        
        Converts Odometry message to VehicleState dataclass.
        """
        self.vehicle_state = vehicle_state_from_odometry(msg)
    
    def trajectory_callback(self, msg: TrajectoryMsg):
        """
        Receive local trajectory from planning node.
        
        Converts TrajectoryMsg to Trajectory dataclass.
        """
        traj = trajectory_from_ros(msg)
        waypoints = []
        
        # Extract target speed from first point (if available)
        if traj.points:
            self.target_speed = traj.points[0].speed
            
        for point in traj.points:
            waypoints.append((point.x, point.y))
        self.trajectory = waypoints
        
        # Publish as Path for RViz visualization
        self._publish_trajectory_path(traj)
    
    def global_route_callback(self, msg: Path):
        """Receive global route from planning node."""
        waypoints = []
        for pose in msg.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            z = pose.pose.position.z
            waypoints.append((x, y, z))
        self.global_route = waypoints
    
    def current_waypoint_callback(self, msg: PointStamped):
        """Receive current waypoint from planning node."""
        self.current_waypoint = (
            msg.point.x,
            msg.point.y,
            msg.point.z
        )
        # Publish marker for RViz
        self._publish_current_waypoint_marker()
    
    def predicted_callback(self, msg: Path):
        """Receive MPC predicted trajectory from control node."""
        states = []
        for pose in msg.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            
            # Extract yaw from quaternion
            quat = pose.pose.orientation
            yaw = quaternion_to_yaw(quat)
            
            # Speed not available in Path, use 0
            states.append((x, y, yaw, 0.0))
        self.predicted_trajectory = states
    
    def control_callback(self, msg: VehicleControl):
        """Receive vehicle control command from control node."""
        self.current_steer = msg.steer
        self.current_throttle = msg.throttle
        self.current_brake = msg.brake
    
    def semantic_callback(self, msg: Image):
        """Receive semantic segmentation image and republish colored version."""
        # Parse semantic class IDs from image
        if msg.encoding == 'bgra8':
            # CARLA semantic camera: class ID in R channel (index 2 for BGRA)
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 4)
            class_ids = img[:, :, 2]
        elif msg.encoding == 'rgb8':
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3)
            class_ids = img[:, :, 0]
        elif msg.encoding == 'mono8':
            class_ids = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width)
        else:
            return
        
        # Colorize using semantic labels palette
        colored = colorize_mask(class_ids)
        
        # Publish colored image
        out_msg = Image()
        out_msg.header = msg.header
        out_msg.height = msg.height
        out_msg.width = msg.width
        out_msg.encoding = 'rgb8'
        out_msg.step = msg.width * 3
        out_msg.data = colored.tobytes()
        
        self.semantic_image_pub.publish(out_msg)
    
    def labeled_pointcloud_callback(self, msg: PointCloud2):
        """Convert labeled pointcloud (XYZ+label) to colored pointcloud (XYZRGB)."""
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
        
        # Parse points
        point_step = msg.point_step
        n_points = msg.width * msg.height
        data = np.frombuffer(msg.data, dtype=np.uint8)
        
        # Extract XYZ and labels
        points = np.zeros((n_points, 3), dtype=np.float32)
        labels = np.zeros(n_points, dtype=np.uint32)
        
        for i in range(n_points):
            offset = i * point_step
            points[i, 0] = struct.unpack('f', data[offset + x_offset:offset + x_offset + 4])[0]
            points[i, 1] = struct.unpack('f', data[offset + y_offset:offset + y_offset + 4])[0]
            points[i, 2] = struct.unpack('f', data[offset + z_offset:offset + z_offset + 4])[0]
            labels[i] = struct.unpack('I', data[offset + label_offset:offset + label_offset + 4])[0]
        
        # Get colors from labels using LUT
        colors = SEMANTIC_COLOR_LUT[np.clip(labels, 0, 255)]
        
        # Create XYZRGB pointcloud
        colored_msg = self._create_colored_pointcloud2(points, colors, msg.header)
        self.colored_pointcloud_pub.publish(colored_msg)
    
    def _create_colored_pointcloud2(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        header: Header
    ) -> PointCloud2:
        """Create XYZRGB PointCloud2 message."""
        n_points = len(points)
        
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        
        point_step = 16
        data = bytearray(n_points * point_step)
        
        for i in range(n_points):
            offset = i * point_step
            struct.pack_into('fff', data, offset,
                             points[i, 0], points[i, 1], points[i, 2])
            
            r, g, b = int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])
            rgb_packed = (r << 16) | (g << 8) | b
            rgb_float = struct.unpack('f', struct.pack('I', rgb_packed))[0]
            struct.pack_into('f', data, offset + 12, rgb_float)
        
        out = PointCloud2()
        out.header = header
        out.height = 1
        out.width = n_points
        out.fields = fields
        out.is_bigendian = False
        out.point_step = point_step
        out.row_step = point_step * n_points
        out.data = bytes(data)
        out.is_dense = True
        
        return out
    
    def _transform_obstacles_to_world(self, obstacle_array: ObstacleArray) -> List[Obstacle]:
        """
        Transform obstacles from ego frame to world frame.
        
        Args:
            obstacle_array: ObstacleArray dataclass (obstacles in ego frame)
            
        Returns:
            List of Obstacle dataclasses in world frame
        """
        if self.vehicle_state is None:
            return []
        
        result = []
        vx, vy, vyaw = self.vehicle_state.x, self.vehicle_state.y, self.vehicle_state.yaw
        c, s = math.cos(vyaw), math.sin(vyaw)
        
        for obs in obstacle_array.obstacles:
            # Obstacle position is in ego frame (x=forward, y=left in ROS)
            # Transform to world coordinates (ROS frame)
            world_x = vx + obs.x * c - obs.y * s
            world_y = vy + obs.x * s + obs.y * c
            world_yaw = vyaw + obs.yaw
            
            # Create new Obstacle with world coordinates
            world_obs = replace(obs, x=world_x, y=world_y, yaw=world_yaw)
            result.append(world_obs)
        
        return result
    
    def _obstacles_to_marker_array(self, obstacles: List[Obstacle], namespace: str, 
                                    color: Tuple[float, float, float]) -> MarkerArray:
        """Convert obstacle list to MarkerArray for RViz."""
        marker_array = MarkerArray()
        
        # Clear previous markers
        clear_marker = Marker()
        clear_marker.header.frame_id = 'map'
        clear_marker.header.stamp = self.get_clock().now().to_msg()
        clear_marker.ns = namespace
        clear_marker.id = 0
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        
        for i, obs in enumerate(obstacles):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = namespace
            marker.id = i + 1
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            # Obstacles are in ROS world coordinates, use directly
            marker.pose.position.x = obs.x
            marker.pose.position.y = obs.y
            marker.pose.position.z = obs.height / 2
            
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = math.sin(obs.yaw / 2)
            marker.pose.orientation.w = math.cos(obs.yaw / 2)
            
            marker.scale.x = obs.length
            marker.scale.y = obs.width
            marker.scale.z = obs.height
            
            marker.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=0.7)
            marker.lifetime.nanosec = 200000000
            
            marker_array.markers.append(marker)
            
            # Add text label
            text_marker = Marker()
            text_marker.header = marker.header
            text_marker.ns = f'{namespace}_labels'
            text_marker.id = i + 1
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = obs.x
            text_marker.pose.position.y = obs.y
            text_marker.pose.position.z = obs.height + 0.5
            text_marker.text = f"{obs.type}"
            text_marker.scale.z = 0.5
            text_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.9)
            text_marker.lifetime.nanosec = 200000000
            marker_array.markers.append(text_marker)
        
        return marker_array
    
    def ground_truth_obstacles_callback(self, msg: ObstacleArrayMsg):
        """
        Receive ground truth obstacles and convert to markers.
        
        Converts ObstacleArrayMsg to ObstacleArray dataclass.
        """
        obstacle_array = obstacle_array_from_ros(msg)
        self.ground_truth_obstacles = self._transform_obstacles_to_world(obstacle_array)
        
        # Publish MarkerArray for RViz
        marker_array = self._obstacles_to_marker_array(
            self.ground_truth_obstacles, 'ground_truth_obstacles', (0.0, 1.0, 0.0))
        self.gt_obstacles_marker_pub.publish(marker_array)
    
    def detected_obstacles_callback(self, msg: ObstacleArrayMsg):
        """
        Receive detected obstacles and convert to markers.
        
        Converts ObstacleArrayMsg to ObstacleArray dataclass.
        """
        obstacle_array = obstacle_array_from_ros(msg)
        self.detected_obstacles = self._transform_obstacles_to_world(obstacle_array)
        
        # Publish MarkerArray for RViz
        marker_array = self._obstacles_to_marker_array(
            self.detected_obstacles, 'detected_obstacles', (1.0, 1.0, 0.0))
        self.detected_obstacles_marker_pub.publish(marker_array)
    
    def _publish_trajectory_path(self, traj: Trajectory):
        """Convert Trajectory dataclass to Path for RViz visualization."""
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        for point in traj.points:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = point.x
            pose.pose.position.y = point.y
            pose.pose.position.z = 0.0  # Flat for visualization
            # Set orientation from yaw
            pose.pose.orientation.z = math.sin(point.yaw / 2)
            pose.pose.orientation.w = math.cos(point.yaw / 2)
            path_msg.poses.append(pose)
        
        self.trajectory_viz_pub.publish(path_msg)
    
    def _publish_current_waypoint_marker(self):
        """Publish current waypoint as a sphere marker for RViz."""
        if self.current_waypoint is None:
            return
        
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'current_waypoint'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        # Position in ROS world coordinates
        marker.pose.position.x = self.current_waypoint[0]
        marker.pose.position.y = self.current_waypoint[1]
        marker.pose.position.z = self.current_waypoint[2] + 1.0  # Lift above road
        
        marker.pose.orientation.w = 1.0
        
        # Make it a visible sphere
        marker.scale.x = 2.0
        marker.scale.y = 2.0
        marker.scale.z = 2.0
        
        # Bright magenta color for visibility
        marker.color = ColorRGBA(r=1.0, g=0.0, b=1.0, a=0.9)
        marker.lifetime.nanosec = 200000000  # 200ms
        
        self.current_waypoint_marker_pub.publish(marker)
    
    # ==================== Visualization Loop ====================
    
    def visualization_loop(self):
        """Main visualization loop - draws in CARLA and updates 2D viz."""
        if self.vehicle_state is None:
            return
        
        # ==================== CARLA Debug Drawing ====================
        if self.enable_carla_viz and self.carla_drawer:
            self._draw_carla()
        
        # ==================== 2D Matplotlib Visualizer ====================
        if self.enable_2d_viz and self.viz_2d:
            self._update_2d_viz()
    
    def _draw_carla(self):
        """Draw debug shapes in CARLA (applies Y-flip for CARLA's left-handed coords)."""
        lt = self.draw_lifetime
        items = self.carla_draw_items
        
        # Draw global route (flip Y for CARLA)
        if 'global_route' in items and self.global_route:
            route_carla = [(x, -y, z) for x, y, z in self.global_route]
            self.carla_drawer.draw_global_route(route_carla, lifetime=lt)
        
        # Draw local trajectory (flip Y for CARLA)
        if 'trajectory' in items and self.trajectory:
            traj_carla = [(x, -y) for x, y in self.trajectory]
            self.carla_drawer.draw_trajectory(traj_carla, lifetime=lt)
        
        # Draw MPC predicted trajectory (flip Y and yaw for CARLA)
        if 'predicted_traj' in items and self.predicted_trajectory:
            pred_carla = [(x, -y, -yaw, speed) for x, y, yaw, speed in self.predicted_trajectory]
            self.carla_drawer.draw_predicted_trajectory(pred_carla, lifetime=lt)
        
        # Draw speed indicator (flip Y for CARLA)
        if 'speed' in items and self.vehicle_state is not None:
            self.carla_drawer.draw_speed_indicator(
                self.vehicle_state.x, -self.vehicle_state.y,
                self.vehicle_state.speed * 3.6,  # Convert to km/h
                self.target_speed * 3.6,   # Target speed from trajectory
                lifetime=lt
            )
        
        # Ground truth obstacles (green boxes, flip Y and yaw for CARLA)
        if 'ground_truth_obstacles' in items:
            for obs in self.ground_truth_obstacles:
                self.carla_drawer.draw_obstacle_box(
                    obs.x, -obs.y, -obs.yaw,
                    obs.length, obs.width, obs.height,
                    is_vehicle=(obs.type == 'vehicle'),
                    lifetime=lt,
                    color=(0, 255, 0)  # Green for ground truth
                )
        
        # Detected obstacles (yellow boxes, flip Y and yaw for CARLA)
        if 'detected_obstacles' in items:
            for obs in self.detected_obstacles:
                self.carla_drawer.draw_obstacle_box(
                    obs.x, -obs.y, -obs.yaw,
                    obs.length, obs.width, obs.height,
                    is_vehicle=(obs.type == 'vehicle'),
                    lifetime=lt,
                    color=(255, 255, 0)  # Yellow for detected
                )
    
    def _update_2d_viz(self):
        """Update 2D matplotlib visualizer."""
        # Update vehicle
        if self.vehicle_state is not None:
            self.viz_2d.update_vehicle(
                self.vehicle_state.x, self.vehicle_state.y, 
                self.vehicle_state.yaw or 0.0, self.vehicle_state.speed
            )
        
        # Update trajectory
        if self.trajectory:
            self.viz_2d.update_trajectory(self.trajectory)
        
        # Update global route
        if self.global_route:
            route_2d = [(p[0], p[1]) for p in self.global_route]
            self.viz_2d.update_global_route(route_2d)
        
        # Update predicted trajectory
        if self.predicted_trajectory:
            self.viz_2d.update_predicted_trajectory(self.predicted_trajectory)
        
        # Update obstacles - combine both lists for 2D viz
        all_obstacles = []
        if 'ground_truth_obstacles' in self.carla_draw_items:
            all_obstacles.extend(self.ground_truth_obstacles)
        if 'detected_obstacles' in self.carla_draw_items:
            all_obstacles.extend(self.detected_obstacles)
        if all_obstacles:
            # Convert to dicts for 2D viz (legacy interface)
            obs_dicts = [{'x': o.x, 'y': o.y, 'yaw': o.yaw, 'length': o.length, 
                          'width': o.width, 'type': o.type} for o in all_obstacles]
            self.viz_2d.update_obstacles(obs_dicts)
    
    def destroy_node(self):
        """Cleanup on shutdown."""
        if self.viz_2d:
            self.viz_2d.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VisualizationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
