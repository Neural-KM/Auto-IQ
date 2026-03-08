#!/usr/bin/env python3
"""
CARLA Bridge Node - Deterministic Tick-Wait-Publish Architecture
=================================================================

ROS2 node that bridges CARLA simulator to ROS2 using a strictly deterministic
synchronization model. The node acts as the simulation time source (/clock)
and guarantees that all published messages for a given simulation step share
the exact same timestamp and frame ID.

Architecture:
1. Simulation Time Source: Bridge publishes to /clock, all messages use CARLA time
2. Tick-Wait-Publish Loop: 
   - Apply vehicle control
   - world.tick() to advance simulation
   - Wait for all sensor data matching current frame
   - Publish all ROS messages with synchronized timestamp
3. Sensor callbacks are pure producers (push to queues, no publishing)
4. TF & Odometry share the synchronized timestamp

"""

import json
import math
import sys
import os
import time
import threading
from queue import Queue, Empty
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

# ROS2 message types
from std_msgs.msg import Header
from builtin_interfaces.msg import Time as TimeMsg
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image, PointCloud2, CameraInfo, NavSatFix, Imu
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from autoiq_msgs.msg import VehicleControl

from cv_bridge import CvBridge
import carla

# Local converters
from autoiq_bridge.converters import (
    carla_image_to_ros,
    carla_depth_to_ros,
    carla_semantic_to_ros,
    carla_lidar_to_ros,
    carla_semantic_lidar_to_ros,
    carla_vehicle_to_odometry,
    carla_transform_to_ros,
    create_camera_info,
    create_header,
    euler_to_quaternion
)


@dataclass
class SensorQueueItem:
    """Container for sensor data with frame information."""
    frame: int
    timestamp: float
    data: Any


class CarlaBridgeNode(Node):
    """
    ROS2 Node that bridges CARLA simulator to ROS2 with deterministic synchronization.
    
    Uses a Tick-Wait-Publish loop to ensure all sensor data for a given
    simulation frame is published with the exact same timestamp.
    
    Publishes:
        - /clock (simulation time source)
        - Camera images (RGB, Depth, Semantic)
        - LiDAR point clouds
        - Vehicle odometry
        - TF transforms
        
    Subscribes:
        - Vehicle control commands
    """
    
    def __init__(self):
        super().__init__('carla_bridge_node')
        
        # ==================== Parameters ====================
        self.declare_parameter('host', 'localhost')
        self.declare_parameter('port', 2000)
        self.declare_parameter('timeout', 10.0)
        self.declare_parameter('fixed_delta_seconds', 0.05)
        self.declare_parameter('objects_file', '')
        self.declare_parameter('spawn_point_index', 50)
        self.declare_parameter('vehicle_type', 'vehicle.tesla.model3')
        self.declare_parameter('sensor_timeout', 2.0)  # Seconds to wait for sensor data
        
        # Get parameters
        self.host = self.get_parameter('host').value
        self.port = self.get_parameter('port').value
        self.timeout = self.get_parameter('timeout').value
        self.fixed_delta_seconds = self.get_parameter('fixed_delta_seconds').value
        self.objects_file = self.get_parameter('objects_file').value
        self.spawn_point_index = self.get_parameter('spawn_point_index').value
        self.vehicle_type = self.get_parameter('vehicle_type').value
        self.sensor_timeout = self.get_parameter('sensor_timeout').value

        # ==================== CARLA Objects ====================
        self.client = None
        self.world = None
        self.vehicle = None
        self.sensors: Dict[str, dict] = {}  # {sensor_id: {'actor', 'publisher', 'queue', ...}}
        
        # ==================== Synchronization ====================
        self.current_frame = 0
        self.running = True
        
        # Thread pool for parallel sensor publishing
        self.publish_executor = ThreadPoolExecutor(max_workers=8)
        
        # Cached CameraInfo messages (avoid recreating each frame)
        self.cached_camera_info: Dict[str, CameraInfo] = {}
        
        # ==================== CV Bridge ====================
        self.cv_bridge = CvBridge()
        
        # ==================== TF Broadcasters ====================
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        
        # ==================== QoS Profiles ====================
        self.sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Clock uses RELIABLE for /clock topic (sim time source)
        self.clock_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # ==================== Publishers ====================
        self.clock_pub = self.create_publisher(Clock, '/clock', self.clock_qos)
        self.odom_pub = self.create_publisher(Odometry, '/carla/ego_vehicle/ground_truth/odometry', 10)
        
        # ==================== Subscribers ====================
        self.control_sub = self.create_subscription(
            VehicleControl,
            '/carla/ego_vehicle/control',
            self.control_callback,
            1
        )
        
        # ==================== Control State ====================
        self.current_control = carla.VehicleControl()
        self.control_lock = threading.Lock()
        
        # ==================== Connect and Setup ====================
        self.connect_to_carla()
        self.spawn_vehicle_and_sensors()
        self.enable_synchronous_mode()
        
        # Publish static transforms once
        self.publish_static_sensor_tfs()
        
        # ==================== Main Loop Thread ====================
        # Run the tick-wait-publish loop in a separate thread
        self.main_loop_thread = threading.Thread(target=self.tick_wait_publish_loop, daemon=True)
        self.main_loop_thread.start()
        
        self.get_logger().info('CARLA Bridge Node initialized (Tick-Wait-Publish mode)')
    
    # ==================== CARLA Connection ====================
    
    def connect_to_carla(self):
        """Connect to CARLA server."""
        self.get_logger().info(f'Connecting to CARLA at {self.host}:{self.port}...')
        
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)
        
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        
        self.get_logger().info(f'Connected to CARLA, map: {self.map.name}')
        
        # Disable synchronous mode first to ensure we can spawn actors
        settings = self.world.get_settings()
        if settings.synchronous_mode:
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
            self.get_logger().info('Disabled existing synchronous mode for spawning')
    
    def enable_synchronous_mode(self):
        """Enable synchronous mode after spawning actors."""
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.fixed_delta_seconds
        self.world.apply_settings(settings)
        
        # Initial tick to get things started
        self.current_frame = self.world.tick()
        self.get_logger().info(f'Synchronous mode enabled, sim dt between every tick={self.fixed_delta_seconds}s')
    
    # ==================== Spawning ====================
    
    def spawn_vehicle_and_sensors(self):
        """Spawn vehicle and attach sensors based on config."""
        # Load objects config if provided
        if self.objects_file and os.path.exists(self.objects_file):
            with open(self.objects_file, 'r') as f:
                config = json.load(f)
            self.get_logger().info(f'Loaded objects config from {self.objects_file}')
            
            # Find ego vehicle in config
            ego_config = None
            for obj in config.get('objects', []):
                if obj.get('id') == 'ego_vehicle':
                    ego_config = obj
                    break
            
            if ego_config:
                self.vehicle_type = ego_config.get('type', self.vehicle_type)
                sensor_configs = ego_config.get('sensors', [])
            else:
                sensor_configs = []
        else:
            sensor_configs = []
            self.get_logger().warn('No objects.json provided, spawning vehicle with no sensors')
        
        # Spawn vehicle
        self.spawn_vehicle()
        
        # Spawn sensors
        for sensor_cfg in sensor_configs:
            self.spawn_sensor(sensor_cfg)
    
    def spawn_vehicle(self):
        """Spawn the ego vehicle."""
        blueprint_library = self.world.get_blueprint_library()
        
        # Find vehicle blueprint
        vehicle_bp = blueprint_library.find(self.vehicle_type)
        if vehicle_bp is None:
            vehicle_bp = blueprint_library.filter('vehicle.*')[0]
            self.get_logger().warn(f'Vehicle {self.vehicle_type} not found, using {vehicle_bp.id}')
        
        # Get spawn point
        spawn_points = self.map.get_spawn_points()
        if self.spawn_point_index < len(spawn_points):
            spawn_point = spawn_points[self.spawn_point_index]
        else:
            spawn_point = spawn_points[0]
            self.get_logger().warn(f'Spawn index {self.spawn_point_index} out of range, using 0')
        
        vehicle_bp.set_attribute('role_name', 'hero')
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.get_logger().info(f'Spawned vehicle: {self.vehicle.type_id} at spawn point {self.spawn_point_index}')
    
    def spawn_sensor(self, config: dict):
        """
        Spawn a sensor, setup its queue and publisher.
        
        Sensor callbacks are now pure producers - they only push data to queues.
        """
        sensor_type = config['type']
        sensor_id = config['id']
        spawn_point = config.get('spawn_point', {'x': 0, 'y': 0, 'z': 0, 'roll': 0, 'pitch': 0, 'yaw': 0})
        
        blueprint_library = self.world.get_blueprint_library()
        bp = blueprint_library.find(sensor_type)
        
        if bp is None:
            self.get_logger().error(f'Sensor blueprint not found: {sensor_type}')
            return
        
        # Set sensor attributes based on type
        if 'sensor.camera' in sensor_type:
            for attr in ['image_size_x', 'image_size_y', 'fov', 'sensor_tick']:
                if attr in config:
                    bp.set_attribute(attr, str(config[attr]))
                    
        elif sensor_type in ('sensor.lidar.ray_cast', 'sensor.lidar.ray_cast_semantic'):
            for attr in ['channels', 'range', 'points_per_second', 'rotation_frequency', 
                         'upper_fov', 'lower_fov', 'horizontal_fov', 'sensor_tick',
                         'atmosphere_attenuation_rate', 'dropoff_general_rate',
                         'dropoff_intensity_limit', 'dropoff_zero_intensity',
                         'noise_stddev']:
                if attr in config:
                    bp.set_attribute(attr, str(config[attr]))
        
        # Create transform
        transform = carla.Transform(
            carla.Location(x=spawn_point['x'], y=spawn_point['y'], z=spawn_point['z']),
            carla.Rotation(pitch=spawn_point['pitch'], yaw=spawn_point['yaw'], roll=spawn_point['roll'])
        )
        
        # Spawn sensor attached to vehicle
        sensor_actor = self.world.spawn_actor(bp, transform, attach_to=self.vehicle)
        
        # Determine frame_id based on sensor type
        frame_id = self._get_frame_id(sensor_type, sensor_id)
        
        # Create sensor data structure with thread-safe queue
        sensor_data = {
            'actor': sensor_actor,
            'config': config,
            'frame_id': frame_id,
            'type': sensor_type,
            'queue': Queue(maxsize=10),
        }
        
        # Create publishers based on sensor type
        self._setup_sensor_publishers(sensor_data, sensor_type, sensor_id)
        
        # Setup callback (pure producer - just pushes to queue)
        self._setup_sensor_callback(sensor_actor, sensor_type, sensor_data['queue'])
        
        self.sensors[frame_id] = sensor_data
        self.get_logger().info(f'Spawned sensor: {sensor_type} as {sensor_id} (frame: {frame_id})')
    
    def _get_frame_id(self, sensor_type: str, sensor_id: str) -> str:
        """Generate frame_id for a sensor type."""
        if 'sensor.camera.rgb' in sensor_type:
            return f'ego_vehicle/sensor/camera/rgb/{sensor_id}'
        elif 'sensor.camera.depth' in sensor_type:
            return f'ego_vehicle/sensor/camera/depth/{sensor_id}'
        elif 'sensor.camera.semantic_segmentation' in sensor_type:
            return f'ego_vehicle/ground_truth/camera/semantic/{sensor_id}'
        elif sensor_type == 'sensor.lidar.ray_cast':
            return f'ego_vehicle/sensor/lidar/{sensor_id}'
        elif sensor_type == 'sensor.lidar.ray_cast_semantic':
            return f'ego_vehicle/ground_truth/lidar/{sensor_id}'
        elif sensor_type == 'sensor.other.gnss':
            return f'ego_vehicle/sensor/gnss/{sensor_id}'
        elif sensor_type == 'sensor.other.imu':
            return f'ego_vehicle/sensor/imu/{sensor_id}'
        else:
            return f'ego_vehicle/{sensor_id}'
    
    def _setup_sensor_publishers(self, sensor_data: dict, sensor_type: str, sensor_id: str):
        """Create publishers for a sensor."""
        if 'sensor.camera.rgb' in sensor_type:
            sensor_data['image_pub'] = self.create_publisher(
                Image, f'/carla/ego_vehicle/sensor/camera/rgb/{sensor_id}/image', self.sensor_qos)
            sensor_data['info_pub'] = self.create_publisher(
                CameraInfo, f'/carla/ego_vehicle/sensor/camera/rgb/{sensor_id}/camera_info', self.sensor_qos)
            
        elif 'sensor.camera.depth' in sensor_type:
            sensor_data['image_pub'] = self.create_publisher(
                Image, f'/carla/ego_vehicle/sensor/camera/depth/{sensor_id}/image', self.sensor_qos)
            sensor_data['info_pub'] = self.create_publisher(
                CameraInfo, f'/carla/ego_vehicle/sensor/camera/depth/{sensor_id}/camera_info', self.sensor_qos)
            
        elif 'sensor.camera.semantic_segmentation' in sensor_type:
            sensor_data['image_pub'] = self.create_publisher(
                Image, f'/carla/ego_vehicle/ground_truth/camera/semantic/{sensor_id}/image', self.sensor_qos)
            sensor_data['info_pub'] = self.create_publisher(
                CameraInfo, f'/carla/ego_vehicle/ground_truth/camera/semantic/{sensor_id}/camera_info', self.sensor_qos)
            
        elif sensor_type == 'sensor.lidar.ray_cast':
            sensor_data['cloud_pub'] = self.create_publisher(
                PointCloud2, f'/carla/ego_vehicle/sensor/lidar/{sensor_id}/point_cloud', self.sensor_qos)
                
        elif sensor_type == 'sensor.lidar.ray_cast_semantic':
            sensor_data['cloud_pub'] = self.create_publisher(
                PointCloud2, f'/carla/ego_vehicle/ground_truth/lidar/{sensor_id}/point_cloud', self.sensor_qos)
            
        elif sensor_type == 'sensor.other.gnss':
            sensor_data['gnss_pub'] = self.create_publisher(
                NavSatFix, f'/carla/ego_vehicle/sensor/gnss/{sensor_id}/fix', self.sensor_qos)
            
        elif sensor_type == 'sensor.other.imu':
            sensor_data['imu_pub'] = self.create_publisher(
                Imu, f'/carla/ego_vehicle/sensor/imu/{sensor_id}/data', self.sensor_qos)
    
    def _setup_sensor_callback(self, sensor_actor, sensor_type: str, queue: Queue):
        """Setup sensor callback as pure producer (pushes to queue only)."""
        def producer_callback(data):
            # Pure producer: just push raw data with frame info to queue
            item = SensorQueueItem(
                frame=data.frame,
                timestamp=data.timestamp,
                data=data
            )
            try:
                queue.put_nowait(item)
            except:
                # Queue full - drop oldest and add new
                try:
                    queue.get_nowait()
                    queue.put_nowait(item)
                except:
                    pass
        
        sensor_actor.listen(producer_callback)
    
    # ==================== Time Utilities ====================
    
    def carla_timestamp_to_ros(self, carla_seconds: float) -> TimeMsg:
        """Convert CARLA timestamp (seconds) to ROS Time message."""
        sec = int(carla_seconds)
        nanosec = int((carla_seconds - sec) * 1e9)
        
        time_msg = TimeMsg()
        time_msg.sec = sec
        time_msg.nanosec = nanosec
        return time_msg
    
    def create_synchronized_header(self, frame_id: str, ros_time: TimeMsg) -> Header:
        """Create a Header with synchronized timestamp."""
        header = Header()
        header.stamp = ros_time
        header.frame_id = frame_id
        return header
    
    # ==================== Main Tick-Wait-Publish Loop ====================
    
    def tick_wait_publish_loop(self):
        """
        Main synchronization loop: Tick-Wait-Publish.
        
        1. Apply vehicle control
        2. Tick the world (advances simulation)
        3. Wait for all sensor data matching current frame
        4. Publish all ROS messages with synchronized timestamp
        """
        self.get_logger().info('Starting Tick-Wait-Publish loop')
        
        while self.running and rclpy.ok():
            time_start = time.time()
            try:
                # ==================== 1. CONTROL ====================
                if self.vehicle is not None and self.vehicle.is_alive:
                    with self.control_lock:
                        self.vehicle.apply_control(self.current_control)
                
                # ==================== 2. TICK ====================
                self.current_frame = self.world.tick()
                
                # Get synchronized timestamp from world snapshot
                snapshot = self.world.get_snapshot()
                carla_time = snapshot.timestamp.elapsed_seconds
                ros_time = self.carla_timestamp_to_ros(carla_time)
                
                # ==================== 3. PUBLISH CLOCK ====================
                clock_msg = Clock()
                clock_msg.clock = ros_time
                self.clock_pub.publish(clock_msg)
                
                # ==================== 4. SYNC BLOCK - Wait for sensors ====================
                time_start_wait = time.time()
                sensor_data_packet = self.wait_for_sensors(self.current_frame)
                self.get_logger().info(f'sensor data collection time: {time.time() - time_start_wait:.3f}s', throttle_duration_sec=10.0)
                # ==================== 5. PUBLISH ====================
                time_start_publish = time.time()
                # Publish TF and Odometry first (they are always available)
                self.publish_tf_synchronized(ros_time)
                self.publish_odometry_synchronized(ros_time)
                
                # Publish sensor data with synchronized timestamp
                self.publish_sensor_data(sensor_data_packet, ros_time)
                self.get_logger().info(f'sensor data+odometry+tf publish time: {time.time() - time_start_publish:.3f}s', throttle_duration_sec=10.0)
                self.get_logger().info(f'Tick time: {time.time() - time_start:.3f}s, Hz: {1.0 / (time.time() - time_start):.2f}, Frame: {self.current_frame}', throttle_duration_sec=10.0)
            except Exception as e:
                if self.running:
                    self.get_logger().error(f'Error in tick loop: {e}')
                    import traceback
                    traceback.print_exc()
    
    def wait_for_sensors(self, target_frame: int) -> Dict[str, Any]:
        """
        Wait for all active sensors to return data matching target_frame.
        
        Uses drain logic: discard stale data (frame < target_frame),
        accept matching data (frame == target_frame).
        
        Returns:
            Dict mapping frame_id to raw CARLA sensor data
        """
        sensor_data = {}
        wait_start = time.time()
        deadline = wait_start + self.sensor_timeout
        
        # Track per-sensor wait times
        sensor_wait_start = {frame_id: wait_start for frame_id in self.sensors.keys()}
        sensor_wait_times = {}
        
        sensors_remaining = set(self.sensors.keys())
        
        while sensors_remaining and time.time() < deadline:
            for frame_id in list(sensors_remaining):
                queue = self.sensors[frame_id]['queue']
                
                try:
                    # Non-blocking get with small timeout
                    item: SensorQueueItem = queue.get(timeout=0.001)
                    
                    if item.frame < target_frame:
                        # Stale data - discard and try again
                        continue
                    elif item.frame == target_frame:
                        # Matching data - accept
                        sensor_data[frame_id] = item.data
                        sensor_wait_times[frame_id] = time.time() - sensor_wait_start[frame_id]
                        sensors_remaining.discard(frame_id)
                    else:
                        # Future data (shouldn't happen in sync mode) - put back
                        queue.put(item)
                        
                except Empty:
                    continue
        
        # Log per-sensor wait times
        if sensor_wait_times:
            timing_str = ', '.join(f'{"/".join(fid.split("/")[-3:])}:{t*1000:.1f}ms' for fid, t in sorted(sensor_wait_times.items(), key=lambda x: -x[1]))
            self.get_logger().info(f'Sensor wait times: {timing_str}', throttle_duration_sec=10.0)
        
        if sensors_remaining:
            self.get_logger().warn(
                f'Sensor timeout for frame {target_frame}: missing {sensors_remaining}'
            )
        
        return sensor_data
    
    # ==================== Synchronized Publishing ====================
    
    def publish_tf_synchronized(self, ros_time: TimeMsg):
        """Publish TF transform with synchronized timestamp."""
        if self.vehicle is None or not self.vehicle.is_alive:
            return
        
        transform = self.vehicle.get_transform()
        
        # Map -> ego_vehicle
        tf_msg = TransformStamped()
        tf_msg.header = self.create_synchronized_header('map', ros_time)
        tf_msg.child_frame_id = 'ego_vehicle'
        
        # Position (flip Y for ROS)
        tf_msg.transform.translation.x = transform.location.x
        tf_msg.transform.translation.y = -transform.location.y
        tf_msg.transform.translation.z = transform.location.z
        
        # Rotation (flip pitch and yaw for ROS)
        roll = math.radians(transform.rotation.roll)
        pitch = math.radians(transform.rotation.pitch)
        yaw = math.radians(-transform.rotation.yaw)
        tf_msg.transform.rotation = euler_to_quaternion(roll, pitch, yaw)
        
        self.tf_broadcaster.sendTransform(tf_msg)
    
    def publish_odometry_synchronized(self, ros_time: TimeMsg):
        """Publish odometry with synchronized timestamp."""
        if self.vehicle is None or not self.vehicle.is_alive:
            return
        
        msg = carla_vehicle_to_odometry(self.vehicle, 'map', 'ego_vehicle')
        msg.header.stamp = ros_time
        self.odom_pub.publish(msg)
    
    def publish_sensor_data(self, sensor_data_packet: Dict[str, Any], ros_time: TimeMsg):
        """
        Process and publish all sensor data with synchronized timestamp.
        
        Uses ThreadPoolExecutor for parallel processing of camera/lidar data.
        """
        futures = []
        
        for frame_id, carla_data in sensor_data_packet.items():
            if frame_id not in self.sensors:
                continue
            
            sensor_info = self.sensors[frame_id]
            sensor_type = sensor_info['type']
            
            # Submit heavy processing to thread pool
            if 'sensor.camera' in sensor_type or 'sensor.lidar' in sensor_type:
                future = self.publish_executor.submit(
                    self._publish_sensor_threaded, frame_id, carla_data, sensor_info, sensor_type, ros_time
                )
                futures.append(future)
            else:
                # Light sensors (IMU, GNSS) - process inline
                try:
                    if sensor_type == 'sensor.other.gnss':
                        self._publish_gnss(carla_data, sensor_info, ros_time)
                    elif sensor_type == 'sensor.other.imu':
                        self._publish_imu(carla_data, sensor_info, ros_time)
                except Exception as e:
                    self.get_logger().error(f'Error publishing {frame_id}: {e}')
        
        # Wait for all parallel tasks to complete
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                self.get_logger().error(f'Error in parallel publish: {e}')
    
    def _publish_sensor_threaded(self, frame_id: str, carla_data, sensor_info: dict, sensor_type: str, ros_time: TimeMsg):
        """Thread-safe sensor publishing for heavy processing."""
        try:
            if 'sensor.camera.rgb' in sensor_type:
                self._publish_rgb(carla_data, sensor_info, ros_time)
            elif 'sensor.camera.depth' in sensor_type:
                self._publish_depth(carla_data, sensor_info, ros_time)
            elif 'sensor.camera.semantic_segmentation' in sensor_type:
                self._publish_semantic(carla_data, sensor_info, ros_time)
            elif sensor_type == 'sensor.lidar.ray_cast':
                self._publish_lidar(carla_data, sensor_info, ros_time)
            elif sensor_type == 'sensor.lidar.ray_cast_semantic':
                self._publish_semantic_lidar(carla_data, sensor_info, ros_time)
        except Exception as e:
            self.get_logger().error(f'Error publishing {frame_id}: {e}')
    
    # ==================== Sensor Processing & Publishing ====================
    
    def _publish_rgb(self, carla_image, sensor_info: dict, ros_time: TimeMsg):
        """Process and publish RGB camera data."""
        # Convert BGRA to RGB
        array = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
        array = array.reshape((carla_image.height, carla_image.width, 4))
        rgb_array = array[:, :, :3][:, :, ::-1]  # BGRA -> RGB
        
        # Create Image message
        msg = self.cv_bridge.cv2_to_imgmsg(
            np.ascontiguousarray(rgb_array), encoding='rgb8')
        msg.header = self.create_synchronized_header(sensor_info['frame_id'], ros_time)
        
        sensor_info['image_pub'].publish(msg)
        
        # Use cached CameraInfo (only create once per sensor)
        frame_id = sensor_info['frame_id']
        if frame_id not in self.cached_camera_info:
            config = sensor_info['config']
            self.cached_camera_info[frame_id] = create_camera_info(
                config.get('image_size_x', carla_image.width),
                config.get('image_size_y', carla_image.height),
                config.get('fov', 90.0),
                frame_id,
                0.0  # Timestamp will be overwritten
            )
        
        info_msg = self.cached_camera_info[frame_id]
        info_msg.header = self.create_synchronized_header(frame_id, ros_time)
        sensor_info['info_pub'].publish(info_msg)
    
    def _publish_depth(self, carla_image, sensor_info: dict, ros_time: TimeMsg):
        """Process and publish depth camera data with optimized conversion."""
        # Parse raw buffer once
        array = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
        array = array.reshape((carla_image.height, carla_image.width, 4))
        
        # Single-pass vectorized depth conversion (avoid 3 separate astype calls)
        # Combine BGR channels into 24-bit depth value, convert to meters
        depth_raw = array[:, :, :3].astype(np.float32)
        depth_meters = (depth_raw[:, :, 2] + 
                       depth_raw[:, :, 1] * 256.0 + 
                       depth_raw[:, :, 0] * 65536.0) / 16777215.0 * 1000.0
        
        msg = carla_depth_to_ros(depth_meters, sensor_info['frame_id'], carla_image.timestamp)
        msg.header = self.create_synchronized_header(sensor_info['frame_id'], ros_time)
        sensor_info['image_pub'].publish(msg)
        
        # Use cached CameraInfo
        frame_id = sensor_info['frame_id']
        if frame_id not in self.cached_camera_info:
            config = sensor_info['config']
            self.cached_camera_info[frame_id] = create_camera_info(
                config.get('image_size_x', carla_image.width),
                config.get('image_size_y', carla_image.height),
                config.get('fov', 90.0),
                frame_id,
                0.0
            )
        
        info_msg = self.cached_camera_info[frame_id]
        info_msg.header = self.create_synchronized_header(frame_id, ros_time)
        sensor_info['info_pub'].publish(info_msg)
    
    def _publish_semantic(self, carla_image, sensor_info: dict, ros_time: TimeMsg):
        """Process and publish semantic segmentation camera data."""
        array = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
        array = array.reshape((carla_image.height, carla_image.width, 4))
        semantic_labels = array[:, :, 2]  # Red channel has class ID
        
        msg = carla_semantic_to_ros(semantic_labels, sensor_info['frame_id'], carla_image.timestamp)
        msg.header = self.create_synchronized_header(sensor_info['frame_id'], ros_time)
        sensor_info['image_pub'].publish(msg)
        
        # Use cached CameraInfo
        frame_id = sensor_info['frame_id']
        if frame_id not in self.cached_camera_info:
            config = sensor_info['config']
            self.cached_camera_info[frame_id] = create_camera_info(
                config.get('image_size_x', carla_image.width),
                config.get('image_size_y', carla_image.height),
                config.get('fov', 90.0),
                frame_id,
                0.0
            )
        
        info_msg = self.cached_camera_info[frame_id]
        info_msg.header = self.create_synchronized_header(frame_id, ros_time)
        sensor_info['info_pub'].publish(info_msg)
    
    def _publish_lidar(self, carla_lidar, sensor_info: dict, ros_time: TimeMsg):
        """Process and publish LiDAR point cloud data."""
        # Direct buffer access without bytes() copy
        lidar_data = np.frombuffer(carla_lidar.raw_data, dtype=np.float32)
        lidar_data = lidar_data.reshape((-1, 4)).copy()
        
        # CARLA LiDAR uses left-handed, ROS uses right-handed (flip Y)
        lidar_data[:, 1] *= -1
        
        msg = carla_lidar_to_ros(lidar_data, sensor_info['frame_id'], carla_lidar.timestamp)
        msg.header = self.create_synchronized_header(sensor_info['frame_id'], ros_time)
        sensor_info['cloud_pub'].publish(msg)
    
    def _publish_semantic_lidar(self, carla_lidar, sensor_info: dict, ros_time: TimeMsg):
        """Process and publish semantic LiDAR point cloud data."""
        semantic_dtype = np.dtype([
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('cos_inc_angle', np.float32),
            ('object_idx', np.uint32),
            ('object_tag', np.uint32),
        ])

        # Direct buffer access without bytes() copy
        raw = np.frombuffer(carla_lidar.raw_data, dtype=semantic_dtype)
        if raw.size == 0:
            return

        x = raw['x']
        y = -raw['y']  # Flip Y for ROS
        z = raw['z']
        labels = raw['object_tag']

        points = np.stack((x, y, z), axis=-1)

        msg = carla_semantic_lidar_to_ros(points, labels, sensor_info['frame_id'], carla_lidar.timestamp)
        msg.header = self.create_synchronized_header(sensor_info['frame_id'], ros_time)
        sensor_info['cloud_pub'].publish(msg)
    
    def _publish_gnss(self, carla_gnss, sensor_info: dict, ros_time: TimeMsg):
        """Process and publish GNSS data."""
        msg = NavSatFix()
        msg.header = self.create_synchronized_header(sensor_info['frame_id'], ros_time)
        msg.latitude = carla_gnss.latitude
        msg.longitude = carla_gnss.longitude
        msg.altitude = carla_gnss.altitude
        msg.status.status = 0  # STATUS_FIX
        msg.status.service = 1  # SERVICE_GPS
        sensor_info['gnss_pub'].publish(msg)
    
    def _publish_imu(self, carla_imu, sensor_info: dict, ros_time: TimeMsg):
        """Process and publish IMU data."""
        msg = Imu()
        msg.header = self.create_synchronized_header(sensor_info['frame_id'], ros_time)
        
        # Linear acceleration (flip Y)
        msg.linear_acceleration.x = carla_imu.accelerometer.x
        msg.linear_acceleration.y = -carla_imu.accelerometer.y
        msg.linear_acceleration.z = carla_imu.accelerometer.z
        
        # Angular velocity (flip Y and Z)
        msg.angular_velocity.x = carla_imu.gyroscope.x
        msg.angular_velocity.y = -carla_imu.gyroscope.y
        msg.angular_velocity.z = -carla_imu.gyroscope.z
        
        # Orientation from compass
        yaw = math.radians(-carla_imu.compass)
        msg.orientation = euler_to_quaternion(0.0, 0.0, yaw)
        
        sensor_info['imu_pub'].publish(msg)
    
    # ==================== Static TF ====================
    
    def publish_static_sensor_tfs(self):
        """Publish static transforms for all sensors (called once)."""
        static_transforms = []
        ros_time = self.carla_timestamp_to_ros(0.0)
        
        for sensor_id, sensor_data in self.sensors.items():
            rel_tf = TransformStamped()
            rel_tf.header = self.create_synchronized_header('ego_vehicle', ros_time)
            rel_tf.child_frame_id = sensor_data['frame_id']
            
            # Use spawn point from config for relative position
            sp = sensor_data['config'].get('spawn_point', {})
            rel_tf.transform.translation.x = sp.get('x', 0.0)
            rel_tf.transform.translation.y = -sp.get('y', 0.0)  # Flip Y
            rel_tf.transform.translation.z = sp.get('z', 0.0)
            
            roll = math.radians(sp.get('roll', 0.0))
            pitch = math.radians(sp.get('pitch', 0.0))
            yaw = math.radians(-sp.get('yaw', 0.0))  # Flip yaw
            rel_tf.transform.rotation = euler_to_quaternion(roll, pitch, yaw)
            
            static_transforms.append(rel_tf)
        
        if static_transforms:
            self.static_tf_broadcaster.sendTransform(static_transforms)
            self.get_logger().info(f'Published {len(static_transforms)} static sensor transforms')
    
    # ==================== Control Callback ====================
    
    def control_callback(self, msg: VehicleControl):
        """Handle vehicle control commands."""
        with self.control_lock:
            self.current_control.throttle = max(0.0, min(1.0, msg.throttle))
            self.current_control.brake = max(0.0, min(1.0, msg.brake))
            self.current_control.steer = max(-1.0, min(1.0, -msg.steer))
            self.current_control.hand_brake = msg.hand_brake
            self.current_control.reverse = msg.reverse
            self.current_control.manual_gear_shift = msg.manual_gear_shift
            self.current_control.gear = msg.gear
    
    # ==================== Cleanup ====================
    
    def destroy_node(self):
        """Clean up CARLA actors before shutting down."""
        self.get_logger().info('Shutting down CARLA Bridge...')
        
        # Stop the main loop
        self.running = False
        
        # Wait for thread to finish
        if hasattr(self, 'main_loop_thread') and self.main_loop_thread.is_alive():
            self.main_loop_thread.join(timeout=2.0)
        
        # Destroy sensors
        for sensor_id, sensor_data in self.sensors.items():
            if sensor_data['actor'].is_alive:
                sensor_data['actor'].stop()
                sensor_data['actor'].destroy()
                self.get_logger().info(f'Destroyed sensor: {sensor_id}')
        
        # Destroy vehicle
        if self.vehicle is not None and self.vehicle.is_alive:
            self.vehicle.destroy()
            self.get_logger().info('Destroyed vehicle')
        
        # Reset world settings
        if self.world is not None:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    node = None
    try:
        node = CarlaBridgeNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
