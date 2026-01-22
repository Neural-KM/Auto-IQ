#!/usr/bin/env python3
"""
CARLA Bridge Node
=================

ROS2 node that connects to CARLA simulator, spawns vehicle with sensors,
publishes sensor data to ROS2 topics, and subscribes to control commands.

"""

import json
import math
import sys
import os
import time
import threading
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

# ROS2 message types
from std_msgs.msg import Header
from sensor_msgs.msg import Image, PointCloud2, CameraInfo, NavSatFix, Imu
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from autoiq_msgs.msg import VehicleControl

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


class CarlaBridgeNode(Node):
    """
    ROS2 Node that bridges CARLA simulator to ROS2.
    
    Publishes:
        - Camera images (RGB, Depth, Semantic)
        - LiDAR point clouds
        - Vehicle odometry
        - TF transforms
        
    Subscribes:
        - Vehicle control commands
    """
    
    def __init__(self):
        super().__init__('carla_bridge_node')
        
        # Declare parameters
        self.declare_parameter('host', 'localhost')
        self.declare_parameter('port', 2000)
        self.declare_parameter('timeout', 10.0)
        self.declare_parameter('synchronous_mode', True)
        self.declare_parameter('fixed_delta_seconds', 0.05)
        self.declare_parameter('objects_file', '')
        self.declare_parameter('spawn_point_index', 50)
        self.declare_parameter('vehicle_type', 'vehicle.tesla.model3')
        
        # Get parameters
        self.host = self.get_parameter('host').value
        self.port = self.get_parameter('port').value
        self.timeout = self.get_parameter('timeout').value
        self.synchronous_mode = self.get_parameter('synchronous_mode').value
        self.fixed_delta_seconds = self.get_parameter('fixed_delta_seconds').value
        self.objects_file = self.get_parameter('objects_file').value
        self.spawn_point_index = self.get_parameter('spawn_point_index').value
        self.vehicle_type = self.get_parameter('vehicle_type').value

        # CARLA objects
        self.client = None
        self.world = None
        self.vehicle = None
        self.sensors = {}  # {sensor_id: {'actor': carla_actor, 'publisher': ros_publisher, ...}}
        
        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        
        # QoS profile for sensor data (BEST_EFFORT for low latency)
        self.sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Publishers
        self.odom_pub = self.create_publisher(Odometry, '/carla/ego_vehicle/ground_truth/odometry', 10)
        
        # Subscribers
        self.control_sub = self.create_subscription(
            VehicleControl,
            '/carla/ego_vehicle/control',
            self.control_callback,
            1
        )
        
        # Control state
        self.current_control = carla.VehicleControl()
        self.control_lock = threading.Lock()
        
        # Connect and setup (sync mode enabled AFTER spawning)
        self.connect_to_carla()
        self.spawn_vehicle_and_sensors()
        
        # NOW enable synchronous mode (after spawning to avoid freeze)
        if self.synchronous_mode:
            self.enable_synchronous_mode()
        
        # Create timer for world tick (in synchronous mode)
        if self.synchronous_mode:
            tick_period = self.fixed_delta_seconds
            self.tick_timer = self.create_timer(tick_period, self.tick_callback)
        else:
            # In async mode, publish odometry at fixed rate
            self.tick_timer = self.create_timer(0.05, self.publish_odometry)
        
        self.get_logger().info('CARLA Bridge Node initialized')
    
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
        # Do initial tick to get things started
        self.world.tick()
        self.get_logger().info(f'Synchronous mode enabled, dt={self.fixed_delta_seconds}s')
    
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
        
        # Publish static transforms for sensors (once, after all sensors spawned)
        self.publish_static_sensor_tfs()
    
    def spawn_vehicle(self):
        """Spawn the ego vehicle."""
        blueprint_library = self.world.get_blueprint_library()
        
        # Find vehicle blueprint
        vehicle_bp = blueprint_library.find(self.vehicle_type)
        if vehicle_bp is None:
            # Fallback to any vehicle
            vehicle_bp = blueprint_library.filter('vehicle.*')[0]
            self.get_logger().warn(f'Vehicle {self.vehicle_type} not found, using {vehicle_bp.id}')
        
        # Get spawn point
        spawn_points = self.map.get_spawn_points()
        if self.spawn_point_index < len(spawn_points):
            spawn_point = spawn_points[self.spawn_point_index]
        else:
            spawn_point = spawn_points[0]  # Default spawn point
            self.get_logger().warn(f'Spawn index {self.spawn_point_index} out of range, using 0')
        
        vehicle_bp.set_attribute('role_name', 'hero')
        # Spawn vehicle
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.get_logger().info(f'Spawned vehicle: {self.vehicle.type_id} at spawn point {self.spawn_point_index}. Role name: {self.vehicle.attributes.get("role_name", "unknown")}')
    
    def spawn_sensor(self, config: dict):
        """
        Spawn a sensor and setup its publisher.
        
        Args:
            config: Sensor configuration dict (carla-ros-bridge format)
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
            if 'image_size_x' in config:
                bp.set_attribute('image_size_x', str(config['image_size_x']))
            if 'image_size_y' in config:
                bp.set_attribute('image_size_y', str(config['image_size_y']))
            if 'fov' in config:
                bp.set_attribute('fov', str(config['fov']))
            if 'sensor_tick' in config:
                bp.set_attribute('sensor_tick', str(config['sensor_tick']))
                
        elif sensor_type in ('sensor.lidar.ray_cast', 'sensor.lidar.ray_cast_semantic'):
            if 'channels' in config:
                bp.set_attribute('channels', str(config['channels']))
            if 'range' in config:
                bp.set_attribute('range', str(config['range']))
            if 'points_per_second' in config:
                bp.set_attribute('points_per_second', str(config['points_per_second']))
            if 'rotation_frequency' in config:
                bp.set_attribute('rotation_frequency', str(config['rotation_frequency']))
            if 'upper_fov' in config:
                bp.set_attribute('upper_fov', str(config['upper_fov']))
            if 'lower_fov' in config:
                bp.set_attribute('lower_fov', str(config['lower_fov']))
            if 'sensor_tick' in config:
                bp.set_attribute('sensor_tick', str(config['sensor_tick']))
        
        # Create transform
        transform = carla.Transform(
            carla.Location(x=spawn_point['x'], y=spawn_point['y'], z=spawn_point['z']),
            carla.Rotation(pitch=spawn_point['pitch'], yaw=spawn_point['yaw'], roll=spawn_point['roll'])
        )
        
        # Spawn sensor attached to vehicle
        sensor_actor = self.world.spawn_actor(bp, transform, attach_to=self.vehicle)
        
        # Determine frame_id based on sensor type (matches topic structure)
        if 'sensor.camera.rgb' in sensor_type:
            frame_id = f'ego_vehicle/sensor/camera/rgb/{sensor_id}'
        elif 'sensor.camera.depth' in sensor_type:
            frame_id = f'ego_vehicle/sensor/camera/depth/{sensor_id}'
        elif 'sensor.camera.semantic_segmentation' in sensor_type:
            frame_id = f'ego_vehicle/ground_truth/camera/semantic/{sensor_id}'
        elif sensor_type == 'sensor.lidar.ray_cast':
            frame_id = f'ego_vehicle/sensor/lidar/{sensor_id}'
        elif sensor_type == 'sensor.lidar.ray_cast_semantic':
            frame_id = f'ego_vehicle/ground_truth/lidar/{sensor_id}'
        elif sensor_type == 'sensor.other.gnss':
            frame_id = f'ego_vehicle/sensor/gnss/{sensor_id}'
        elif sensor_type == 'sensor.other.imu':
            frame_id = f'ego_vehicle/sensor/imu/{sensor_id}'
        else:
            frame_id = f'ego_vehicle/{sensor_id}'
        
        # Create publishers based on sensor type
        sensor_data = {
            'actor': sensor_actor,
            'config': config,
            'frame_id': frame_id
        }
        
        if 'sensor.camera.rgb' in sensor_type:
            sensor_data['image_pub'] = self.create_publisher(
                Image, f'/carla/ego_vehicle/sensor/camera/rgb/{sensor_id}/image', self.sensor_qos)
            sensor_data['info_pub'] = self.create_publisher(
                CameraInfo, f'/carla/ego_vehicle/sensor/camera/rgb/{sensor_id}/camera_info', self.sensor_qos)
            sensor_actor.listen(lambda data, sd=sensor_data: self.rgb_callback(data, sd))
            
        elif 'sensor.camera.depth' in sensor_type:
            sensor_data['image_pub'] = self.create_publisher(
                Image, f'/carla/ego_vehicle/sensor/camera/depth/{sensor_id}/image', self.sensor_qos)
            sensor_data['info_pub'] = self.create_publisher(
                CameraInfo, f'/carla/ego_vehicle/sensor/camera/depth/{sensor_id}/camera_info', self.sensor_qos)
            sensor_actor.listen(lambda data, sd=sensor_data: self.depth_callback(data, sd))
            
        elif 'sensor.camera.semantic_segmentation' in sensor_type:
            sensor_data['image_pub'] = self.create_publisher(
                Image, f'/carla/ego_vehicle/ground_truth/camera/semantic/{sensor_id}/image', self.sensor_qos)
            sensor_data['info_pub'] = self.create_publisher(
                CameraInfo, f'/carla/ego_vehicle/ground_truth/camera/semantic/{sensor_id}/camera_info', self.sensor_qos)
            sensor_actor.listen(lambda data, sd=sensor_data: self.semantic_callback(data, sd))
            
        elif sensor_type == 'sensor.lidar.ray_cast':
            sensor_data['cloud_pub'] = self.create_publisher(
                PointCloud2, f'/carla/ego_vehicle/sensor/lidar/{sensor_id}/point_cloud', self.sensor_qos)
            sensor_actor.listen(lambda data, sd=sensor_data: self.lidar_callback(data, sd))
        elif sensor_type == 'sensor.lidar.ray_cast_semantic':
            sensor_data['cloud_pub'] = self.create_publisher(
                PointCloud2, f'/carla/ego_vehicle/ground_truth/lidar/{sensor_id}/point_cloud', self.sensor_qos)
            sensor_actor.listen(lambda data, sd=sensor_data: self.semantic_lidar_callback(data, sd))
            
        elif sensor_type == 'sensor.other.gnss':
            sensor_data['gnss_pub'] = self.create_publisher(
                NavSatFix, f'/carla/ego_vehicle/sensor/gnss/{sensor_id}/fix', self.sensor_qos)
            sensor_actor.listen(lambda data, sd=sensor_data: self.gnss_callback(data, sd))
            
        elif sensor_type == 'sensor.other.imu':
            sensor_data['imu_pub'] = self.create_publisher(
                Imu, f'/carla/ego_vehicle/sensor/imu/{sensor_id}/data', self.sensor_qos)
            sensor_actor.listen(lambda data, sd=sensor_data: self.imu_callback(data, sd))
        
        self.sensors[frame_id] = sensor_data
        self.get_logger().info(f'Spawned sensor: {sensor_type} as {sensor_id} (frame: {frame_id})')
    
    # ==================== Sensor Callbacks ====================
    
    def rgb_callback(self, carla_image, sensor_data: dict):
        """Process RGB camera data."""
        # Convert BGRA to RGB
        array = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
        array = array.reshape((carla_image.height, carla_image.width, 4))
        rgb_array = array[:, :, :3][:, :, ::-1]  # BGRA -> RGB
        
        # Use ROS time for TF synchronization
        ros_stamp = self.get_clock().now().to_msg()
        
        # Create and publish Image message
        msg = Image()
        msg.header = create_header(sensor_data['frame_id'], carla_image.timestamp)
        msg.header.stamp = ros_stamp
        msg.height = carla_image.height
        msg.width = carla_image.width
        msg.encoding = 'rgb8'
        msg.is_bigendian = False
        msg.step = carla_image.width * 3
        msg.data = rgb_array.tobytes()
        
        sensor_data['image_pub'].publish(msg)
        
        # Publish camera info
        config = sensor_data['config']
        info_msg = create_camera_info(
            config.get('image_size_x', carla_image.width),
            config.get('image_size_y', carla_image.height),
            config.get('fov', 90.0),
            sensor_data['frame_id'],
            carla_image.timestamp
        )
        info_msg.header.stamp = ros_stamp
        sensor_data['info_pub'].publish(info_msg)
    
    def depth_callback(self, carla_image, sensor_data: dict):
        """Process depth camera data."""
        # Convert to depth in meters
        array = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
        array = array.reshape((carla_image.height, carla_image.width, 4))
        
        # CARLA depth encoding: depth = (R + G*256 + B*256*256) / (256^3 - 1) * 1000
        normalized = (array[:, :, 2].astype(np.float32) + 
                     array[:, :, 1].astype(np.float32) * 256.0 +
                     array[:, :, 0].astype(np.float32) * 256.0 * 256.0)
        normalized /= (256.0 * 256.0 * 256.0 - 1.0)
        depth_meters = normalized * 1000.0  # Convert to meters
        
        # Use ROS time for TF synchronization
        ros_stamp = self.get_clock().now().to_msg()
        
        # Create and publish Image message (32FC1)
        msg = carla_depth_to_ros(depth_meters, sensor_data['frame_id'], carla_image.timestamp)
        msg.header.stamp = ros_stamp
        sensor_data['image_pub'].publish(msg)
        
        # Publish camera info
        config = sensor_data['config']
        info_msg = create_camera_info(
            config.get('image_size_x', carla_image.width),
            config.get('image_size_y', carla_image.height),
            config.get('fov', 90.0),
            sensor_data['frame_id'],
            carla_image.timestamp
        )
        info_msg.header.stamp = ros_stamp
        sensor_data['info_pub'].publish(info_msg)
    
    def semantic_callback(self, carla_image, sensor_data: dict):
        """Process semantic segmentation camera data."""
        # Get semantic labels (red channel contains class ID)
        array = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
        array = array.reshape((carla_image.height, carla_image.width, 4))
        semantic_labels = array[:, :, 2]  # Red channel has class ID
        
        # Use ROS time for TF synchronization
        ros_stamp = self.get_clock().now().to_msg()
        
        # Create and publish Image message (mono8)
        msg = carla_semantic_to_ros(semantic_labels, sensor_data['frame_id'], carla_image.timestamp)
        msg.header.stamp = ros_stamp
        sensor_data['image_pub'].publish(msg)
        
        # Publish camera info
        config = sensor_data['config']
        info_msg = create_camera_info(
            config.get('image_size_x', carla_image.width),
            config.get('image_size_y', carla_image.height),
            config.get('fov', 90.0),
            sensor_data['frame_id'],
            carla_image.timestamp
        )
        info_msg.header.stamp = ros_stamp
        sensor_data['info_pub'].publish(info_msg)
    
    def lidar_callback(self, carla_lidar, sensor_data: dict):
        """Process LiDAR point cloud data."""
        t0 = time.perf_counter()

        # Convert to numpy array (x, y, z, intensity)
        # Use bytes() to ensure proper conversion like original carla-ros-bridge
        lidar_data = np.frombuffer(bytes(carla_lidar.raw_data), dtype=np.float32)
        lidar_data = lidar_data.reshape((-1, 4)).copy()
        
        # CARLA LiDAR uses left-handed coordinate system, ROS uses right-handed
        # We take the opposite of y axis
        lidar_data[:, 1] *= -1
        
        num_points = lidar_data.shape[0]

        t1 = time.perf_counter()

        # Create and publish PointCloud2 message with ROS time (for TF sync)
        msg = carla_lidar_to_ros(lidar_data, sensor_data['frame_id'], carla_lidar.timestamp)
        msg.header.stamp = self.get_clock().now().to_msg()
        
        t2 = time.perf_counter()

        sensor_data['cloud_pub'].publish(msg)

        t3 = time.perf_counter()
        
        # Log the results
        prep_time = (t1 - t0) * 1000  # ms
        conv_time = (t2 - t1) * 1000  # ms
        pub_time  = (t3 - t2) * 1000  # ms
        
        # self.get_logger().info(
        #     f'LiDAR Profiling: Points={num_points} | Prep={prep_time:.2f}ms | Conv={conv_time:.2f}ms | Publish (Overhead)={pub_time:.2f}ms'
        # )

    def semantic_lidar_callback(self, carla_lidar, sensor_data: dict):
        """Process Semantic LiDAR point cloud data (XYZ + semantic label)."""
        semantic_dtype = np.dtype([
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('cos_inc_angle', np.float32),
            ('object_idx', np.uint32),
            ('object_tag', np.uint32),
        ])

        raw = np.frombuffer(bytes(carla_lidar.raw_data), dtype=semantic_dtype)
        if raw.size == 0:
            return

        # Flip Y to match ROS frame convention
        x = raw['x']
        y = -raw['y']
        z = raw['z']
        labels = raw['object_tag']

        points = np.stack((x, y, z), axis=-1)

        msg = carla_semantic_lidar_to_ros(points, labels, sensor_data['frame_id'], carla_lidar.timestamp)
        msg.header.stamp = self.get_clock().now().to_msg()
        sensor_data['cloud_pub'].publish(msg)
    
    def gnss_callback(self, carla_gnss, sensor_data: dict):
        """Process GNSS data."""
        msg = NavSatFix()
        msg.header = create_header(sensor_data['frame_id'], carla_gnss.timestamp)
        msg.latitude = carla_gnss.latitude
        msg.longitude = carla_gnss.longitude
        msg.altitude = carla_gnss.altitude
        msg.status.status = 0  # STATUS_FIX
        msg.status.service = 1  # SERVICE_GPS
        sensor_data['gnss_pub'].publish(msg)
    
    def imu_callback(self, carla_imu, sensor_data: dict):
        """Process IMU data."""
        msg = Imu()
        msg.header = create_header(sensor_data['frame_id'], carla_imu.timestamp)
        
        # Linear acceleration
        msg.linear_acceleration.x = carla_imu.accelerometer.x
        msg.linear_acceleration.y = -carla_imu.accelerometer.y  # Flip Y
        msg.linear_acceleration.z = carla_imu.accelerometer.z
        
        # Angular velocity
        msg.angular_velocity.x = carla_imu.gyroscope.x
        msg.angular_velocity.y = -carla_imu.gyroscope.y  # Flip Y
        msg.angular_velocity.z = -carla_imu.gyroscope.z  # Flip Z
        
        # Orientation from compass
        # CARLA compass is degrees from north, convert to quaternion
        yaw = math.radians(-carla_imu.compass)  # Flip for ROS convention
        msg.orientation = euler_to_quaternion(0.0, 0.0, yaw)
        
        sensor_data['imu_pub'].publish(msg)
    
    # ==================== Control ====================
    
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
    
    # ==================== Tick ====================
    
    def tick_callback(self):
        """Main tick callback for synchronous mode."""
        if self.vehicle is None or not self.vehicle.is_alive:
            return
        
        # Apply control
        with self.control_lock:
            self.vehicle.apply_control(self.current_control)
        
        # Tick the world
        self.world.tick()
        
        # Publish odometry and TF
        self.publish_odometry()
        self.publish_tf()
    
    def publish_odometry(self):
        """Publish vehicle odometry."""
        if self.vehicle is None or not self.vehicle.is_alive:
            return
        
        timestamp = self.world.get_snapshot().timestamp.elapsed_seconds
        
        msg = carla_vehicle_to_odometry(self.vehicle, 'map', 'ego_vehicle')
        msg.header.stamp = self.get_clock().now().to_msg()
        self.odom_pub.publish(msg)
        
        # Also publish TF in async mode
        self.publish_tf()
    
    def publish_static_sensor_tfs(self):
        """Publish static transforms for all sensors (called once after spawning)."""
        static_transforms = []
        
        for sensor_id, sensor_data in self.sensors.items():
            rel_tf = TransformStamped()
            rel_tf.header.stamp = self.get_clock().now().to_msg()
            rel_tf.header.frame_id = 'ego_vehicle'
            rel_tf.child_frame_id = sensor_data['frame_id']  # Use the full frame path
            
            # Use the spawn point from config for relative position
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
    
    def publish_tf(self):
        """Publish TF transforms (only dynamic transforms like map->ego_vehicle)."""
        if self.vehicle is None or not self.vehicle.is_alive:
            return
        
        timestamp = self.world.get_snapshot().timestamp.elapsed_seconds
        transform = self.vehicle.get_transform()
        
        # Map -> ego_vehicle (dynamic - vehicle moves)
        tf_msg = carla_transform_to_ros(transform, 'map', 'ego_vehicle', timestamp)
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        self.tf_broadcaster.sendTransform(tf_msg)
    
    # ==================== Cleanup ====================
    
    def destroy_node(self):
        """Clean up CARLA actors before shutting down."""
        self.get_logger().info('Shutting down CARLA Bridge...')
        
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
        if self.world is not None and self.synchronous_mode:
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
