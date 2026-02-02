#!/usr/bin/env python3
"""
Camera Snapshot Tool
====================

ROS2 utility to capture and save a snapshot from any camera topic.
Useful for capturing test images for offline model testing.

Usage:
    # Default RGB front camera
    python save_camera_snapshot.py
    
    # Specify camera topic
    python save_camera_snapshot.py --topic /carla/ego_vehicle/sensor/camera/rgb/front/image
    
    # Custom output path
    python save_camera_snapshot.py --output my_snapshot.png
    
    # Wait for specific number of frames before capturing
    python save_camera_snapshot.py --skip 10
    
    # Capture multiple frames
    python save_camera_snapshot.py --count 5 --interval 1.0
"""

import argparse
import time
from pathlib import Path

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg import Image


class CameraSnapshotNode(Node):
    """Node to capture and save camera snapshots."""
    
    def __init__(
        self, 
        topic: str,
        output_path: str,
        skip_frames: int = 0,
        count: int = 1,
        interval: float = 0.0,
        timeout: float = 10.0
    ):
        super().__init__('camera_snapshot')
        
        self.output_path = Path(output_path)
        self.skip_frames = skip_frames
        self.count = count
        self.interval = interval
        self.timeout = timeout
        
        # State
        self._frame_count = 0
        self._captured_count = 0
        self._last_capture_time = 0.0
        self._done = False
        self._start_time = time.time()
        
        # Create output directory if needed
        if self.count > 1:
            self.output_path.mkdir(parents=True, exist_ok=True)
        else:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # QoS for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )
        
        # Subscribe to camera topic
        self.subscription = self.create_subscription(
            Image,
            topic,
            self.image_callback,
            sensor_qos
        )
        
        self.get_logger().info(f'Waiting for images on: {topic}')
        self.get_logger().info(f'Output: {self.output_path}')
        if skip_frames > 0:
            self.get_logger().info(f'Skipping first {skip_frames} frames')
        
        # Timeout timer
        self.timeout_timer = self.create_timer(1.0, self.check_timeout)
    
    def image_callback(self, msg: Image):
        """Handle incoming image message."""
        if self._done:
            return
        
        self._frame_count += 1
        
        # Skip initial frames if requested
        if self._frame_count <= self.skip_frames:
            self.get_logger().info(f'Skipping frame {self._frame_count}/{self.skip_frames}')
            return
        
        # Check interval between captures
        current_time = time.time()
        if self._captured_count > 0 and self.interval > 0:
            if current_time - self._last_capture_time < self.interval:
                return
        
        # Convert and save image
        try:
            image_np = self._ros_to_numpy(msg)
            self._save_image(image_np)
            self._captured_count += 1
            self._last_capture_time = current_time
            
            self.get_logger().info(
                f'Captured {self._captured_count}/{self.count} '
                f'({msg.width}x{msg.height}, {msg.encoding})'
            )
            
            if self._captured_count >= self.count:
                self._done = True
                self.get_logger().info('All snapshots captured!')
                
        except Exception as e:
            self.get_logger().error(f'Failed to save image: {e}')
    
    def _ros_to_numpy(self, msg: Image) -> np.ndarray:
        """Convert ROS Image to numpy array (BGR for OpenCV)."""
        if msg.encoding == 'rgb8':
            image = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif msg.encoding == 'bgr8':
            image = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3)
        elif msg.encoding == 'bgra8':
            image = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 4)
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif msg.encoding == 'rgba8':
            image = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 4)
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        elif msg.encoding == 'mono8':
            image = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width)
        elif msg.encoding == '32FC1':
            # Depth image - normalize for visualization
            image = np.frombuffer(msg.data, dtype=np.float32).reshape(
                msg.height, msg.width)
            # Normalize to 0-255 (assuming max depth ~100m)
            image = np.clip(image / 100.0 * 255, 0, 255).astype(np.uint8)
        elif msg.encoding == '16UC1':
            # Depth image 16-bit
            image = np.frombuffer(msg.data, dtype=np.uint16).reshape(
                msg.height, msg.width)
            image = (image / 256).astype(np.uint8)
        else:
            raise ValueError(f'Unsupported encoding: {msg.encoding}')
        
        return image
    
    def _save_image(self, image: np.ndarray):
        """Save image to disk."""
        if self.count > 1:
            # Multiple images - use numbered filenames within batch directory
            filename = self.output_path / f'{self._captured_count}.png'
        else:
            filename = self.output_path
            # Ensure the filename has a valid extension
            if filename.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                filename = filename.with_suffix('.png')

        # Ensure parent directory exists
        filename.parent.mkdir(parents=True, exist_ok=True)
        
        self.get_logger().debug(f'Saving to: {filename}')
        success = cv2.imwrite(str(filename), image)
        if not success:
            raise RuntimeError(f'cv2.imwrite failed for: {filename}')
    
    def check_timeout(self):
        """Check for timeout."""
        if self._done:
            rclpy.shutdown()
            return
        
        elapsed = time.time() - self._start_time
        if elapsed > self.timeout:
            if self._captured_count == 0:
                self.get_logger().error(f'Timeout: No images received after {self.timeout}s')
            else:
                self.get_logger().warn(
                    f'Timeout: Captured {self._captured_count}/{self.count} images'
                )
            self._done = True
            rclpy.shutdown()
    
    @property
    def is_done(self) -> bool:
        return self._done


def main():
    parser = argparse.ArgumentParser(
        description='Capture snapshots from ROS2 camera topics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Capture from default camera
  python save_camera_snapshot.py
  
  # Capture from semantic camera
  python save_camera_snapshot.py -t /carla/ego_vehicle/ground_truth/camera/semantic/front/image
  
  # Capture 10 frames, 0.5s apart
  python save_camera_snapshot.py --count 10 --interval 0.5 --output ./captures/
  
  # Skip first 5 frames (wait for camera to stabilize)
  python save_camera_snapshot.py --skip 5
        '''
    )
    
    parser.add_argument(
        '--topic', '-t',
        type=str,
        default='/carla/ego_vehicle/sensor/camera/rgb/front/image',
        help='Camera image topic to capture from'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output path (file for single image, directory for multiple)'
    )
    parser.add_argument(
        '--skip', '-s',
        type=int,
        default=0,
        help='Number of initial frames to skip'
    )
    parser.add_argument(
        '--count', '-c',
        type=int,
        default=1,
        help='Number of snapshots to capture'
    )
    parser.add_argument(
        '--interval', '-i',
        type=float,
        default=0.0,
        help='Interval between captures in seconds (for count > 1)'
    )
    parser.add_argument(
        '--timeout',
        type=float,
        default=30.0,
        help='Timeout in seconds'
    )
    
    args = parser.parse_args()
    
    # Generate default output path in perception/data directory
    if args.output is None:
        # Get the data directory relative to this script
        script_dir = Path(__file__).resolve().parent.parent  # perception/perception/
        data_dir = script_dir / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract camera type from topic (e.g., rgb_front, semantic_back, etc.)
        # Topic format: /carla/ego_vehicle/sensor/camera/rgb/front/image
        topic_parts = args.topic.strip('/').split('/')
        
        # Try to extract camera type (rgb/depth/semantic) and position (front/back/left/right)
        camera_type = 'camera'
        camera_position = ''
        
        if 'rgb' in topic_parts:
            camera_type = 'rgb'
        elif 'semantic' in topic_parts:
            camera_type = 'semantic'
        elif 'depth' in topic_parts:
            camera_type = 'depth'
        
        for pos in ['front', 'back', 'left', 'right', 'top', 'rear']:
            if pos in topic_parts:
                camera_position = pos
                break
        
        # Build base name: snapshot_rgb_front or snapshot_semantic_back, etc.
        if camera_position:
            base_name = f'snapshot_{camera_type}_{camera_position}'
        else:
            base_name = f'snapshot_{camera_type}'
        
        # Find the next available number
        existing_files = list(data_dir.glob(f'{base_name}_*.png'))
        if existing_files:
            # Extract numbers from existing files
            numbers = []
            for f in existing_files:
                # filename like snapshot_rgb_front_1.png -> extract 1
                try:
                    num_str = f.stem.rsplit('_', 1)[-1]
                    numbers.append(int(num_str))
                except ValueError:
                    # Ignore files that do not match the expected "<base_name>_<number>.png" pattern
                    pass
            next_num = max(numbers) + 1 if numbers else 1
        else:
            next_num = 1
        
        if args.count > 1:
            # For multiple captures, use directory
            args.output = str(data_dir / f'{base_name}_batch_{next_num}')
        else:
            args.output = str(data_dir / f'{base_name}_{next_num}.png')
    
    # Initialize ROS2
    rclpy.init()
    
    node = CameraSnapshotNode(
        topic=args.topic,
        output_path=args.output,
        skip_frames=args.skip,
        count=args.count,
        interval=args.interval,
        timeout=args.timeout
    )
    
    try:
        while rclpy.ok() and not node.is_done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        print('\nInterrupted by user')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
