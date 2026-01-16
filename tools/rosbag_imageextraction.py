#!/usr/bin/env python3
"""
ROS2 Image Subscriber - Saves images from 2 topics at 2-second intervals.
# Command to run this script:
python3 rosbag_imageextraction.py
# Or in a ROS2 environment:
ros2 run perception rosbag_imageextraction.py

# Change the topic names and output directory in the main function as needed
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import os
from datetime import datetime
import time
import numpy as np
from PIL import Image as PILImage


class ImageSubscriber(Node):
    def __init__(self, topic_name_1='/camera/image_raw', topic_name_2='/camera/image_raw_2', output_dir='./captured_images'):
        super().__init__('image_subscriber')
        
        # Configuration
        self.output_dir = output_dir
        self.save_interval = 2.0  # seconds
        
        # Separate tracking for each topic
        self.last_save_time_1 = 0.0
        self.last_save_time_2 = 0.0
        self.image_count_1 = 0
        self.image_count_2 = 0
        
        # Store topic names for logging
        self.topic_name_1 = topic_name_1
        self.topic_name_2 = topic_name_2
        
        # Create output directories for each topic
        self.output_dir_1 = os.path.join(self.output_dir, 'topic_1')
        self.output_dir_2 = os.path.join(self.output_dir, 'topic_2')
        os.makedirs(self.output_dir_1, exist_ok=True)
        os.makedirs(self.output_dir_2, exist_ok=True)
        
        # Create subscription for topic 1
        self.subscription_1 = self.create_subscription(
            Image,
            topic_name_1,
            self.image_callback_1,
            10  # QoS queue size
        )
        
        # Create subscription for topic 2
        self.subscription_2 = self.create_subscription(
            Image,
            topic_name_2,
            self.image_callback_2,
            10  # QoS queue size
        )
        
        self.get_logger().info(f'Subscribed to topic 1: {topic_name_1}')
        self.get_logger().info(f'Subscribed to topic 2: {topic_name_2}')
        self.get_logger().info(f'Saving images to: {os.path.abspath(self.output_dir)}')
        self.get_logger().info(f'Save interval: {self.save_interval} seconds')

    def image_callback_1(self, msg):
        """Callback function for topic 1 image messages."""
        self._process_image(msg, 1)
    
    def image_callback_2(self, msg):
        """Callback function for topic 2 image messages."""
        self._process_image(msg, 2)

    def _process_image(self, msg, topic_num):
        """Process and save image from either topic."""
        current_time = time.time()
        
        # Get the appropriate last save time and image count
        if topic_num == 1:
            last_save_time = self.last_save_time_1
            image_count = self.image_count_1
            output_dir = self.output_dir_1
        else:
            last_save_time = self.last_save_time_2
            image_count = self.image_count_2
            output_dir = self.output_dir_2
        
        # Check if enough time has passed since last save
        if current_time - last_save_time >= self.save_interval:
            try:
                # Get the image encoding from the message
                encoding = msg.encoding
                self.get_logger().info(f'[Topic {topic_num}] Image encoding: {encoding}, size: {msg.width}x{msg.height}')
                
                # Convert msg.data to numpy array
                # Use np.array with list() to ensure proper conversion from ROS2 array
                raw_data = np.array(list(msg.data), dtype=np.uint8)
                
                # Reshape based on encoding
                if encoding == 'bgra8':
                    # BGRA 8-bit: 4 channels
                    img_array = raw_data.reshape(msg.height, msg.width, 4)
                    # Convert BGRA to RGB (drop alpha, swap B and R)
                    rgb_image = img_array[:, :, [2, 1, 0]]  # BGR -> RGB (ignore alpha)
                elif encoding == 'rgba8':
                    # RGBA 8-bit: 4 channels
                    img_array = raw_data.reshape(msg.height, msg.width, 4)
                    rgb_image = img_array[:, :, :3]  # Drop alpha channel
                elif encoding == 'bgr8':
                    # BGR 8-bit: 3 channels
                    img_array = raw_data.reshape(msg.height, msg.width, 3)
                    rgb_image = img_array[:, :, [2, 1, 0]]  # BGR -> RGB
                elif encoding == 'rgb8':
                    # RGB 8-bit: 3 channels
                    rgb_image = raw_data.reshape(msg.height, msg.width, 3)
                elif encoding == 'mono8':
                    # Grayscale 8-bit: 1 channel
                    rgb_image = raw_data.reshape(msg.height, msg.width)
                else:
                    self.get_logger().error(f'[Topic {topic_num}] Unsupported encoding: {encoding}')
                    return
                
                # Make a contiguous copy
                rgb_image = np.ascontiguousarray(rgb_image)
                
                self.get_logger().info(f'[Topic {topic_num}] Array shape: {rgb_image.shape}, dtype: {rgb_image.dtype}')
                
                # Generate filename with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                filename = f'image_{image_count:06d}_{timestamp}.png'
                filepath = os.path.join(output_dir, filename)
                
                # Save using PIL
                if len(rgb_image.shape) == 2:
                    pil_image = PILImage.fromarray(rgb_image, mode='L')
                else:
                    pil_image = PILImage.fromarray(rgb_image, mode='RGB')
                
                pil_image.save(filepath)
                
                # Update the appropriate counters
                if topic_num == 1:
                    self.image_count_1 += 1
                    self.last_save_time_1 = current_time
                    self.get_logger().info(f'[Topic 1] Saved image {self.image_count_1}: {filename}')
                else:
                    self.image_count_2 += 1
                    self.last_save_time_2 = current_time
                    self.get_logger().info(f'[Topic 2] Saved image {self.image_count_2}: {filename}')
                
            except Exception as e:
                import traceback
                self.get_logger().error(f'[Topic {topic_num}] Error saving image: {str(e)}')
                self.get_logger().error(traceback.format_exc())


def main(args=None):
    rclpy.init(args=args)
    
    # You can customize these parameters
    topic_name_1 = '/zed_multi/zed_left/right/image_rect_color'  # Change this to your first image topic
    topic_name_2 = '/zed_multi/zed_right/left/image_rect_color'  # Change this to your second image topic
    output_dir = '/home/autoiq/nakul/Auto-IQ/data/rosbag_images_15jan/Images_15jan_bot_transit'   # Change this to your desired output directory
    
    image_subscriber = ImageSubscriber(
        topic_name_1=topic_name_1,
        topic_name_2=topic_name_2,
        output_dir=output_dir
    )
    
    try:
        rclpy.spin(image_subscriber)
    except KeyboardInterrupt:
        image_subscriber.get_logger().info('Shutting down...')
    finally:
        image_subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
