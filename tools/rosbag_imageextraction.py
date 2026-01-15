#!/usr/bin/env python3
"""
ROS2 Image Subscriber - Saves images from a topic at 2-second intervals.
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
    def __init__(self, topic_name='/camera/image_raw', output_dir='./captured_images'):
        super().__init__('image_subscriber')
        
        # Configuration
        self.output_dir = output_dir
        self.save_interval = 2.0  # seconds
        self.last_save_time = 0.0
        self.image_count = 0
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create subscription
        self.subscription = self.create_subscription(
            Image,
            topic_name,
            self.image_callback,
            10  # QoS queue size
        )
        
        self.get_logger().info(f'Subscribed to topic: {topic_name}')
        self.get_logger().info(f'Saving images to: {os.path.abspath(self.output_dir)}')
        self.get_logger().info(f'Save interval: {self.save_interval} seconds')

    def image_callback(self, msg):
        """Callback function for image messages."""
        current_time = time.time()
        
        # Check if enough time has passed since last save
        if current_time - self.last_save_time >= self.save_interval:
            try:
                # Get the image encoding from the message
                encoding = msg.encoding
                self.get_logger().info(f'Image encoding: {encoding}, size: {msg.width}x{msg.height}')
                
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
                    self.get_logger().error(f'Unsupported encoding: {encoding}')
                    return
                
                # Make a contiguous copy
                rgb_image = np.ascontiguousarray(rgb_image)
                
                self.get_logger().info(f'Array shape: {rgb_image.shape}, dtype: {rgb_image.dtype}')
                
                # Generate filename with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                filename = f'image_{self.image_count:06d}_{timestamp}.png'
                filepath = os.path.join(self.output_dir, filename)
                
                # Save using PIL
                if len(rgb_image.shape) == 2:
                    pil_image = PILImage.fromarray(rgb_image, mode='L')
                else:
                    pil_image = PILImage.fromarray(rgb_image, mode='RGB')
                
                pil_image.save(filepath)
                
                self.image_count += 1
                self.last_save_time = current_time
                self.get_logger().info(f'Saved image {self.image_count}: {filename}')
                
            except Exception as e:
                import traceback
                self.get_logger().error(f'Error saving image: {str(e)}')
                self.get_logger().error(traceback.format_exc())


def main(args=None):
    rclpy.init(args=args)
    
    # You can customize these parameters
    topic_name =  '/zed_multi/zed_left/right/image_rect_color'  # Change this to your image topic
    output_dir = '/home/autoiq/nakul/Auto-IQ/data/rosbag_images_15jan'   # Change this to your desired output directory
    
    image_subscriber = ImageSubscriber(
        topic_name=topic_name,
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
