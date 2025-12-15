"""
LiDAR Sensor
============

LiDAR point cloud sensor for 3D perception.
"""

import carla
import numpy as np

from sensors.base_sensor import BaseSensor


class LiDAR(BaseSensor):
    """LiDAR point cloud sensor."""
    
    def __init__(self, vehicle, world, position,
                 channels=32, range=100.0, points_per_second=500000,
                 rotation_frequency=20.0, upper_fov=10.0, lower_fov=-30.0,
                 sensor_tick=0.0):
        """
        Initialize LiDAR sensor.
        
        Args:
            vehicle: CARLA vehicle actor
            world: CARLA world
            position: Dict with x, y, z, pitch, yaw, roll
            channels: Number of laser channels (vertical resolution)
            range: Maximum detection range in meters
            points_per_second: Number of points generated per second
            rotation_frequency: Rotation frequency in Hz
            upper_fov: Upper field of view limit in degrees
            lower_fov: Lower field of view limit in degrees
            sensor_tick: Sensor capture rate (0.0 = every frame)
        """
        super().__init__(vehicle, world)
        
        self.position = position
        self.channels = channels
        self.range = range
        self.points_per_second = points_per_second
        self.rotation_frequency = rotation_frequency
        self.upper_fov = upper_fov
        self.lower_fov = lower_fov
        self.sensor_tick = sensor_tick
        
        # Store raw point cloud data
        self.point_cloud = None
    
    def _get_blueprint(self):
        """Get LiDAR blueprint."""
        bp_library = self.world.get_blueprint_library()
        bp = bp_library.find('sensor.lidar.ray_cast')
        bp.set_attribute('channels', str(self.channels))
        bp.set_attribute('range', str(self.range))
        bp.set_attribute('points_per_second', str(self.points_per_second))
        bp.set_attribute('rotation_frequency', str(self.rotation_frequency))
        bp.set_attribute('upper_fov', str(self.upper_fov))
        bp.set_attribute('lower_fov', str(self.lower_fov))
        bp.set_attribute('sensor_tick', str(self.sensor_tick))
        return bp
    
    def _get_transform(self):
        """Get LiDAR transform."""
        return carla.Transform(
            carla.Location(
                x=self.position['x'],
                y=self.position['y'],
                z=self.position['z']
            ),
            carla.Rotation(
                pitch=self.position['pitch'],
                yaw=self.position['yaw'],
                roll=self.position['roll']
            )
        )
    
    def _process_data(self, lidar_data):
        """
        Process LiDAR data into point cloud array.
        
        Returns:
            numpy array of shape (N, 4) with columns [x, y, z, intensity]
        """
        # Get raw data as bytes
        data = np.frombuffer(lidar_data.raw_data, dtype=np.float32)
        # Reshape to (N, 4): x, y, z, intensity
        points = data.reshape(-1, 4)
        self.point_cloud = points
        return points
    
    def get_point_cloud(self):
        """Get the raw point cloud data."""
        return self.point_cloud
    
    def get_bev_image(self, image_size=400, pixels_per_meter=5):
        """
        Generate a bird's eye view (BEV) image from point cloud.
        
        Args:
            image_size: Size of the output square image
            pixels_per_meter: Resolution (pixels per meter)
            
        Returns:
            numpy array (image_size, image_size, 3) RGB image
        """
        if self.point_cloud is None or len(self.point_cloud) == 0:
            # Return empty image
            return np.zeros((image_size, image_size, 3), dtype=np.uint8)
        
        points = self.point_cloud
        
        # Create BEV image
        img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        
        # Center of image
        center = image_size // 2
        
        # Extract x, y, z, intensity
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        intensity = points[:, 3]
        
        # Convert to pixel coordinates
        # CARLA: x = forward, y = right, z = up
        # BEV image: px = horizontal (right), py = vertical (forward = up in image)
        px = center + (y * pixels_per_meter).astype(np.int32)  # y (right) -> horizontal (right)
        py = center - (x * pixels_per_meter).astype(np.int32)  # x (forward) -> vertical (up)
        
        # Filter points within image bounds
        valid = (px >= 0) & (px < image_size) & (py >= 0) & (py < image_size)
        px = px[valid]
        py = py[valid]
        z = z[valid]
        intensity = intensity[valid]
        
        # Color by height (z) - normalize to 0-255
        z_min, z_max = -2.0, 3.0  # Typical range for ground to car roof
        z_norm = np.clip((z - z_min) / (z_max - z_min), 0, 1)
        
        # Create color based on height (blue=low, green=mid, red=high)
        colors = np.zeros((len(z_norm), 3), dtype=np.uint8)
        colors[:, 0] = (z_norm * 255).astype(np.uint8)  # Red for high
        colors[:, 1] = ((1 - np.abs(z_norm - 0.5) * 2) * 255).astype(np.uint8)  # Green for mid
        colors[:, 2] = ((1 - z_norm) * 255).astype(np.uint8)  # Blue for low
        
        # Draw points
        img[py, px] = colors
        
        # Draw vehicle marker at center (white rectangle)
        car_length = int(4.5 * pixels_per_meter)  # ~4.5m car
        car_width = int(2.0 * pixels_per_meter)   # ~2m width
        cv_x1 = center - car_width // 2
        cv_x2 = center + car_width // 2
        cv_y1 = center - car_length // 2
        cv_y2 = center + car_length // 2
        img[cv_y1:cv_y2, cv_x1:cv_x2] = [255, 255, 255]
        
        # Draw direction indicator (triangle pointing forward/up)
        img[cv_y1-5:cv_y1, center-2:center+2] = [0, 255, 255]
        
        return img
