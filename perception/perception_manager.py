"""Perception Manager - Orchestrates obstacle and lane detection."""

import carla
from typing import Optional, List

from config.config import PERCEPTION
from navigation.obstacle_state import Obstacle
from perception.fake_perception import FakePerception
from perception.perception import Perception
from perception.lane_detector import LaneDetector, LaneDetectorDebugWindow
from visualization.bounding_box_visualizer import BoundingBoxVisualizer


class PerceptionManager:
    """Orchestrates obstacle detection and lane detection, updates shared state."""
    
    def __init__(
        self,
        vehicle,
        world,
        sensor_suite,
        perception_mode: str = 'fake',
        enable_lane_detection: bool = True,
        visualize_options: Optional[set] = None,
        debug: bool = False,
    ):
        self.vehicle = vehicle
        self.world = world
        self.sensor_suite = sensor_suite
        self.visualize_options = visualize_options or set()
        self.perception_mode = perception_mode
        
        # Read perception config
        model_path = PERCEPTION['object_detector']['model_path']
        conf_threshold = PERCEPTION['object_detector']['conf_threshold']
        
        # Obstacle detection
        if self.perception_mode == 'fake':
            self._obstacle_detector = FakePerception(vehicle, world)
            print("[Perception] Using FakePerception (ground truth)")
        else:
            self._obstacle_detector = Perception(
                vehicle=vehicle, world=world, sensor_suite=sensor_suite,
                model_path=model_path,
                conf_threshold=conf_threshold
            )
            print("[Perception] Using real perception (YOLO + Depth)")
        
        # Bounding box visualization (handled here, not in detector)
        self._bbox_visualizer = None
        if debug or 'all' in self.visualize_options or 'bbox' in self.visualize_options:
            self._bbox_visualizer = BoundingBoxVisualizer(world, ego_vehicle=vehicle)
        
        # Lane detection
        self._lane_detector = None
        self._lane_debug_window = None
        
        if enable_lane_detection and sensor_suite:
            self._init_lane_detection(sensor_suite)
        elif not enable_lane_detection:
            print("[Perception] Lane detection disabled")
        else:
            print("[Perception] Warning: No sensor suite provided, lane detection disabled")
    
    def _init_lane_detection(self, sensor_suite) -> None:
        """Initialize lane detector from semantic camera parameters."""
        semantic_sensor = sensor_suite.get_sensor('semantic')
        if not semantic_sensor:
            print("[Perception] Warning: No semantic sensor found, lane detection disabled")
            return
        
        camera_fov = semantic_sensor.fov
        camera_pitch = semantic_sensor.position.get('pitch', 0.0)
        camera_height = semantic_sensor.position.get('z', 1.5)
        
        self._lane_detector = LaneDetector(
            image_width=semantic_sensor.width,
            image_height=semantic_sensor.height,
            camera_fov=camera_fov,
            camera_pitch=camera_pitch,
            camera_height=camera_height,
        )
        
        if 'lane' in self.visualize_options or 'all' in self.visualize_options:
            self._lane_debug_window = LaneDetectorDebugWindow("Lane Detection Debug")
        
        print(f"[Perception] Lane detector initialized (FOV={camera_fov}°, pitch={camera_pitch}°, height={camera_height}m)")
    
    def run_step(self) -> tuple:
        """
        Execute one perception cycle.
        
        Returns:
            (obstacles, lanes) - tuple of detection results for caller to update shared state
        """
        obstacles = self._obstacle_detector.detect_obstacles(radius=PERCEPTION['obstacle_radius'])
        if self._bbox_visualizer and obstacles:
            self._visualize_obstacles(obstacles)
        
        if obstacles:
            self._print_obstacles(obstacles)
        
        lanes_result = None
        if self._lane_detector:
            semantic_sensor = self.sensor_suite.get_sensor('semantic')
            if semantic_sensor:
                semantic_image = semantic_sensor.get_data()
                if semantic_image is not None:
                    lanes_result = self._lane_detector.detect(semantic_image)
                    if 'lane' in self.visualize_options or 'all' in self.visualize_options:
                        self.show_lane_debug(lanes_result)
        return obstacles, lanes_result
    
    def _visualize_obstacles(self, obstacles: List[Obstacle]) -> None:
        """Draw bounding boxes for detected obstacles."""
        ego_transform = self.vehicle.get_transform()
        
        for obs in obstacles:
            # For real perception, we don't have actor IDs, so yaw is estimated as 0
            if self.perception_mode == 'fake':
                actor = self.world.get_actor(obs.id)
                if actor:
                    actor_yaw = actor.get_transform().rotation.yaw
                    relative_yaw = actor_yaw - ego_transform.rotation.yaw
                else:
                    relative_yaw = 0.0
            else:
                # Real perception: assume obstacle is facing same direction as ego
                relative_yaw = 0.0
            
            color = carla.Color(255, 0, 0) if obs.obstacle_type == 'vehicle' else carla.Color(0, 255, 0)
            dimensions = carla.Vector3D(x=obs.length, y=obs.width, z=obs.height)
            
            self._bbox_visualizer.draw_bounding_box(
                relative_location={'x': obs.x, 'y': obs.y, 'z': obs.z},
                dimensions=dimensions,
                yaw=relative_yaw,
                color=color,
                life_time=0.5,
                thickness=0.1
            )
    
    def _print_obstacles(self, obstacles: List[Obstacle]) -> None:
        """Print obstacle detection results for debugging."""
        print(f"\n[Perception] Detected {len(obstacles)} obstacles:")
        for obs in obstacles:
            distance = (obs.x**2 + obs.y**2 + obs.z**2)**0.5
            speed = (obs.vx**2 + obs.vy**2 + obs.vz**2)**0.5
            print(f"  - {obs.obstacle_type} (ID: {obs.id}) at {distance:.1f}m | Speed: {speed:.1f} m/s")
    
    def show_lane_debug(self, lanes_result) -> None:
        """Show lane detection debug window if enabled."""
        if self._lane_debug_window and self._lane_detector:
            semantic_sensor = self.sensor_suite.get_sensor('semantic')
            if semantic_sensor:
                semantic_image = semantic_sensor.get_data()
                self._lane_debug_window.show(
                    self._lane_detector,
                    semantic_image,
                    lanes_result
                )
                LaneDetectorDebugWindow.wait_key(1)
    
    def get_debug_images(self) -> dict:
        """Get debug visualization images from obstacle detector."""
        if hasattr(self._obstacle_detector, 'get_debug_images'):
            return self._obstacle_detector.get_debug_images()
        return {}
    
    def get_images(self) -> dict:
        """
        Get all perception images for display.
        
        Returns:
            Dict of image name -> image data for visualization.
        """
        images = {}
        
        # Get debug images from obstacle detector
        images.update(self.get_debug_images())
        
        # Get annotated image with detections (if real perception)
        if hasattr(self._obstacle_detector, 'get_annotated_image'):
            annotated = self._obstacle_detector.get_annotated_image()
            if annotated is not None:
                images['yolo_detections'] = annotated
        
        return images
    
    def close(self) -> None:
        """Clean up perception resources."""
        if self._lane_debug_window:
            self._lane_debug_window.close()
