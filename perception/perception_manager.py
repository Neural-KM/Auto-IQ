"""Perception Manager - Orchestrates obstacle and lane detection."""

import carla
from typing import Optional, List
import math
from config.config import PERCEPTION
from state.obstacle import Obstacle
from state.world_state import WorldState
from perception.fake_perception import FakePerception
from perception.perception import Perception
from perception.lane_detector import LaneDetector, LaneDetectorDebugWindow
from visualization.bounding_box_visualizer import BoundingBoxVisualizer


class PerceptionManager:
    """Orchestrates obstacle detection and lane detection, updates WorldState."""
    
    def __init__(
        self,
        vehicle,
        world,
        sensor_suite,
        world_state: WorldState,
        use_ground_truth: bool = True,
        enable_lane_detection: bool = True,
        visualize_options: Optional[set] = None,
        debug: bool = False,
    ):
        self.vehicle = vehicle
        self.world = world
        self.sensor_suite = sensor_suite
        self.world_state = world_state
        self.visualize_options = visualize_options or set()
        self.use_ground_truth = use_ground_truth
        
        # Read perception config
        model_path = PERCEPTION['object_detector']['model_path']
        conf_threshold = PERCEPTION['object_detector']['conf_threshold']
        
        # Ground Truth detector (FakePerception) - always available
        self._ground_truth_detector = FakePerception(vehicle, world)
        
        # Real Perception detector (YOLO + LiDAR) - only initialized if not using ground truth
        self._real_detector = None
        if not use_ground_truth:
            self._real_detector = Perception(
                vehicle=vehicle, world=world, sensor_suite=sensor_suite,
                model_path=model_path,
                conf_threshold=conf_threshold
            )
            print("[Perception] Real perception enabled (YOLO + LiDAR)")
        
        print(f"[Perception] Initialized with use_ground_truth={use_ground_truth}")
        
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
    
    def run_step(self) -> None:
        """
        Execute one perception cycle and update WorldState.
        
        Updates ground_truth_obstacles always.
        Updates perceived_obstacles only if real perception is enabled.
        """
        # Always run ground truth detector
        ground_truth_obstacles = self._ground_truth_detector.detect_obstacles(radius=PERCEPTION['obstacle_radius'])
        self.world_state.update_ground_truth(ground_truth_obstacles or [])
        
        # Run real detector only if initialized
        perceived_obstacles = []
        if self._real_detector:
            perceived_obstacles = self._real_detector.detect_obstacles(radius=PERCEPTION['obstacle_radius'])
            self.world_state.update_perception(perceived_obstacles or [])
        
        # Visualize the active obstacles (based on use_ground_truth flag)
        active_obstacles = ground_truth_obstacles if self.use_ground_truth else perceived_obstacles
        
        if self._bbox_visualizer and active_obstacles:
            self._visualize_obstacles(active_obstacles)
        
        if active_obstacles:
            self._print_obstacles(active_obstacles)
        
        lanes_result = None
        if self._lane_detector:
            semantic_sensor = self.sensor_suite.get_sensor('semantic')
            if semantic_sensor:
                semantic_image = semantic_sensor.get_data()
                if semantic_image is not None:
                    lanes_result = self._lane_detector.detect(semantic_image)
                    if 'lane' in self.visualize_options or 'all' in self.visualize_options:
                        self.show_lane_debug(lanes_result)
        
        # Update lane detection in WorldState
        if lanes_result is not None:
            self.world_state.update_lane_detection(lanes_result)
    
    def _visualize_obstacles(self, obstacles: List[Obstacle]) -> None:
        """Draw bounding boxes for detected obstacles."""
        for obs in obstacles:
            # Use the yaw stored in the obstacle (relative to ego, in radians)
            # Convert to degrees for CARLA visualizer
            relative_yaw_deg = math.degrees(obs.yaw)
            
            color = carla.Color(255, 0, 0) if obs.obstacle_type == 'vehicle' else carla.Color(0, 255, 0)
            dimensions = carla.Vector3D(x=obs.length, y=obs.width, z=obs.height)
            
            self._bbox_visualizer.draw_bounding_box(
                relative_location={'x': obs.x, 'y': obs.y, 'z': obs.z},
                dimensions=dimensions,
                yaw=relative_yaw_deg,
                color=color,
                life_time=0.5,
                thickness=0.1
            )
    
    def _print_obstacles(self, obstacles: List[Obstacle]) -> None:
        """Print obstacle detection results for debugging."""
        print(f"\n[Perception Manager] Detected {len(obstacles)} obstacles:")
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
    
    def get_images(self) -> dict:
        """
        Get all perception images for display.
        
        Returns:
            Dict of image name -> image data for visualization.
        """
        images = {}
        
        # Get YOLO annotated image and masked image from real detector (if enabled)
        if self._real_detector:
            annotated = self._real_detector.get_annotated_image()
            if annotated is not None:
                images['yolo_detections'] = annotated
            
            masked = self._real_detector.get_masked_image()
            if masked is not None:
                images['yolo_input'] = masked
        
        return images
    
    def close(self) -> None:
        """Clean up perception resources."""
        if self._lane_debug_window:
            self._lane_debug_window.close()
