"""
Navigation Manager
==================

Orchestrates autonomous navigation from point A to point B.
Integrates global planning, behavior planning, and obstacle avoidance.
"""

import carla
from planning.global_planner import GlobalPlanner, RoadOption
from planning.local_planner import LocalPlanner
from planning.behavior_planner import BehaviorPlanner, BehaviorState
from perception.obstacle_detector import ObstacleDetector
from control import VehicleController
from config import Settings


class NavigationManager:
    """
    Manages autonomous navigation from origin to destination.
    
    Integrates:
    - Global Planner: Route planning from A to B
    - Local Planner: Waypoint-level planning and lane changes
    - Behavior Planner: Decision making (lane follow, lane change, stop)
    - Obstacle Detector: Perception for obstacles ahead
    - Vehicle Controller: Low-level vehicle control
    """
    
    def __init__(self, vehicle, world, sensor_suite=None, draw_on_simulator=False):
        """
        Initialize navigation manager.
        
        Args:
            vehicle: CARLA vehicle actor
            world: CARLA world instance
            sensor_suite: Optional SensorSuite instance for perception
            draw_on_simulator: Whether to draw debug objects in the simulator
        """
        self.vehicle = vehicle
        self.world = world
        self.map = world.get_map()
        self.sensor_suite = sensor_suite
        self.draw_on_simulator = draw_on_simulator
        
        # Initialize planners
        self.global_planner = GlobalPlanner(self.map, sampling_resolution=2.0)
        self.local_planner = LocalPlanner(vehicle, self.map)
        self.behavior_planner = BehaviorPlanner(self.local_planner)
        
        # Initialize perception
        self.obstacle_detector = ObstacleDetector()
        
        # Initialize controller
        self.controller = VehicleController(vehicle)
        
        # Navigation state
        self.target_speed = Settings.TARGET_SPEED
        self.destination = None
        self.is_navigating = False
        self._route_drawn = False
        
        # Behavior integration state
        self._current_behavior = BehaviorState.LANE_FOLLOWING
        self._lane_change_speed_factor = 0.7  # Slow down during lane change
        self._obstacle_detection = None
        
        print("[Navigation] Initialized with behavior planner integration")
    
    def set_destination(self, destination):
        """Set navigation destination and compute route."""
        if isinstance(destination, carla.Transform):
            destination = destination.location
        elif isinstance(destination, (tuple, list)):
            destination = carla.Location(x=destination[0], y=destination[1], 
                                         z=destination[2] if len(destination) > 2 else 0)
        
        self.destination = destination
        origin = self.vehicle.get_location()
        
        success = self.global_planner.set_destination(origin, destination)
        
        if success:
            self.is_navigating = True
            self.behavior_planner.reset()  # Reset behavior state for new route
            if self.draw_on_simulator:
                self._draw_route()
        else:
            self.is_navigating = False
        
        return success
    
    def set_sensor_suite(self, sensor_suite):
        """
        Set sensor suite for perception.
        
        Args:
            sensor_suite: SensorSuite instance
        """
        self.sensor_suite = sensor_suite
        print("[Navigation] Sensor suite connected")
    
    def _draw_route(self):
        """Draw route with short lifetime - needs to be refreshed each frame."""
        self.global_planner.draw_route(self.world, life_time=0.5)
        self._route_drawn = True
    
    def _draw_current_waypoint(self, waypoint):
        """Draw the current waypoint being followed (larger, red)."""
        if waypoint is None:
            return
        self.world.debug.draw_point(
            waypoint.transform.location + carla.Location(z=0.5),
            size=0.2,
            color=carla.Color(255, 0, 0),  # Red
            life_time=1
        )
    
    def _clear_route(self):
        """Clear the drawn route by not refreshing it."""
        self._route_drawn = False
    
    def set_destination_by_index(self, spawn_index):
        """Set destination using spawn point index."""
        spawn_points = self.map.get_spawn_points()
        if spawn_index < 0 or spawn_index >= len(spawn_points):
            print(f"[Navigation] Invalid index. Valid: 0-{len(spawn_points)-1}")
            return False
        return self.set_destination(spawn_points[spawn_index].location)
    
    def _get_perception_data(self):
        """
        Get perception data from sensors.
        
        Returns:
            Tuple of (semantic_labels, depth_meters) or (None, None)
        """
        if self.sensor_suite is None:
            return None, None
        
        # Get semantic segmentation camera
        semantic_camera = self.sensor_suite.get_sensor('semantic')
        depth_camera = self.sensor_suite.get_sensor('depth')
        
        semantic_labels = None
        depth_meters = None
        
        if semantic_camera is not None:
            semantic_data = semantic_camera.get_data()
            if semantic_data is not None:
                # Semantic camera may return 2D labels directly or 3D BGRA
                if len(semantic_data.shape) == 2:
                    semantic_labels = semantic_data
                elif len(semantic_data.shape) == 3:
                    # Semantic camera stores label in red channel (index 2 in BGRA)
                    semantic_labels = semantic_data[:, :, 2]
        
        if depth_camera is not None:
            # Depth camera already processes depth to meters via get_depth_meters()
            depth_meters = depth_camera.get_depth_meters()
        
        return semantic_labels, depth_meters
    
    def _detect_obstacles(self):
        """
        Run obstacle detection.
        
        Returns:
            Detection result dictionary
        """
        semantic_labels, depth_meters = self._get_perception_data()
        
        if semantic_labels is None or depth_meters is None:
            # No sensor data - assume clear
            return {
                'obstacle_detected': False,
                'distance': float('inf'),
                'position': None
            }
        
        return self.obstacle_detector.detect(semantic_labels, depth_meters)

    def run_step(self):
        """
        Execute one navigation step with behavior planning.
        
        Returns:
            carla.VehicleControl or None if navigation complete
        """
        if not self.is_navigating:
            return None
        
        vehicle_location = self.vehicle.get_location()
        
        # Refresh route visualization while navigating
        if self.draw_on_simulator and self._route_drawn:
            self._draw_route()
        
        # Check if destination reached
        if self.global_planner.is_destination_reached(vehicle_location, threshold=5.0):
            print("[Navigation] Destination reached!")
            self.is_navigating = False
            self._clear_route()
            return self.controller.emergency_stop()
        
        # Run obstacle detection
        self._obstacle_detection = self._detect_obstacles()
        
        # Update behavior planner with obstacle detection
        self._current_behavior = self.behavior_planner.get_state()
        behavior_waypoint = self.behavior_planner.update(self._obstacle_detection)
        
        # Get target waypoint based on current behavior
        if self._current_behavior in [BehaviorState.LANE_CHANGING_LEFT, 
                                       BehaviorState.LANE_CHANGING_RIGHT]:
            # During lane change, use behavior planner's waypoint
            target_wp = behavior_waypoint
            effective_speed = self.target_speed * self._lane_change_speed_factor
        elif self._current_behavior == BehaviorState.EMERGENCY_STOP:
            # Emergency stop
            print("[Navigation] Emergency stop triggered")
            return self.controller.emergency_stop()
        else:
            # Normal navigation - use global planner
            self.global_planner.advance(vehicle_location)
            target_wp, road_option = self.global_planner.get_next_waypoint()
            effective_speed = self.target_speed
        
        if target_wp is None:
            self.is_navigating = False
            self._clear_route()
            return self.controller.emergency_stop()
        
        # Draw the current waypoint being followed (only if draw_on_simulator enabled)
        if self.draw_on_simulator:
            self._draw_current_waypoint(target_wp)
            
            # Draw obstacle detection visualization
            if self._obstacle_detection and self._obstacle_detection.get('obstacle_detected'):
                self._draw_obstacle_warning()
        
        return self.controller.run_step(target_wp, effective_speed)
    
    def _draw_obstacle_warning(self):
        """Draw visual warning when obstacle detected."""
        if not self._obstacle_detection:
            return
        
        vehicle_transform = self.vehicle.get_transform()
        forward = vehicle_transform.get_forward_vector()
        distance = self._obstacle_detection.get('distance', 10)
        
        # Draw warning point ahead
        obstacle_loc = vehicle_transform.location + carla.Location(
            x=forward.x * distance,
            y=forward.y * distance,
            z=1.0
        )
        
        self.world.debug.draw_point(
            obstacle_loc,
            size=0.3,
            color=carla.Color(255, 0, 0),  # Red
            life_time=0.1
        )
    
    def get_status(self):
        """Get navigation status including behavior state."""
        status = {
            'is_navigating': self.is_navigating,
            'progress': self.global_planner.get_progress(),
            'remaining_distance': self.global_planner.get_remaining_distance(),
            'target_speed': self.target_speed,
            'road_option': self.global_planner.get_current_road_option().name,
            'behavior_state': self.behavior_planner.get_state_name(),
            'obstacle_detected': False,
            'obstacle_distance': float('inf'),
            'obstacle_position': None,
            'obstacle_center': None,
            'obstacle_bbox': None,
            'obstacle_type': None,
            'obstacle_pixels': 0
        }
        
        # Add obstacle detection details if available
        if self._obstacle_detection:
            status['obstacle_detected'] = self._obstacle_detection.get('obstacle_detected', False)
            status['obstacle_distance'] = self._obstacle_detection.get('distance', float('inf'))
            status['obstacle_position'] = self._obstacle_detection.get('position', None)
            status['obstacle_center'] = self._obstacle_detection.get('center', None)
            status['obstacle_bbox'] = self._obstacle_detection.get('bbox', None)
            status['obstacle_type'] = self._obstacle_detection.get('obstacle_type', None)
            status['obstacle_pixels'] = self._obstacle_detection.get('pixel_count', 0)
        
        return status
    
    def get_behavior_state(self):
        """Get current behavior state."""
        return self.behavior_planner.get_state()
    
    def get_behavior_state_name(self):
        """Get current behavior state as string."""
        return self.behavior_planner.get_state_name()
    
    def set_target_speed(self, speed_kmh):
        self.target_speed = speed_kmh
    
    def stop(self):
        """Stop navigation."""
        self.is_navigating = False
        self._clear_route()
        self.behavior_planner.reset()
        return self.controller.emergency_stop()
    
    def resume(self):
        """Resume navigation."""
        if self.global_planner.get_route():
            self.is_navigating = True
            if self.draw_on_simulator:
                self._draw_route()
    
    def reset_behavior(self):
        """Reset behavior planner to lane following."""
        self.behavior_planner.reset()
        print("[Navigation] Behavior reset to lane following")
    
    def force_lane_change(self, direction):
        """
        Force a lane change in specified direction.
        
        Args:
            direction: 'left' or 'right'
            
        Returns:
            True if lane change initiated, False otherwise
        """
        if not self.local_planner.can_change_lane(direction):
            print(f"[Navigation] Cannot change lane {direction} - no valid lane")
            return False
        
        waypoints = self.local_planner.generate_lane_change_waypoints(direction)
        if not waypoints:
            print(f"[Navigation] Failed to generate lane change path")
            return False
        
        # Manually set behavior planner state
        self.behavior_planner.lane_change_waypoints = waypoints
        self.behavior_planner.lane_change_index = 0
        if direction == 'left':
            self.behavior_planner.state = BehaviorState.LANE_CHANGING_LEFT
        else:
            self.behavior_planner.state = BehaviorState.LANE_CHANGING_RIGHT
        
        print(f"[Navigation] Forced lane change {direction}")
        return True
