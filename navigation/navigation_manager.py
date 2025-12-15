"""
Navigation Manager
==================

Orchestrates autonomous navigation from point A to point B.
Integrates global planning, behavior planning, and obstacle avoidance.
Supports both MPC and PID control strategies.
"""

import math
import numpy as np
import carla
from planning.global_planner import GlobalPlanner, RoadOption
from planning.local_planner import LocalPlanner
from planning.behavior_planner import BehaviorPlanner, BehaviorState
from perception.obstacle_detector import ObstacleDetector
from control.vehicle_controller import VehicleController
from config.config import CONTROL
from visualization.live_2d_visualizer import visualizer as live_visualizer
from config.settings import Settings


class NavigationManager:
    """
    Manages autonomous navigation from origin to destination.
    
    Integrates:
    - Global Planner: Route planning from A to B
    - Local Planner: Waypoint-level planning and lane changes
    - Behavior Planner: Decision making (lane follow, lane change, stop)
    - Obstacle Detector: Perception for obstacles ahead
    - Vehicle Controller: Low-level vehicle control (MPC or PID)
    """
    
    def __init__(self, vehicle, world, sensor_suite=None,
                 controller_type='mpc', visualize_options=None):
        """
        Initialize navigation manager.
        
        Args:
            vehicle: CARLA vehicle actor
            world: CARLA world instance
            sensor_suite: Optional SensorSuite instance for perception
            controller_type: 'mpc' for Model Predictive Control, 'pid' for PID control
            visualize_options: Set of visualization tags to draw in simulator
        """
        self.vehicle = vehicle
        self.world = world
        self.map = world.get_map()
        self.sensor_suite = sensor_suite
        self.visualize_options = set(visualize_options) if visualize_options else set()
        self.controller_type = controller_type
        
        # Initialize planners
        self.global_planner = GlobalPlanner(self.map, sampling_resolution=2.0)
        self.local_planner = LocalPlanner(vehicle, self.map)
        self.behavior_planner = BehaviorPlanner(self.local_planner)
        
        # Initialize perception
        self.obstacle_detector = ObstacleDetector()
        
        # Initialize controller with selected type
        self.controller = VehicleController(vehicle, controller_type=controller_type)
        
        # Navigation state
        self.target_speed = CONTROL['planning']['target_speed']
        self.destination = None
        self.is_navigating = False
        self._route_planned = False  # Route computed but not yet started
        self._route_drawn = False
        self._destination_reached = False  # Track if we've arrived
        
        # Behavior integration state
        self._current_behavior = BehaviorState.LANE_FOLLOWING
        self._previous_behavior = BehaviorState.LANE_FOLLOWING
        self._lane_change_speed_factor = 0.7  # Slow down during lane change
        self._obstacle_detection = None
        self.lane_detection_result = None
        
        # MPC-specific: waypoint lookahead for trajectory
        self._mpc_lookahead_waypoints = 30  # Number of waypoints to feed MPC
        
        print(f"[Navigation] Initialized with {controller_type.upper()} controller")
        # Start live visualizer if requested
        if 'live' in self.visualize_options or 'all' in self.visualize_options:
            try:
                live_visualizer.start()
                print("[Navigation] Live 2D visualizer started")
            except Exception as e:
                print(f"[Navigation] Failed to start live visualizer: {e}")
    
    def update_perception(self, lane_detection_result):
        """Update lane detection result from perception system."""
        self.lane_detection_result = lane_detection_result
    
    def plan_route(self, destination):
        """Plan route to destination without starting navigation."""
        if isinstance(destination, carla.Transform):
            destination = destination.location
        elif isinstance(destination, (tuple, list)):
            destination = carla.Location(x=destination[0], y=destination[1], 
                                         z=destination[2] if len(destination) > 2 else 0)
        
        self.destination = destination
        self._destination_reached = False  # Reset for new route
        origin = self.vehicle.get_location()
        
        success = self.global_planner.set_destination(origin, destination)
        
        if success:
            self._route_planned = True
            self.is_navigating = False  # Don't start yet
            self.behavior_planner.reset()  # Reset behavior state for new route
            if self._should_draw('route'):
                self._draw_route()
            print("[Navigation] Route planned. Press N to start driving.")
        else:
            self._route_planned = False
        
        return success
    
    def plan_route_by_index(self, spawn_index):
        """Plan route using spawn point index without starting navigation."""
        spawn_points = self.map.get_spawn_points()
        if spawn_index < 0 or spawn_index >= len(spawn_points):
            print(f"[Navigation] Invalid index. Valid: 0-{len(spawn_points)-1}")
            return False
        return self.plan_route(spawn_points[spawn_index].location)
    
    def start_navigation(self):
        """Start driving the planned route."""
        if not self._route_planned:
            print("[Navigation] No route planned. Call plan_route() first.")
            return False
        
        self.is_navigating = True
        print("[Navigation] Navigation started!")
        return True
    
    def is_route_planned(self):
        """Check if a route has been planned."""
        return self._route_planned
    
    def set_destination(self, destination):
        """Set navigation destination, compute route, and start immediately."""
        if isinstance(destination, carla.Transform):
            destination = destination.location
        elif isinstance(destination, (tuple, list)):
            destination = carla.Location(x=destination[0], y=destination[1], 
                                         z=destination[2] if len(destination) > 2 else 0)
        
        self.destination = destination
        origin = self.vehicle.get_location()
        
        success = self.global_planner.set_destination(origin, destination)
        
        if success:
            self._route_planned = True
            self.is_navigating = True
            self.behavior_planner.reset()  # Reset behavior state for new route
            if self._should_draw('route'):
                self._draw_route()
        else:
            self.is_navigating = False
            self._route_planned = False
        
        return success
    
    def _replan_route_from_current(self):
        """
        Replan the global route from current vehicle position.
        Used after lane changes to get waypoints in the new lane.
        """
        if self.destination is None:
            return False
        
        origin = self.vehicle.get_location()
        success = self.global_planner.set_destination(origin, self.destination)
        
        if success:
            if self._should_draw('route'):
                self._draw_route()
            print("[Navigation] Route replanned from current position")
        
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
        """Draw route with short lifetime - refreshed each frame so they disappear when script stops."""
        self.global_planner.draw_route(self.world, life_time=1.0)
        self._route_drawn = True
    
    def _clear_debug_drawings(self):
        """Clear all debug drawings by redrawing with 0 lifetime."""
        if self._route_drawn:
            # Redraw with 0 lifetime to clear
            self.global_planner.draw_route(self.world, life_time=0.01)
            self._route_drawn = False
    
    def cleanup(self):
        """Clean up navigation resources and debug drawings."""
        self._clear_debug_drawings()
        # Stop live visualizer if running
        try:
            live_visualizer.stop()
        except Exception:
            pass

    def _should_draw(self, tag):
        """Check if a specific visualization tag should be drawn."""

        if not self.visualize_options or 'all' in self.visualize_options:
            return True
        return tag in self.visualize_options
    
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
            # If destination was reached, keep applying brakes
            if self._destination_reached:
                return self.controller.emergency_stop()
            return None
        
        vehicle_location = self.vehicle.get_location()
        
        # Refresh route visualization (short lifetime so it disappears when script stops)
        if self._route_drawn:
            self._draw_route()
        
        # Check if destination reached
        if self.global_planner.is_destination_reached(vehicle_location, threshold=5.0):
            print("[Navigation] Destination reached!")
            self.is_navigating = False
            self._destination_reached = True
            self._clear_debug_drawings()
            return self.controller.emergency_stop()
        
        # Run obstacle detection
        self._obstacle_detection = self._detect_obstacles()
        
        # Update behavior planner with obstacle detection
        self._previous_behavior = self._current_behavior
        self._current_behavior = self.behavior_planner.get_state()
        self.behavior_planner.update(self._obstacle_detection)
        
        # Check if lane change just completed - replan route from new lane
        if (self._previous_behavior in [BehaviorState.LANE_CHANGING_LEFT, 
                                         BehaviorState.LANE_CHANGING_RIGHT] and 
            self._current_behavior == BehaviorState.LANE_FOLLOWING):
            print("[Navigation] Lane change completed - replanning route")
            self._replan_route_from_current()
        
        # Emergency stop check
        if self._current_behavior == BehaviorState.EMERGENCY_STOP:
            print("[Navigation] Emergency stop triggered")
            return self.controller.emergency_stop()
        
        # Determine effective target speed based on behavior
        if self._current_behavior in [BehaviorState.LANE_CHANGING_LEFT, 
                                       BehaviorState.LANE_CHANGING_RIGHT]:
            effective_speed = self.target_speed * self._lane_change_speed_factor
        else:
            effective_speed = self.target_speed
        
        # Advance global planner to keep state in sync
        self.global_planner.advance(vehicle_location)
        
        # Get waypoints for controller based on mode
        waypoints = self._get_waypoints_for_control(vehicle_location)
        
        if not waypoints:
            self.is_navigating = False
            self._clear_route()
            return self.controller.emergency_stop()
        
        # Update controller with waypoints
        self.controller.set_waypoints(waypoints)
        
        # Draw debug visualizations
        self.handle_visualization(waypoints, vehicle_location)
        
        return self.controller.run_step(effective_speed)
    
    def _get_waypoints_for_control(self, vehicle_location):
        """
        Get waypoints for controller based on current mode.
        
        Decides between perception-based waypoints and map-based waypoints.
        
        Returns:
            List of waypoint objects or None
        """
        # Check if we should use perception-based control
        use_perception = self._should_use_perception(vehicle_location)
        
        if use_perception:
            # Generate waypoints from lane detection
            waypoints = self.local_planner.generate_waypoints_from_lane_fits(self.lane_detection_result)
            if waypoints:
                return waypoints
            # Fall through to map-based if perception fails
        
        # Map-based waypoints
        if self._current_behavior in [BehaviorState.LANE_CHANGING_LEFT, 
                                       BehaviorState.LANE_CHANGING_RIGHT]:
            return self._get_lane_change_waypoints()
        else:
            return self._get_lookahead_waypoints()
    
    def _should_use_perception(self, vehicle_location):
        """
        Check if perception-based control should be used.
        
        Criteria:
        1. Lane detection is valid and high confidence
        2. Not in a junction or sharp turn
        3. Not changing lanes
        4. Controller supports it (PID)
        """
        if not (self.lane_detection_result and 
                self.lane_detection_result['valid'] and 
                self.lane_detection_result['confidence'] > 0.5 and
                self._current_behavior == BehaviorState.LANE_FOLLOWING and
                self.controller_type == 'pid'):
            return False
        
        # Check road geometry
        road_option = self.global_planner.get_current_road_option()
        current_wp = self.map.get_waypoint(vehicle_location)
        
        if current_wp.is_junction:
            return False
        
        if road_option in [RoadOption.LEFT, RoadOption.RIGHT, 
                           RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]:
            return False
        
        return True
    
    def handle_visualization(self, waypoints, vehicle_location):
        """Draw all debug visualizations."""
        # Draw waypoints
        if self._should_draw('waypoints') or self._should_draw('route'):
            self._draw_waypoints(waypoints)
        
        # Draw perception mode indicator
        if self._should_use_perception(vehicle_location):
            self.world.debug.draw_string(
                vehicle_location + carla.Location(z=2.0), 
                "PERCEPTION CONTROL", 
                color=carla.Color(0, 255, 0),
                life_time=0.1
            )
        
        # Draw MPC trajectory if using MPC
        if self.controller_type == 'mpc':
            self._draw_mpc_trajectory()
            self._draw_coordinate_axes()  # Debug: show coordinate system
        
        # Draw Pure Pursuit lookahead point if using PID
        if self.controller_type == 'pid':
            self._draw_pure_pursuit_target()
        
        # Draw obstacle detection visualization
        if self._obstacle_detection and self._obstacle_detection.get('obstacle_detected'):
            self._draw_obstacle_warning()

        # Live 2D visualizer update
        if 'live' in self.visualize_options or 'all' in self.visualize_options:
            try:
                vehicle_state = self.controller.get_current_state()
                ref = None
                pred = None
                if self.controller.is_using_mpc():
                    mpc_info = self.controller.get_mpc_info()
                    if mpc_info:
                        ref = mpc_info.get('reference_trajectory', None)
                        pred = mpc_info.get('predicted_trajectory', None)
                # Convert waypoints (carla.Waypoint) to simple (x,y) tuples for visualizer
                wp_pairs = None
                if waypoints:
                    wp_pairs = []
                    for w in waypoints:
                        if hasattr(w, 'transform') and w.transform is not None:
                            wp_pairs.append((w.transform.location.x, w.transform.location.y))
                        elif isinstance(w, (tuple, list)) and len(w) >= 2:
                            wp_pairs.append((float(w[0]), float(w[1])))
                live_visualizer.update(vehicle_state, wp_pairs, ref, pred)
            except Exception as e:
                print(f"[Navigation] Live visualizer update failed: {e}")
    
    def _get_lookahead_waypoints(self):
        """
        Get multiple waypoints ahead for MPC trajectory planning.
        
        Returns:
            List of carla.Waypoint objects
        """
        # Use global planner's built-in method
        upcoming = self.global_planner.get_upcoming_waypoints(self._mpc_lookahead_waypoints)
        
        if not upcoming:
            return []
        
        # Extract just the waypoints (not road options)
        waypoints = [wp for wp, _ in upcoming]
        
        return waypoints
    
    def _get_lane_change_waypoints(self):
        """
        Get waypoints during lane change maneuver.
        
        Returns:
            List of carla.Waypoint objects
        """
        # Get remaining lane change waypoints
        lc_waypoints = self.behavior_planner.lane_change_waypoints
        lc_index = self.behavior_planner.lane_change_index
        
        if not lc_waypoints or lc_index >= len(lc_waypoints):
            # Lane change waypoints exhausted - get waypoints from current position
            return self._get_waypoints_from_current_position()
        
        # Get remaining lane change waypoints
        remaining_lc = lc_waypoints[lc_index:]
        
        # If we have few remaining waypoints, extend from the last lane change waypoint
        # This ensures continuity in the NEW lane, not the old global route
        if len(remaining_lc) < self._mpc_lookahead_waypoints:
            # Generate additional waypoints from the last lane change waypoint
            last_lc_wp = lc_waypoints[-1]
            additional = self._extend_waypoints_from(last_lc_wp, 
                                                      self._mpc_lookahead_waypoints - len(remaining_lc))
            return remaining_lc + additional
        
        return remaining_lc[:self._mpc_lookahead_waypoints]
    
    def _get_waypoints_from_current_position(self):
        """
        Generate waypoints from the vehicle's current position.
        Used when lane change waypoints are exhausted.
        
        Returns:
            List of carla.Waypoint objects
        """
        vehicle_location = self.vehicle.get_location()
        current_wp = self.map.get_waypoint(vehicle_location)
        
        if current_wp is None:
            return []
        
        return self._extend_waypoints_from(current_wp, self._mpc_lookahead_waypoints)
    
    def _extend_waypoints_from(self, start_waypoint, count):
        """
        Generate waypoints extending from a starting waypoint.
        
        Args:
            start_waypoint: Starting carla.Waypoint
            count: Number of waypoints to generate
            
        Returns:
            List of carla.Waypoint objects
        """
        waypoints = []
        wp = start_waypoint
        spacing = CONTROL['planning']['waypoint_spacing']
        
        for _ in range(count):
            next_wps = wp.next(spacing)
            if next_wps:
                waypoints.append(next_wps[0])
                wp = next_wps[0]
            else:
                break
        
        return waypoints
    
    def _convert_obstacles_to_world(self):
        """
        Convert detected obstacles from camera frame to world coordinates.
        
        Returns:
            List of [x, y] world coordinates for obstacles
        """
        obstacles = []
        
        if not self._obstacle_detection or not self._obstacle_detection.get('obstacle_detected'):
            return obstacles
        
        # Get obstacle distance and estimate world position
        distance = self._obstacle_detection.get('distance', float('inf'))
        
        if distance < float('inf'):
            vehicle_transform = self.vehicle.get_transform()
            forward = vehicle_transform.get_forward_vector()
            
            # Estimate obstacle world position (straight ahead)
            obstacle_x = vehicle_transform.location.x + forward.x * distance
            obstacle_y = vehicle_transform.location.y + forward.y * distance
            
            obstacles.append([obstacle_x, obstacle_y])
            
            # If we have lateral offset info from detection
            center = self._obstacle_detection.get('center')
            if center is not None:
                # center[0] is horizontal pixel position
                # Estimate lateral offset (assuming 90 degree FOV camera)
                image_width = 800  # Default camera width
                lateral_offset = (center[0] - image_width / 2) / image_width * distance * 0.8
                
                right = vehicle_transform.get_right_vector()
                obstacle_x += right.x * lateral_offset
                obstacle_y += right.y * lateral_offset
                
                obstacles[0] = [obstacle_x, obstacle_y]
        
        return obstacles
    
    def _draw_mpc_trajectory(self):
        """Draw MPC predicted trajectory for visualization."""
        mpc_info = self.controller.get_mpc_info()
        if mpc_info is None:
            return
        
        predicted_trajectory = mpc_info.get('predicted_trajectory')
        if predicted_trajectory is None or len(predicted_trajectory) == 0:
            return

        # Print solve time
        solve_time = mpc_info.get('solve_time_ms', 0)
        print(f"[MPC] Solve time: {solve_time:.2f} ms")

        # Print a concise snapshot of XY + heading coordinates
        try:
            max_print = min(10, len(predicted_trajectory))
            lines = [
                f"{i:02d}: x={float(state[0]):7.2f}, y={float(state[1]):7.2f}, psi={float(state[2])*180/3.14159:6.1f}°"
                for i, state in enumerate(predicted_trajectory[:max_print])
            ]
            print(f"[MPC] Predicted trajectory (first {max_print}/{len(predicted_trajectory)}):\n  " + "\n  ".join(lines))
        except Exception:
            pass
        
        # Draw predicted trajectory points with longer lifetime
        life_time = 2.0  # 2 seconds so we can see the path clearly
        for i, state in enumerate(predicted_trajectory):
            loc = carla.Location(x=state[0], y=state[1], z=0.5)
            # Color gradient: start=bright green, end=dark green
            intensity = int(255 * (1 - i / len(predicted_trajectory)))
            self.world.debug.draw_point(
                loc,
                size=0.15,
                color=carla.Color(0, intensity, 0),  # Green gradient
                life_time=life_time
            )
            
            # Draw heading arrow at every 5th point
            if i % 5 == 0 and len(state) > 2:
                psi = state[2]
                arrow_len = 1.0
                end_loc = carla.Location(
                    x=state[0] + arrow_len * float(np.cos(psi)),
                    y=state[1] + arrow_len * float(np.sin(psi)),
                    z=0.5
                )
                self.world.debug.draw_arrow(
                    loc, end_loc,
                    thickness=0.05,
                    arrow_size=0.1,
                    color=carla.Color(255, 255, 0),  # Yellow for heading
                    life_time=life_time
                )
        
        # Draw connecting lines between trajectory points
        for i in range(len(predicted_trajectory) - 1):
            loc1 = carla.Location(x=predicted_trajectory[i][0], y=predicted_trajectory[i][1], z=0.5)
            loc2 = carla.Location(x=predicted_trajectory[i+1][0], y=predicted_trajectory[i+1][1], z=0.5)
            self.world.debug.draw_line(
                loc1, loc2,
                thickness=0.03,
                color=carla.Color(0, 200, 0),
                life_time=life_time
            )
        
        # Draw REFERENCE trajectory (CYAN) - what MPC is trying to track
        reference_trajectory = mpc_info.get('reference_trajectory')
        if reference_trajectory is not None and len(reference_trajectory) > 0:
            for i, state in enumerate(reference_trajectory):
                loc = carla.Location(x=state[0], y=state[1], z=0.7)  # Slightly higher
                self.world.debug.draw_point(
                    loc,
                    size=0.12,
                    color=carla.Color(0, 255, 255),  # Cyan for reference
                    life_time=life_time
                )
                
                # Draw reference heading arrow at every 5th point
                if i % 5 == 0 and len(state) > 2:
                    psi = state[2]
                    arrow_len = 1.5
                    end_loc = carla.Location(
                        x=state[0] + arrow_len * float(np.cos(psi)),
                        y=state[1] + arrow_len * float(np.sin(psi)),
                        z=0.7
                    )
                    self.world.debug.draw_arrow(
                        loc, end_loc,
                        thickness=0.06,
                        arrow_size=0.15,
                        color=carla.Color(0, 200, 255),  # Cyan for reference heading
                        life_time=life_time
                    )
    
    def _draw_coordinate_axes(self):
        """Draw world coordinate axes at origin and vehicle location for debugging."""
        vehicle_transform = self.vehicle.get_transform()
        vehicle_loc = vehicle_transform.location
        
        # Axis length
        axis_len = 5.0
        life_time = 0.15
        
        # --- Draw axes at vehicle location ---
        # X axis (RED) - East in CARLA
        self.world.debug.draw_arrow(
            carla.Location(x=vehicle_loc.x, y=vehicle_loc.y, z=0.5),
            carla.Location(x=vehicle_loc.x + axis_len, y=vehicle_loc.y, z=0.5),
            thickness=0.08,
            arrow_size=0.2,
            color=carla.Color(255, 0, 0),  # RED = +X
            life_time=life_time
        )
        self.world.debug.draw_string(
            carla.Location(x=vehicle_loc.x + axis_len + 0.5, y=vehicle_loc.y, z=1.0),
            "+X (East)",
            color=carla.Color(255, 0, 0),
            life_time=life_time
        )
        
        # Y axis (GREEN) - South in CARLA (positive Y = right = south)
        self.world.debug.draw_arrow(
            carla.Location(x=vehicle_loc.x, y=vehicle_loc.y, z=0.5),
            carla.Location(x=vehicle_loc.x, y=vehicle_loc.y + axis_len, z=0.5),
            thickness=0.08,
            arrow_size=0.2,
            color=carla.Color(0, 255, 0),  # GREEN = +Y
            life_time=life_time
        )
        self.world.debug.draw_string(
            carla.Location(x=vehicle_loc.x, y=vehicle_loc.y + axis_len + 0.5, z=1.0),
            "+Y (South)",
            color=carla.Color(0, 255, 0),
            life_time=life_time
        )
        
        # Z axis (BLUE) - Up
        self.world.debug.draw_arrow(
            carla.Location(x=vehicle_loc.x, y=vehicle_loc.y, z=0.5),
            carla.Location(x=vehicle_loc.x, y=vehicle_loc.y, z=0.5 + axis_len),
            thickness=0.08,
            arrow_size=0.2,
            color=carla.Color(0, 0, 255),  # BLUE = +Z
            life_time=life_time
        )
        
        # --- Draw vehicle's forward direction (YAW) ---
        yaw_rad = np.radians(vehicle_transform.rotation.yaw)
        forward_len = 4.0
        forward_end = carla.Location(
            x=vehicle_loc.x + forward_len * np.cos(yaw_rad),
            y=vehicle_loc.y + forward_len * np.sin(yaw_rad),
            z=0.8
        )
        self.world.debug.draw_arrow(
            carla.Location(x=vehicle_loc.x, y=vehicle_loc.y, z=0.8),
            forward_end,
            thickness=0.12,
            arrow_size=0.3,
            color=carla.Color(255, 0, 255),  # MAGENTA = vehicle forward/yaw
            life_time=life_time
        )
        
        # Print vehicle state
        print(f"[DEBUG] Vehicle: x={vehicle_loc.x:.1f}, y={vehicle_loc.y:.1f}, yaw={vehicle_transform.rotation.yaw:.1f}° (CARLA)")
        print(f"[DEBUG] Forward dir: cos(yaw)={np.cos(yaw_rad):.3f}, sin(yaw)={np.sin(yaw_rad):.3f}")
    
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
    
    def _draw_pure_pursuit_target(self):
        """Draw Pure Pursuit lookahead target point (cyan/blue)."""
        pp_info = self.controller.get_pure_pursuit_info()
        if pp_info is None:
            return
        
        lookahead_point = pp_info.get('lookahead_point')
        if lookahead_point is None:
            return
        
        # Draw the lookahead target point in cyan
        loc = carla.Location(x=lookahead_point[0], y=lookahead_point[1], z=0.5)
        self.world.debug.draw_point(
            loc,
            size=0.25,
            color=carla.Color(0, 255, 255),  # Cyan for Pure Pursuit target
            life_time=0.1
        )
    
    def _draw_waypoints(self, waypoints):
        """Draw waypoints for controller visualization."""
        if not waypoints:
            return
        
        # Draw each waypoint as a small sphere
        for i, wp in enumerate(waypoints):
            loc = wp.transform.location
            loc_draw = carla.Location(x=loc.x, y=loc.y, z=loc.z + 0.3)
            
            # Color gradient: green (near) to yellow (far)
            ratio = i / max(len(waypoints) - 1, 1)
            color = carla.Color(
                int(255 * ratio),      # R: 0 -> 255
                int(255 * (1 - ratio * 0.3)),  # G: 255 -> 178
                0                       # B: 0
            )
            
            self.world.debug.draw_point(
                loc_draw,
                size=0.12,
                color=color,
                life_time=0.1
            )
        
        # Draw lines connecting waypoints
        for i in range(len(waypoints) - 1):
            loc1 = waypoints[i].transform.location
            loc2 = waypoints[i + 1].transform.location
            
            self.world.debug.draw_line(
                carla.Location(x=loc1.x, y=loc1.y, z=loc1.z + 0.3),
                carla.Location(x=loc2.x, y=loc2.y, z=loc2.z + 0.3),
                thickness=0.04,
                color=carla.Color(0, 255, 100),  # Green
                life_time=0.1
            )
    
    def get_status(self):
        """Get navigation status including behavior state and MPC info."""
        status = {
            'is_navigating': self.is_navigating,
            'progress': self.global_planner.get_progress(),
            'remaining_distance': self.global_planner.get_remaining_distance(),
            'target_speed': self.target_speed,
            'road_option': self.global_planner.get_current_road_option().name,
            'behavior_state': self.behavior_planner.get_state_name(),
            'controller_type': self.controller_type,
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
        
        # Add MPC-specific info
        if self.controller_type == 'mpc':
            mpc_info = self.controller.get_mpc_info()
            if mpc_info:
                status['mpc_solve_time'] = mpc_info.get('solve_time_ms', 0)
                status['mpc_cost'] = mpc_info.get('cost', 0)
                status['mpc_feasible'] = mpc_info.get('feasible', False)
        
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
