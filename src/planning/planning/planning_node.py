#!/usr/bin/env python3
"""
Planning Node
=============

ROS2 node for path planning (Global + Local + Behavior).
Connects directly to CARLA for map access, receives vehicle state via ROS2 topics.

Subscribes:
- /carla/ego_vehicle/ground_truth/odometry (nav_msgs/Odometry) - Vehicle state
- /planning/goal_pose (geometry_msgs/PoseStamped) - Navigation goal
- /perception/obstacles (autoiq_msgs/ObstacleArray) - Detected obstacles

Publishes:
- /planning/trajectory (autoiq_msgs/Trajectory) - Planned trajectory for control
- /planning/global_route (nav_msgs/Path) - Full global route for visualization
- /planning/status (std_msgs/String) - Planning status
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path, Odometry, OccupancyGrid
from geometry_msgs.msg import PoseStamped, PointStamped
from std_msgs.msg import String, Int32, ColorRGBA
from visualization_msgs.msg import Marker

from autoiq_msgs.msg import ObstacleArray, Trajectory, TrajectoryPoint

# Import common types and converters
from autoiq_common.types import VehicleState
from autoiq_common.converters import vehicle_state_from_odometry

from typing import List

import carla

from planning.global_planner import GlobalPlanner, RoadOption
from planning.behavior_planner import (
    BehaviorPlanner,
    BehaviorPlannerConfig,
    BehaviorState,
    ObstacleInfo,
    LaneAvailability,
)

from planning.motion_planner import (
    MotionPlanner, 
    MotionPlannerConfig,
)
from planning.utils.corridor import CorridorGenerator

class PlanningNode(Node):
    """
    ROS2 Planning Node.
    
    Connects directly to CARLA for map access.
    Receives vehicle state and publishes trajectories via ROS2.
    """
    
    def __init__(self):
        super().__init__('planning_node')
        
        # ==================== Parameters ====================
        # CARLA connection
        self.declare_parameter('carla_host', 'localhost')
        self.declare_parameter('carla_port', 2000)
        self.declare_parameter('carla_timeout', 10.0)
        
        # Planning parameters
        self.declare_parameter('target_speed', 40.0)  # km/h
        self.declare_parameter('lookahead_distance', 15.0)
        self.declare_parameter('waypoint_spacing', 2.0)
        self.declare_parameter('lookahead_waypoints', 40)
        self.declare_parameter('lane_change_trigger_distance', 35.0)  # Base distance
        self.declare_parameter('lane_change_speed_factor', 0.7)
        self.declare_parameter('lane_change_time', 3.0)  # Time to complete lane change (seconds)
        self.declare_parameter('sampling_resolution', 2.0)
        
        # Perception mode - determines which obstacle topic to subscribe to
        self.declare_parameter('use_ground_truth', True)
        
        # Get CARLA parameters
        carla_host = self.get_parameter('carla_host').value
        carla_port = self.get_parameter('carla_port').value
        carla_timeout = self.get_parameter('carla_timeout').value
        
        # Get planning parameters
        self.target_speed = self.get_parameter('target_speed').value
        self.lookahead_distance = self.get_parameter('lookahead_distance').value
        self.waypoint_spacing = self.get_parameter('waypoint_spacing').value
        self.lookahead_waypoints = self.get_parameter('lookahead_waypoints').value
        self.lane_change_trigger_distance = self.get_parameter('lane_change_trigger_distance').value
        self.lane_change_speed_factor = self.get_parameter('lane_change_speed_factor').value
        self.lane_change_time = self.get_parameter('lane_change_time').value
        self.sampling_resolution = self.get_parameter('sampling_resolution').value
        
        # ==================== Connect to CARLA ====================
        self.get_logger().info(f'Connecting to CARLA at {carla_host}:{carla_port}...')
        
        try:
            self.carla_client = carla.Client(carla_host, carla_port)
            self.carla_client.set_timeout(carla_timeout)
            self.carla_world = self.carla_client.get_world()
            self.carla_map = self.carla_world.get_map()
            self.get_logger().info(f'Connected to CARLA, map: {self.carla_map.name}')
        except Exception as e:
            self.get_logger().error(f'Failed to connect to CARLA: {e}')
            raise RuntimeError(f'CARLA connection failed: {e}')
        
        # ==================== Initialize Planners ====================
        self.global_planner = GlobalPlanner(
            self.carla_map, 
            sampling_resolution=self.sampling_resolution
        )
        
        # Initialize Behavior Planner (state machine with TTC-based detection)
        behavior_config = BehaviorPlannerConfig(
            lane_width=3.5,
            lane_change_time=self.lane_change_time,
            ttc_threshold=4.0,  # Trigger if TTC < 4 seconds
            ttc_critical=2.0,   # Immediate action if TTC < 2 seconds
            min_trigger_distance=10.0,
            max_trigger_distance=60.0,
            prepare_timeout=2.0,
            lane_change_cooldown=3.0,
        )
        self.behavior_planner = BehaviorPlanner(config=behavior_config)
        
        # Initialize Motion Planner (trajectory generation)
        motion_config = MotionPlannerConfig(
            lane_width=3.5,
            lane_change_time=self.lane_change_time,
            longitudinal_horizon=4.0,
            dt=0.1,
        )
        self.motion_planner = MotionPlanner(config=motion_config)
        
        # Initialize Corridor Generator
        self.corridor_generator = CorridorGenerator(
            resolution=self.sampling_resolution, # Will update with map resolution
            min_skeleton_points=10
        )
        
        # ==================== State ====================
        self.initialized = True
        self.is_navigating = False
        self.destination_reached = False
        self._destination_carla: carla.Location = None  # Destination in CARLA left-handed coords (Y-flipped from ROS)
        
        # Vehicle state (from odometry) - stored as VehicleState dataclass
        self.vehicle_state: VehicleState = VehicleState()
        
        # Vehicle position in CARLA left-handed coords (Y-flipped from ROS) for map queries
        self.vehicle_location_carla: carla.Location = None
        
        # Obstacle detection (from perception) - list of all obstacles
        self.current_obstacles: List[ObstacleInfo] = []
        
        # Corridor state (from local costmap)
        self.corridor = {
            'x_spline': None,
            'y_spline': None,
            's_max': None,
            'grid': None,
            'origin_x': 0.0,
            'origin_y': 0.0,
            'valid': False,
            'last_update': None,
        }
        
        # ==================== Subscribers ====================
        # Subscribe to odometry (convert to VehicleState in callback)
        self.odom_sub = self.create_subscription(
            Odometry, '/carla/ego_vehicle/ground_truth/odometry',
            self.odom_callback, 10)
        
        self.goal_sub = self.create_subscription(
            PoseStamped, '/planning/goal_pose',
            self.goal_callback, 10)
        
        self.goal_index_sub = self.create_subscription(
            Int32, '/planning/goal_index',
            self.goal_index_callback, 10)
        
        # Subscribe to obstacle topic based on use_ground_truth parameter
        use_ground_truth = self.get_parameter('use_ground_truth').value
        if use_ground_truth:
            obstacle_topic = '/perception/ground_truth/obstacles'
        else:
            obstacle_topic = '/perception/obstacles'
        self.get_logger().info(f'Subscribing to obstacles on: {obstacle_topic}')
        
        self.obstacles_sub = self.create_subscription(
            ObstacleArray, obstacle_topic,
            self.obstacles_callback, 10)
            
        self.grid_sub = self.create_subscription(
            OccupancyGrid, '/mapping/local_costmap',
            self.grid_callback, 10
        )
        
        # ==================== Publishers ====================
        self.trajectory_pub = self.create_publisher(Trajectory, '/planning/trajectory', 10)
        self.global_route_pub = self.create_publisher(Path, '/planning/global_route', 10)
        self.current_waypoint_pub = self.create_publisher(PointStamped, '/planning/current_waypoint', 10)
        self.status_pub = self.create_publisher(String, '/planning/status', 10)
        self.corridor_pub = self.create_publisher(Path, '/planning/corridor_reference', 10)
        self.corridor_start_pub = self.create_publisher(Marker, '/planning/corridor_start', 10)
        self.corridor_end_pub = self.create_publisher(Marker, '/planning/corridor_end', 10)
        self.motion_goal_pub = self.create_publisher(Marker, '/planning/motion_goal', 10)
        
        # ==================== Timer ====================
        self.create_timer(0.1, self.planning_loop)  # 10 Hz
        
        self.get_logger().info('Planning Node initialized successfully!')
    
    # ==================== Callbacks ====================

    def odom_callback(self, msg: Odometry):
        """
        Update vehicle state from odometry.
        
        Converts Odometry message to VehicleState dataclass.
        
        Args:
            msg: Odometry message from CARLA
        """
        # Convert Odometry to VehicleState
        state = vehicle_state_from_odometry(msg)
        self.vehicle_state = state
        
        # Also store CARLA-format location for map queries (Y-flipped)
        self.vehicle_location_carla = carla.Location(
            x=state.x,
            y=-state.y,  # Y-axis inversion for CARLA
            z=state.z    # Use actual vehicle height for proper waypoint matching
        )
    
    def goal_callback(self, msg: PoseStamped):
        """
        Set navigation goal from PoseStamped (ROS coordinates).
        
        Converts from ROS right-handed to CARLA left-handed coordinates.
        """
        if self.vehicle_location_carla is None:
            self.get_logger().warn('Cannot set goal - no vehicle position')
            return
        
        # Convert goal to CARLA location (Y-flipped)
        goal_location_carla = carla.Location(
            x=msg.pose.position.x,
            y=-msg.pose.position.y,  # Y-axis inversion
            z=msg.pose.position.z
        )
        
        self._set_destination(goal_location_carla, 'goal pose')
    
    def goal_index_callback(self, msg: Int32):
        """
        Set navigation goal using CARLA spawn point index.
        
        Spawn points are already in CARLA left-handed coordinates.
        """
        if self.vehicle_location_carla is None:
            self.get_logger().warn('Cannot set goal - no vehicle position')
            return
        
        spawn_index = msg.data
        spawn_points = self.carla_map.get_spawn_points()
        
        if spawn_index < 0 or spawn_index >= len(spawn_points):
            self.get_logger().error(f'Invalid spawn index {spawn_index}. Valid range: 0-{len(spawn_points)-1}')
            return
        
        goal_location_carla = spawn_points[spawn_index].location
        self._set_destination(goal_location_carla, f'spawn point {spawn_index}')
    
    def _set_destination(self, goal_location_carla: carla.Location, description: str):
        """
        Set navigation destination and compute route.
        
        Args:
            goal_location_carla: Destination in CARLA left-handed coordinates
            description: Human-readable description for logging
        """
        self._destination_carla = goal_location_carla  # Store for potential replanning
        
        success = self.global_planner.set_destination(
            self.vehicle_location_carla,
            goal_location_carla
        )
        
        if success:
            self.is_navigating = True
            self.destination_reached = False
            self.get_logger().info(f'Route computed to {description}')
            self._publish_global_route()
        else:
            self.get_logger().error(f'Failed to compute route to {description}')
    
    def obstacles_callback(self, msg: ObstacleArray):
        """
        Process detected obstacles from perception.
        
        Uses the new ObstacleArray message with explicit velocity fields.
        """
        self.current_obstacles = []
        
        if not msg.obstacles:
            return
        
        for obstacle in msg.obstacles:
            # Position (in ego frame)
            x = obstacle.pose.position.x
            y = obstacle.pose.position.y
            z = obstacle.pose.position.z
            
            dist = math.sqrt(x**2 + y**2 + z**2)
            
            # Velocity (in ego frame)
            vx = obstacle.velocity.x  # Longitudinal velocity
            vy = obstacle.velocity.y  # Lateral velocity
            
            # Store obstacle with velocity info
            self.current_obstacles.append(
                ObstacleInfo(x=x, y=y, distance=dist, vx=vx, vy=vy)
            )
    
    # ==================== Planning Loop ====================
    
    def planning_loop(self):
        """Main planning loop - delegates to MotionPlanner."""
        if self.vehicle_location_carla is None:
            return
        
        # Publish status
        self._publish_status()
        
        if not self.is_navigating:
            self._publish_empty_trajectory()
            return

        # Check if destination reached
        if self.global_planner.is_destination_reached(self.vehicle_location_carla, threshold=5.0):
            self.get_logger().info('Destination reached!')
            self.is_navigating = False
            self.destination_reached = True
            self._publish_empty_trajectory()
            return
        
        # === STRICT COSTMAP DEPENDENCY ===
        # Check if corridor data is valid and not stale
        costmap_timeout = 1.0  # seconds
        if not self.corridor['valid']:
            self.get_logger().warn('No valid corridor - publishing stop trajectory', throttle_duration_sec=1.0)
            self._publish_empty_trajectory()
            return
        
        age = (self.get_clock().now() - self.corridor['last_update']).nanoseconds / 1e9
        if age > costmap_timeout:
            self.get_logger().warn(f'Costmap stale ({age:.1f}s old) - publishing stop trajectory', throttle_duration_sec=1.0)
            self._publish_empty_trajectory()
            return
        
        # 1. Update motion planner with current vehicle state (already a VehicleState)
        self.motion_planner.update_vehicle_state(self.vehicle_state, dt=0.1)
        
        # 2. Update behavior planner with Frenet state from motion planner
        s, d, d_dot = self.motion_planner.get_frenet_state()
        self.behavior_planner.update_frenet_state(d, d_dot)
        
        # 3. Calculate lateral constraints from corridor
        # Corridor is ego-centric, so use small lookahead in grid coordinates (5m ahead)
        lookahead_m = 5.0
        lookahead_s = min(lookahead_m / self.corridor_generator.resolution, 
                         self.corridor['s_max'] - 0.1)
        d_min, d_max = self.corridor_generator.get_constraints(
            s=lookahead_s,
            x_spline=self.corridor['x_spline'],
            y_spline=self.corridor['y_spline'],
            grid=self.corridor['grid']
        )
        # Convert from grid cells to meters
        d_min_m = d_min * self.corridor_generator.resolution
        d_max_m = d_max * self.corridor_generator.resolution
        self.motion_planner.set_lateral_constraints(d_min_m, d_max_m)
        
        # 4. Get lane availability from CARLA map (only place we use CARLA)
        lane_availability = LaneAvailability(
            left=self._can_change_lane('left'),
            right=self._can_change_lane('right')
        )
        
        # 5. Update behavior state machine
        prev_state = self.behavior_planner.behavior_state
        behavior_output = self.behavior_planner.update(
            obstacles=self.current_obstacles,
            lane_availability=lane_availability,
            speed=self.vehicle_state.speed,
            dt=0.1
        )
        
        # Log state transitions for debugging
        if behavior_output.state != prev_state:
            debug = self.behavior_planner.get_debug_info()
            self.get_logger().info(
                f'Behavior: {prev_state.name} -> {behavior_output.state.name} | '
                f'lane={debug["current_lane"]} -> {debug["target_lane"]} '
                f'd={debug["current_d"]:.2f} cooldown={debug["cooldown"]:.1f}s '
                f'obstacles={len(self.current_obstacles)}'
            )
        
        # Advance global planner for status tracking
        self.global_planner.advance(self.vehicle_location_carla)
        
        # Publish current waypoint for visualization
        self._publish_current_waypoint()
        
        # 6. Generate trajectory (motion planner handles stop trajectory too)
        target_speed_ms = self.target_speed / 3.6
        if behavior_output.state in [BehaviorState.LANE_CHANGING_LEFT, BehaviorState.LANE_CHANGING_RIGHT]:
            target_speed_ms *= self.lane_change_speed_factor
            
        trajectory = self.motion_planner.generate_trajectory(
            behavior=behavior_output,
            speed=self.vehicle_state.speed,
            target_speed=target_speed_ms
        )
        
        if not trajectory:
            self.get_logger().warn('No trajectory generated')
            self._publish_empty_trajectory()
            return
        
        # 7. Publish trajectory
        self._publish_trajectory(trajectory)
        self._publish_corridor_endpoints(trajectory)
        self._publish_motion_goal(trajectory)
    
    def _can_change_lane(self, direction):
        """
        Check if lane change is possible using CARLA map.
        
        This is the ONLY place we query CARLA in the planning loop.
        The result is passed to the MotionPlanner as data.
        """
        current_wp = self.carla_map.get_waypoint(self.vehicle_location_carla)
        if direction == 'left':
            target_wp = current_wp.get_left_lane()
        else:
            target_wp = current_wp.get_right_lane()
        return target_wp is not None and target_wp.lane_type == carla.LaneType.Driving
    
    # ==================== Publishers ====================
    
    def _publish_trajectory(self, trajectory):
        """Publish trajectory as Trajectory message."""
        # Create Trajectory message for control
        traj_msg = Trajectory()
        traj_msg.header.frame_id = 'map'
        traj_msg.header.stamp = self.get_clock().now().to_msg()
        
        # Set behavior state
        behavior_state = self.behavior_planner.behavior_state
        if behavior_state == BehaviorState.LANE_FOLLOWING:
            traj_msg.behavior_state = Trajectory.LANE_FOLLOWING
        elif behavior_state == BehaviorState.PREPARING_LANE_CHANGE:
            traj_msg.behavior_state = Trajectory.PREPARING_LANE_CHANGE
        elif behavior_state in (BehaviorState.LANE_CHANGING_LEFT, BehaviorState.LANE_CHANGING_RIGHT):
            traj_msg.behavior_state = Trajectory.LANE_CHANGING
        elif behavior_state == BehaviorState.EMERGENCY_STOP:
            traj_msg.behavior_state = Trajectory.EMERGENCY_STOP
        else:
            traj_msg.behavior_state = Trajectory.DESTINATION_REACHED
        
        traj_msg.target_lane = self.behavior_planner.target_lane
        
        prev_x, prev_y = None, None
        time_from_start = 0.0
        
        for i, point in enumerate(trajectory):
            # Calculate time from start based on distance and speed
            if prev_x is not None and point.speed > 0.1:
                dist = math.sqrt((point.x - prev_x)**2 + (point.y - prev_y)**2)
                time_from_start += dist / point.speed
            
            # Calculate curvature from path tangent
            curvature = 0.0
            if i > 0 and i < len(trajectory) - 1:
                p0 = trajectory[i-1]
                p1 = trajectory[i]
                p2 = trajectory[i+1]
                dx1 = p1.x - p0.x
                dy1 = p1.y - p0.y
                dx2 = p2.x - p1.x
                dy2 = p2.y - p1.y
                cross = dx1 * dy2 - dy1 * dx2
                dist1 = math.sqrt(dx1**2 + dy1**2)
                dist2 = math.sqrt(dx2**2 + dy2**2)
                if dist1 > 0 and dist2 > 0:
                    curvature = 2 * cross / (dist1 * dist2 * (dist1 + dist2))
            
            # Create TrajectoryPoint
            traj_point = TrajectoryPoint()
            traj_point.x = float(point.x)
            traj_point.y = float(point.y)
            traj_point.yaw = float(point.yaw)
            traj_point.speed = float(point.speed)
            traj_point.curvature = float(curvature)
            traj_point.time_from_start = float(time_from_start)
            traj_msg.points.append(traj_point)
            
            prev_x, prev_y = point.x, point.y
        
        self.trajectory_pub.publish(traj_msg)
    
    def _publish_empty_trajectory(self):
        """Publish empty trajectory (stop signal)."""
        traj_msg = Trajectory()
        traj_msg.header.frame_id = 'map'
        traj_msg.header.stamp = self.get_clock().now().to_msg()
        traj_msg.behavior_state = Trajectory.EMERGENCY_STOP
        self.trajectory_pub.publish(traj_msg)
    
    def _publish_global_route(self):
        """Publish full global route for visualization."""
        route = self.global_planner.get_route()
        if not route:
            return
        
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        for wp, _ in route:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = wp.transform.location.x
            pose.pose.position.y = -wp.transform.location.y
            pose.pose.position.z = wp.transform.location.z
            path_msg.poses.append(pose)
        
        self.global_route_pub.publish(path_msg)
    
    def _publish_current_waypoint(self):
        """Publish current waypoint position for visualization."""
        wp, _ = self.global_planner.get_next_waypoint()
        if wp is None:
            return
        
        msg = PointStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        # Convert from CARLA (left-handed) to ROS (right-handed)
        msg.point.x = wp.transform.location.x
        msg.point.y = -wp.transform.location.y  # Y-axis inversion
        msg.point.z = wp.transform.location.z
        
        self.current_waypoint_pub.publish(msg)
    
    def _get_lookahead_goal_in_grid(self, grid_info) -> tuple:
        """
        Get a lookahead waypoint from global route, transformed to grid coordinates.
        
        This guides the corridor generator to follow the navigation intent
        (e.g., take highway exit instead of continuing straight).
        
        Returns:
            (grid_x, grid_y) tuple, or None if no valid waypoint
        """
        # Get upcoming waypoints from current position (global planner tracks this)
        upcoming = self.global_planner.get_upcoming_waypoints(count=30)
        if not upcoming or len(upcoming) < 2:
            return None
        
        # Find waypoint ~20m ahead from current position
        lookahead_dist = 20.0
        accumulated_dist = 0.0
        target_wp = None
        
        for i in range(1, len(upcoming)):
            wp = upcoming[i][0]
            prev_wp = upcoming[i-1][0]
            dx = wp.transform.location.x - prev_wp.transform.location.x
            dy = wp.transform.location.y - prev_wp.transform.location.y
            accumulated_dist += math.sqrt(dx*dx + dy*dy)
            if accumulated_dist >= lookahead_dist:
                target_wp = wp
                break
        
        if target_wp is None and upcoming:
            target_wp = upcoming[-1][0]  # Use final waypoint if route is short
        
        if target_wp is None:
            return None
        
        # Transform waypoint from map frame to ego_vehicle frame, then to grid
        # Map frame (ROS): wp.x, -wp.y (CARLA Y-flip)
        wp_map_x = target_wp.transform.location.x
        wp_map_y = -target_wp.transform.location.y
        
        # Map -> ego_vehicle transform (inverse of ego->map)
        cos_yaw = math.cos(self.vehicle_state.yaw)
        sin_yaw = math.sin(self.vehicle_state.yaw)
        dx = wp_map_x - self.vehicle_state.x
        dy = wp_map_y - self.vehicle_state.y
        
        # Rotate by -yaw to get ego frame
        ego_x = dx * cos_yaw + dy * sin_yaw
        ego_y = -dx * sin_yaw + dy * cos_yaw
        
        # Ego -> grid coordinates
        grid_x = (ego_x - grid_info.origin.position.x) / grid_info.resolution
        grid_y = (ego_y - grid_info.origin.position.y) / grid_info.resolution
        
        return (grid_x, grid_y)

    def grid_callback(self, msg: OccupancyGrid):
        """
        Process new occupancy grid to update reference path using corridor generator.
        
        Uses global route waypoints to guide corridor generation at junctions.
        """
        # 1. Update generator resolution
        self.corridor_generator.resolution = msg.info.resolution
        
        # 2. Convert grid to binary obstacle map
        width = msg.info.width
        height = msg.info.height
        grid_data = np.array(msg.data, dtype=np.int8).reshape(height, width)
        
        # Binary mask: 0=free, 1=obstacle
        grid_binary = np.zeros_like(grid_data, dtype=np.uint8)
        grid_binary[grid_data != 0] = 1
        
        # 3. Get goal point from global route (for navigation guidance at junctions)
        goal_grid = self._get_lookahead_goal_in_grid(msg.info)
        
        # Calculate vehicle position in grid coordinates (0,0 in ego frame)
        # origin is the position of cell (0,0) in ego frame
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y
        resolution = msg.info.resolution
        
        vehicle_grid_x = (0.0 - origin_x) / resolution
        vehicle_grid_y = (0.0 - origin_y) / resolution
        
        # 4. Generate reference line from corridor with goal guidance
        try:
            x_spline, y_spline, s_max = self.corridor_generator.generate_reference_line(
                grid_binary, 
                start_point=(vehicle_grid_x, vehicle_grid_y),
                goal_point=goal_grid
            )
        except ValueError as e:
            self.get_logger().error(f'Corridor generation failed: {e}')
            self.corridor['valid'] = False
            return
        
        # 5. Store splines and grid data
        self.corridor['x_spline'] = x_spline
        self.corridor['y_spline'] = y_spline
        self.corridor['s_max'] = s_max
        self.corridor['grid'] = grid_binary
        self.corridor['origin_x'] = msg.info.origin.position.x
        self.corridor['origin_y'] = msg.info.origin.position.y
        self.corridor['last_update'] = self.get_clock().now()
        
        # 6. Sample ego-frame waypoints from spline and transform to map frame
        path_grid_points = self.corridor_generator.get_reference_path_points(
            x_spline, y_spline, s_max, num_points=100
        )
        
        # Transform from ego_vehicle frame to map frame
        cos_yaw = math.cos(self.vehicle_state.yaw)
        sin_yaw = math.sin(self.vehicle_state.yaw)
        vx, vy = self.vehicle_state.x, self.vehicle_state.y
        
        path_x_map = []
        path_y_map = []
        
        for px, py in path_grid_points:
            # Grid -> ego_vehicle frame
            ex, ey = self.corridor_generator.to_world_coordinates(
                px, py, self.corridor['origin_x'], self.corridor['origin_y']
            )
            # ego_vehicle -> map frame
            mx = vx + ex * cos_yaw - ey * sin_yaw
            my = vy + ex * sin_yaw + ey * cos_yaw
            path_x_map.append(mx)
            path_y_map.append(my)
        
        # 7. Update motion planner with new reference path (map frame)
        self.motion_planner.set_reference_path(path_x_map, path_y_map)
        self.corridor['valid'] = True
        
        # 8. Publish corridor reference line for visualization (map frame)
        self._publish_corridor_reference(path_x_map, path_y_map, msg.header.stamp)
    
    def _publish_corridor_reference(self, path_x: list, path_y: list, stamp):
        """Publish the corridor reference line as a Path message for RViz."""
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = stamp
        
        for x, y in zip(path_x, path_y):
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.1  # Slight elevation for visibility
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        
        self.corridor_pub.publish(path_msg)

    def _publish_corridor_endpoints(self, trajectory: list):
        """Publish start and end points of the generated corridor trajectory as Markers."""
        if not trajectory:
            return

        start_point = trajectory[0]
        end_point = trajectory[-1]
        stamp = self.get_clock().now().to_msg()

        # Start point marker (GREEN sphere)
        start_msg = Marker()
        start_msg.header.frame_id = 'map'
        start_msg.header.stamp = stamp
        start_msg.ns = 'corridor_endpoints'
        start_msg.id = 0
        start_msg.type = Marker.SPHERE
        start_msg.action = Marker.ADD
        start_msg.pose.position.x = float(start_point.x)
        start_msg.pose.position.y = float(start_point.y)
        start_msg.pose.position.z = 0.5
        start_msg.pose.orientation.w = 1.0
        start_msg.scale.x = 1.0
        start_msg.scale.y = 1.0
        start_msg.scale.z = 1.0
        start_msg.color.r = 0.0
        start_msg.color.g = 1.0
        start_msg.color.b = 0.0
        start_msg.color.a = 1.0
        self.corridor_start_pub.publish(start_msg)

        # End point marker (RED sphere)
        end_msg = Marker()
        end_msg.header.frame_id = 'map'
        end_msg.header.stamp = stamp
        end_msg.ns = 'corridor_endpoints'
        end_msg.id = 1
        end_msg.type = Marker.SPHERE
        end_msg.action = Marker.ADD
        end_msg.pose.position.x = float(end_point.x)
        end_msg.pose.position.y = float(end_point.y)
        end_msg.pose.position.z = 0.5
        end_msg.pose.orientation.w = 1.0
        end_msg.scale.x = 1.0
        end_msg.scale.y = 1.0
        end_msg.scale.z = 1.0
        end_msg.color.r = 1.0
        end_msg.color.g = 0.0
        end_msg.color.b = 0.0
        end_msg.color.a = 1.0
        self.corridor_end_pub.publish(end_msg)

    def _publish_motion_goal(self, trajectory: list):
        """Publish the goal point the motion planner is trying to achieve as a Marker."""
        if not trajectory:
            return

        goal_point = trajectory[-1]
        
        # Motion goal marker (CYAN cube)
        goal_msg = Marker()
        goal_msg.header.frame_id = 'map'
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.ns = 'motion_goal'
        goal_msg.id = 0
        goal_msg.type = Marker.CUBE
        goal_msg.action = Marker.ADD
        goal_msg.pose.position.x = float(goal_point.x)
        goal_msg.pose.position.y = float(goal_point.y)
        goal_msg.pose.position.z = 0.5
        goal_msg.pose.orientation.w = 1.0
        goal_msg.scale.x = 1.5
        goal_msg.scale.y = 1.5
        goal_msg.scale.z = 1.5
        goal_msg.color.r = 0.0
        goal_msg.color.g = 1.0
        goal_msg.color.b = 1.0
        goal_msg.color.a = 0.8
        self.motion_goal_pub.publish(goal_msg)

    def _publish_status(self):
        """Publish planning status."""
        s, d, d_dot = self.motion_planner.get_frenet_state()
        behavior_debug = self.behavior_planner.get_debug_info()
        
        status = {
            'initialized': self.initialized,
            'is_navigating': self.is_navigating,
            'destination_reached': self.destination_reached,
            'behavior_state': behavior_debug['state'],
            'current_lane': behavior_debug['current_lane'],
            'target_lane': behavior_debug['target_lane'],
            'frenet_s': round(s, 2),
            'frenet_d': round(d, 2),
            'target_d': behavior_debug['target_d'],
            'cooldown': behavior_debug['cooldown'],
            'progress': self.global_planner.get_progress() if self.global_planner else 0.0,
            'remaining_distance': self.global_planner.get_remaining_distance() if self.global_planner else 0.0,
            'target_speed': self.target_speed,
            'obstacle_count': len(self.current_obstacles)
        }
        
        status_msg = String()
        status_msg.data = str(status)
        self.status_pub.publish(status_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PlanningNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


# if __name__ == '__main__':
#     main()
