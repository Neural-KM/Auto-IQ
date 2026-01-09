#!/usr/bin/env python3
"""
Planning Node
=============

ROS2 node for path planning (Global + Local + Behavior).
Connects directly to CARLA for map access, receives vehicle state via ROS2 topics.

Subscribes:
- /carla/ego_vehicle/ground_truth/odometry (nav_msgs/Odometry) - Vehicle state
- /planning/goal_pose (geometry_msgs/PoseStamped) - Navigation goal
- /perception/obstacles (visualization_msgs/MarkerArray) - Detected obstacles

Publishes:
- /planning/trajectory (nav_msgs/Path) - Planned trajectory for control
- /planning/global_route (nav_msgs/Path) - Full global route for visualization
- /planning/status (std_msgs/String) - Planning status
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import String, Int32
from visualization_msgs.msg import MarkerArray

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
    VehicleState,
)


def quaternion_to_yaw(q):
    """Convert quaternion to yaw angle in radians."""
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


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
        self.declare_parameter('target_speed', 30.0)  # km/h
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
        
        # Initialize Behavior Planner (state machine)
        behavior_config = BehaviorPlannerConfig(
            lane_width=3.5,
            lane_change_time=self.lane_change_time,
            base_trigger_distance=self.lane_change_trigger_distance,
            safety_buffer=5.0,
            prepare_timeout=2.0,  # Timeout for PREPARING state
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
        
        # ==================== State ====================
        self.initialized = True
        self.is_navigating = False
        self.destination_reached = False
        self._destination = None  # Current destination for replanning
        
        # Vehicle state (from odometry) - stored as ROS coordinates
        self.vehicle_x = 0.0
        self.vehicle_y = 0.0
        self.vehicle_yaw = 0.0
        self.vehicle_speed = 0.0
        
        # CARLA-format location (for map queries only)
        self.vehicle_location = None
        
        # Obstacle detection (from perception) - list of all obstacles
        self.current_obstacles: List[ObstacleInfo] = []
        
        # ==================== Subscribers ====================
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
            MarkerArray, obstacle_topic,
            self.obstacles_callback, 10)
        
        # ==================== Publishers ====================
        self.trajectory_pub = self.create_publisher(Path, '/planning/trajectory', 10)
        self.global_route_pub = self.create_publisher(Path, '/planning/global_route', 10)
        self.status_pub = self.create_publisher(String, '/planning/status', 10)
        
        # ==================== Timer ====================
        self.create_timer(0.1, self.planning_loop)  # 10 Hz
        
        self.get_logger().info('Planning Node initialized successfully!')
    
    # ==================== Callbacks ====================

    def odom_callback(self, msg: Odometry):
        """Update vehicle state from odometry."""
        pos = msg.pose.pose.position
        quat = msg.pose.pose.orientation
        vel = msg.twist.twist.linear
        
        # Store ROS coordinates
        self.vehicle_x = pos.x
        self.vehicle_y = pos.y
        self.vehicle_yaw = quaternion_to_yaw(quat)
        self.vehicle_speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        
        # Also store CARLA-format location for map queries
        self.vehicle_location = carla.Location(
            x=pos.x,
            y=-pos.y,  # Y-axis inversion for CARLA
            z=pos.z
        )
    
    def goal_callback(self, msg: PoseStamped):
        """Set navigation goal from PoseStamped."""
        if self.vehicle_location is None:
            self.get_logger().warn('Cannot set goal - no vehicle position')
            return
        
        # Convert goal to CARLA location
        goal_location = carla.Location(
            x=msg.pose.position.x,
            y=-msg.pose.position.y,  # Y-axis inversion
            z=msg.pose.position.z
        )
        
        self._set_destination(goal_location, 'goal pose')
    
    def goal_index_callback(self, msg: Int32):
        """Set navigation goal using spawn point index."""
        if self.vehicle_location is None:
            self.get_logger().warn('Cannot set goal - no vehicle position')
            return
        
        spawn_index = msg.data
        spawn_points = self.carla_map.get_spawn_points()
        
        if spawn_index < 0 or spawn_index >= len(spawn_points):
            self.get_logger().error(f'Invalid spawn index {spawn_index}. Valid range: 0-{len(spawn_points)-1}')
            return
        
        goal_location = spawn_points[spawn_index].location
        self._set_destination(goal_location, f'spawn point {spawn_index}')
    
    def _set_destination(self, goal_location: carla.Location, description: str):
        """Set navigation destination and compute route."""
        self._destination = goal_location  # Store for replanning
        
        success = self.global_planner.set_destination(
            self.vehicle_location,
            goal_location
        )
        
        if success:
            self.is_navigating = True
            self.destination_reached = False
            
            # Initialize Motion Planner with reference path
            route = self.global_planner.get_route()
            if route:
                # Convert to ROS coordinates for the motion planner
                rx = [wp.transform.location.x for wp, _ in route]
                ry = [-wp.transform.location.y for wp, _ in route]
                self.motion_planner.set_reference_path(rx, ry)
                self.get_logger().info('Motion planner initialized with reference path')

            self.get_logger().info(f'Route computed to {description}')
            self._publish_global_route()
        else:
            self.get_logger().error(f'Failed to compute route to {description}')
    

    def obstacles_callback(self, msg: MarkerArray):
        """Process detected obstacles from perception."""
        self.current_obstacles = []
        
        if not msg.markers:
            return
        
        for marker in msg.markers:
            x = marker.pose.position.x
            y = marker.pose.position.y
            z = marker.pose.position.z
            
            dist = math.sqrt(x**2 + y**2 + z**2)
            
            # Store all obstacles (motion planner will filter by relevance)
            self.current_obstacles.append(
                ObstacleInfo(x=x, y=y, distance=dist)
            )
    
    # ==================== Planning Loop ====================
    
    def planning_loop(self):
        """Main planning loop - delegates to MotionPlanner."""
        if self.vehicle_location is None:
            return
        
        # Publish status
        self._publish_status()
        
        if not self.is_navigating:
            self._publish_empty_trajectory()
            return

        # Check if destination reached
        if self.global_planner.is_destination_reached(self.vehicle_location, threshold=5.0):
            self.get_logger().info('Destination reached!')
            self.is_navigating = False
            self.destination_reached = True
            self._publish_empty_trajectory()
            return
        
        # 1. Update motion planner with current vehicle state (ROS coords)
        vehicle_state = VehicleState(
            x=self.vehicle_x,
            y=self.vehicle_y,
            yaw=self.vehicle_yaw,
            speed=self.vehicle_speed
        )
        self.motion_planner.update_vehicle_state(vehicle_state, dt=0.1)
        
        # 2. Update behavior planner with Frenet state from motion planner
        s, d, d_dot = self.motion_planner.get_frenet_state()
        self.behavior_planner.update_frenet_state(d, d_dot)
        
        # 3. Get lane availability from CARLA map (only place we use CARLA)
        lane_availability = LaneAvailability(
            left=self._can_change_lane('left'),
            right=self._can_change_lane('right')
        )
        
        # 4. Update behavior state machine
        prev_state = self.behavior_planner.behavior_state
        behavior_output = self.behavior_planner.update(
            obstacles=self.current_obstacles,
            lane_availability=lane_availability,
            speed=self.vehicle_speed,
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
        self.global_planner.advance(self.vehicle_location)
        
        # 5. Generate trajectory (motion planner handles stop trajectory too)
        target_speed_ms = self.target_speed / 3.6
        if behavior_output.state in [BehaviorState.LANE_CHANGING_LEFT, BehaviorState.LANE_CHANGING_RIGHT]:
            target_speed_ms *= self.lane_change_speed_factor
            
        trajectory = self.motion_planner.generate_trajectory(
            behavior=behavior_output,
            speed=self.vehicle_speed,
            target_speed=target_speed_ms
        )
        
        if not trajectory:
            self.get_logger().warn('No trajectory generated')
            self._publish_empty_trajectory()
            return
        
        # 5. Publish trajectory
        self._publish_trajectory(trajectory)
    
    def _can_change_lane(self, direction):
        """
        Check if lane change is possible using CARLA map.
        
        This is the ONLY place we query CARLA in the planning loop.
        The result is passed to the MotionPlanner as data.
        """
        current_wp = self.carla_map.get_waypoint(self.vehicle_location)
        if direction == 'left':
            target_wp = current_wp.get_left_lane()
        else:
            target_wp = current_wp.get_right_lane()
        return target_wp is not None and target_wp.lane_type == carla.LaneType.Driving
    
    # ==================== Publishers ====================
    
    def _publish_trajectory(self, trajectory):
        """Publish trajectory from MotionPlanner output."""
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        points = []
        for point in trajectory:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = point.x
            pose.pose.position.y = point.y
            pose.pose.position.z = 0.0
            
            # Calculate yaw from path tangent for smoother orientation
            if len(points) > 0:
                prev_x, prev_y = points[-1]
                path_yaw = math.atan2(point.y - prev_y, point.x - prev_x)
            else:
                path_yaw = point.yaw
                
            pose.pose.orientation.z = math.sin(path_yaw / 2)
            pose.pose.orientation.w = math.cos(path_yaw / 2)
            
            points.append((point.x, point.y))
            path_msg.poses.append(pose)
        
        self.trajectory_pub.publish(path_msg)
    
    def _publish_empty_trajectory(self):
        """Publish empty trajectory (stop signal)."""
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()
        self.trajectory_pub.publish(path_msg)
    
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


if __name__ == '__main__':
    main()
