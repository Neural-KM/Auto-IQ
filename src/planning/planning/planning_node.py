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

import carla

from planning.global_planner import GlobalPlanner, RoadOption
from planning.behavior_planner import BehaviorState


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
        self.declare_parameter('lane_change_trigger_distance', 20.0)
        self.declare_parameter('lane_change_speed_factor', 0.7)
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
        
        # ==================== State ====================
        self.initialized = True
        self.is_navigating = False
        self.destination_reached = False
        
        # Vehicle state (from odometry)
        self.vehicle_location = None
        self.vehicle_transform = None
        self.vehicle_speed = 0.0
        
        # Behavior state
        self.behavior_state = BehaviorState.LANE_FOLLOWING
        self.lane_change_waypoints = []
        self.lane_change_index = 0
        
        # Obstacle detection (from perception)
        self.current_obstacle = None
        
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
        
        # Convert to CARLA coordinates (ROS uses different convention)
        self.vehicle_location = carla.Location(
            x=pos.x,
            y=-pos.y,  # Y-axis inversion for CARLA
            z=pos.z
        )
        
        yaw_rad = quaternion_to_yaw(quat)
        self.vehicle_transform = carla.Transform(
            self.vehicle_location,
            carla.Rotation(
                pitch=0.0,
                yaw=math.degrees(-yaw_rad),  # Yaw inversion for CARLA
                roll=0.0
            )
        )
        
        # Speed in m/s
        self.vehicle_speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
    
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
        success = self.global_planner.set_destination(
            self.vehicle_location,
            goal_location
        )
        
        if success:
            self.is_navigating = True
            self.destination_reached = False
            self.behavior_state = BehaviorState.LANE_FOLLOWING
            self.lane_change_waypoints = []
            self.lane_change_index = 0
            self.get_logger().info(f'Route computed to {description}')
            self._publish_global_route()
        else:
            self.get_logger().error(f'Failed to compute route to {description}')
    
    def obstacles_callback(self, msg: MarkerArray):
        """Process detected obstacles from perception."""
        if not msg.markers:
            self.current_obstacle = None
            return
        
        # Find closest obstacle in front
        closest_obstacle = None
        closest_dist = float('inf')
        
        for marker in msg.markers:
            # Obstacle position in ego frame (from marker position)
            x = marker.pose.position.x
            y = marker.pose.position.y
            z = marker.pose.position.z
            
            # Only consider obstacles ahead
            if x > 0:
                dist = math.sqrt(x**2 + y**2 + z**2)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_obstacle = type('Obstacle', (), {
                        'x': x, 'y': y, 'z': z,
                        'distance': dist
                    })()
        
        self.current_obstacle = closest_obstacle
    
    # ==================== Planning Loop ====================
    
    def planning_loop(self):
        """Main planning loop."""
        if self.vehicle_location is None:
            return
        
        # Publish status
        self._publish_status()
        
        if not self.is_navigating:
            # Publish empty trajectory when not navigating
            self._publish_empty_trajectory()
            return
        
        # Check if destination reached
        if self.global_planner.is_destination_reached(self.vehicle_location, threshold=5.0):
            self.get_logger().info('Destination reached!')
            self.is_navigating = False
            self.destination_reached = True
            self._publish_empty_trajectory()
            return
        
        # Update behavior based on obstacles
        self._update_behavior()
        
        # Emergency stop check
        if self.behavior_state == BehaviorState.EMERGENCY_STOP:
            self.get_logger().warn('Emergency stop triggered')
            self._publish_empty_trajectory()
            return
        
        # Advance global planner
        self.global_planner.advance(self.vehicle_location)
        
        # Get waypoints based on behavior
        waypoints = self._get_waypoints_for_control()
        
        if not waypoints:
            self.get_logger().warn('No waypoints available')
            self._publish_empty_trajectory()
            return
        
        # Determine effective target speed
        if self.behavior_state in [BehaviorState.LANE_CHANGING_LEFT, 
                                   BehaviorState.LANE_CHANGING_RIGHT]:
            effective_speed = self.target_speed * self.lane_change_speed_factor
        else:
            effective_speed = self.target_speed
        
        # Publish trajectory
        self._publish_trajectory(waypoints, effective_speed)
    
    def _update_behavior(self):
        """Update behavior state based on obstacles."""
        if self.behavior_state == BehaviorState.LANE_FOLLOWING:
            if self.current_obstacle:
                dist = self.current_obstacle.distance
                is_center = self.current_obstacle.x > 0 and abs(self.current_obstacle.y) < 2.0
                
                if is_center and dist < self.lane_change_trigger_distance:
                    self.behavior_state = BehaviorState.PREPARING_LANE_CHANGE
                    self.get_logger().info(f'Obstacle at {dist:.1f}m - preparing lane change')
        
        elif self.behavior_state == BehaviorState.PREPARING_LANE_CHANGE:
            # Try to generate lane change waypoints
            can_left = self._can_change_lane('left')
            can_right = self._can_change_lane('right')
            
            if can_left:
                self.lane_change_waypoints = self._generate_lane_change_waypoints('left')
                if self.lane_change_waypoints:
                    self.lane_change_index = 0
                    self.behavior_state = BehaviorState.LANE_CHANGING_LEFT
                    self.get_logger().info('Executing lane change left')
                else:
                    self.behavior_state = BehaviorState.EMERGENCY_STOP
            elif can_right:
                self.lane_change_waypoints = self._generate_lane_change_waypoints('right')
                if self.lane_change_waypoints:
                    self.lane_change_index = 0
                    self.behavior_state = BehaviorState.LANE_CHANGING_RIGHT
                    self.get_logger().info('Executing lane change right')
                else:
                    self.behavior_state = BehaviorState.EMERGENCY_STOP
            else:
                self.behavior_state = BehaviorState.EMERGENCY_STOP
                self.get_logger().warn('No lane available - emergency stop')
        
        elif self.behavior_state in [BehaviorState.LANE_CHANGING_LEFT, 
                                     BehaviorState.LANE_CHANGING_RIGHT]:
            if self.lane_change_index >= len(self.lane_change_waypoints):
                self.behavior_state = BehaviorState.LANE_FOLLOWING
                self.lane_change_waypoints = []
                self.lane_change_index = 0
                self.get_logger().info('Lane change complete')
            else:
                # Check if we've reached current waypoint
                target_wp = self.lane_change_waypoints[self.lane_change_index]
                dist = self.vehicle_location.distance(target_wp.transform.location)
                if dist < 2.5:
                    self.lane_change_index += 1
    
    def _can_change_lane(self, direction):
        """Check if lane change is possible."""
        current_wp = self.carla_map.get_waypoint(self.vehicle_location)
        if direction == 'left':
            target_wp = current_wp.get_left_lane()
        else:
            target_wp = current_wp.get_right_lane()
        return target_wp is not None and target_wp.lane_type == carla.LaneType.Driving
    
    def _generate_lane_change_waypoints(self, direction):
        """Generate waypoints for lane change."""
        current_wp = self.carla_map.get_waypoint(self.vehicle_location)
        waypoints = []
        
        # Approach phase
        wp = current_wp
        for _ in range(3):
            next_wps = wp.next(self.waypoint_spacing)
            if next_wps:
                waypoints.append(next_wps[0])
                wp = next_wps[0]
        
        # Transition to new lane
        if direction == 'left':
            transition_wp = wp.get_left_lane()
        else:
            transition_wp = wp.get_right_lane()
        
        if transition_wp:
            waypoints.append(transition_wp)
            wp = transition_wp
            for _ in range(25):
                next_wps = wp.next(self.waypoint_spacing)
                if next_wps:
                    waypoints.append(next_wps[0])
                    wp = next_wps[0]
        
        return waypoints
    
    def _get_waypoints_for_control(self):
        """Get waypoints for controller based on current behavior."""
        if self.behavior_state in [BehaviorState.LANE_CHANGING_LEFT, 
                                   BehaviorState.LANE_CHANGING_RIGHT]:
            return self._get_lane_change_waypoints()
        else:
            return self._get_lookahead_waypoints()
    
    def _get_lookahead_waypoints(self):
        """Get waypoints from global route."""
        upcoming = self.global_planner.get_upcoming_waypoints(self.lookahead_waypoints)
        if not upcoming:
            return []
        return [wp for wp, _ in upcoming]
    
    def _get_lane_change_waypoints(self):
        """Get waypoints during lane change."""
        if not self.lane_change_waypoints or self.lane_change_index >= len(self.lane_change_waypoints):
            current_wp = self.carla_map.get_waypoint(self.vehicle_location)
            return self._extend_waypoints_from(current_wp, self.lookahead_waypoints)
        
        remaining = self.lane_change_waypoints[self.lane_change_index:]
        
        if len(remaining) < self.lookahead_waypoints:
            last_wp = self.lane_change_waypoints[-1]
            additional = self._extend_waypoints_from(last_wp, self.lookahead_waypoints - len(remaining))
            return remaining + additional
        
        return remaining[:self.lookahead_waypoints]
    
    def _extend_waypoints_from(self, start_wp, count):
        """Extend waypoints from a starting waypoint."""
        waypoints = []
        wp = start_wp
        for _ in range(count):
            next_wps = wp.next(self.waypoint_spacing)
            if next_wps:
                waypoints.append(next_wps[0])
                wp = next_wps[0]
            else:
                break
        return waypoints
    
    # ==================== Publishers ====================
    
    def _publish_trajectory(self, waypoints, target_speed):
        """Publish trajectory as nav_msgs/Path."""
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        for wp in waypoints:
            pose = PoseStamped()
            pose.header = path_msg.header
            
            # Convert CARLA coordinates to ROS
            pose.pose.position.x = wp.transform.location.x
            pose.pose.position.y = -wp.transform.location.y  # Y-axis inversion
            pose.pose.position.z = wp.transform.location.z
            
            # Convert yaw to quaternion
            yaw = math.radians(-wp.transform.rotation.yaw)  # Yaw inversion
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = math.sin(yaw / 2)
            pose.pose.orientation.w = math.cos(yaw / 2)
            
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
        status = {
            'initialized': self.initialized,
            'is_navigating': self.is_navigating,
            'destination_reached': self.destination_reached,
            'behavior_state': self.behavior_state.name,
            'progress': self.global_planner.get_progress() if self.global_planner else 0.0,
            'remaining_distance': self.global_planner.get_remaining_distance() if self.global_planner else 0.0,
            'target_speed': self.target_speed,
            'obstacle_detected': self.current_obstacle is not None
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
