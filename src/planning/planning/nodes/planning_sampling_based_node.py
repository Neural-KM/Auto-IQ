#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
import numpy as np
import math

from nav_msgs.msg import Path, Odometry, OccupancyGrid
from geometry_msgs.msg import PoseStamped, PointStamped
from std_msgs.msg import String, Int32
from autoiq_msgs.msg import Trajectory, TrajectoryPoint
import tf2_ros
from tf2_ros import TransformException

from autoiq_common.types import VehicleState
from autoiq_common.converters import vehicle_state_from_odometry
from mapping.utils.geometry import inflate_to_cspace
import carla

# Import your planners
from planning.core.global_planner import GlobalPlanner
from planning.core.planner_unified import SamplingBasedPlanner, PlannerConfig

class PlanningNode(Node):
    def __init__(self):
        super().__init__('planning_node')
        
        # --- Parameters ---
        self.declare_parameter('carla_host', 'localhost')
        self.declare_parameter('target_speed', 10.0) # km/h
        self.declare_parameter('vehicle_radius', 1.5) # meters (for C-space inflation)
        self.declare_parameter('costmap_topic', '/mapping/ipm_costmap')
        carla_host = self.get_parameter('carla_host').value
        self.target_speed = self.get_parameter('target_speed').value / 3.6 # m/s
        self.vehicle_radius = self.get_parameter('vehicle_radius').value
        
        # --- CARLA Connection (Only for Global Route) ---
        self.get_logger().info('Connecting to CARLA...')
        client = carla.Client(carla_host, 2000)
        client.set_timeout(4.0)
        self.carla_map = client.get_world().get_map()
        
        # --- Initialize Planners ---
        self.global_planner = GlobalPlanner(self.carla_map, sampling_resolution=2.0)
        
        self.motion_planner = SamplingBasedPlanner()
        
        # --- State Variables ---
        self.vehicle_state = VehicleState()
        self.latest_costmap = None # Holds numpy array of grid
        self.costmap_origin_x = 0.0  # Position of cell [0,0] in costmap frame
        self.costmap_origin_y = 0.0
        self.costmap_resolution = 0.4
        self.costmap_frame_id = None  # Set from OccupancyGrid message
        self.is_active = False
        
        # --- TF2 for frame transforms ---
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=1.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # --- ROS2 Subscribers ---
        self.create_subscription(Odometry, '/carla/ego_vehicle/ground_truth/odometry', self.odom_cb, 1)
        self.create_subscription(PoseStamped, '/planning/goal_pose', self.goal_cb, 1)
        self.goal_index_sub = self.create_subscription(
            Int32, '/planning/goal_index',
            self.goal_index_callback, 10)
        self.create_subscription(OccupancyGrid, self.get_parameter('costmap_topic').value, self.costmap_cb, 1)
        
        # --- ROS2 Publishers ---
        self.traj_pub = self.create_publisher(Trajectory, '/planning/trajectory', 1)
        self.global_route_pub = self.create_publisher(Path, '/planning/global_route', 1)
        self.cspace_pub = self.create_publisher(OccupancyGrid, '/planning/cspace_costmap', 1)
        
        # --- Timer Loop (10 Hz) ---
        self.create_timer(0.1, self.planning_loop)
        self.get_logger().info('Discrete sampling Planner Ready.')

    def odom_cb(self, msg):
        self.vehicle_state = vehicle_state_from_odometry(msg)

    def costmap_cb(self, msg):
        # Store grid data and metadata for the planning loop
        grid = np.array(msg.data, dtype=np.int8)
        grid = grid.reshape((msg.info.height, msg.info.width))
        
        # Inflate obstacles to create C-space (configuration space)
        resolution = msg.info.resolution
        grid = inflate_to_cspace(grid, self.vehicle_radius, resolution)
        
        self.latest_costmap = grid
        self.costmap_origin_x = msg.info.origin.position.x
        self.costmap_origin_y = msg.info.origin.position.y
        self.costmap_resolution = resolution
        self.costmap_frame_id = msg.header.frame_id
        
        # Publish C-space for debugging
        cspace_msg = OccupancyGrid()
        cspace_msg.header = msg.header
        cspace_msg.info = msg.info
        cspace_msg.data = grid.flatten().tolist()
        self.cspace_pub.publish(cspace_msg)

    def goal_cb(self, msg):
        # Convert ROS Goal -> CARLA Location
        goal_loc = carla.Location(x=msg.pose.position.x, y=-msg.pose.position.y, z=msg.pose.position.z)
        self._set_destination(goal_loc)
            
    def goal_index_callback(self, msg: Int32):
        """
        Set navigation goal using CARLA spawn point index.
        
        Spawn points are already in CARLA left-handed coordinates.
        """
        spawn_index = msg.data
        spawn_points = self.carla_map.get_spawn_points()
        
        if spawn_index < 0 or spawn_index >= len(spawn_points):
            self.get_logger().error(f'Invalid spawn index {spawn_index}. Valid range: 0-{len(spawn_points)-1}')
            return
        
        goal_location_carla = spawn_points[spawn_index].location
        self._set_destination(goal_location_carla)

    def _set_destination(self, goal_location_carla: carla.Location):
        # Get Vehicle Location
        start_loc = carla.Location(x=self.vehicle_state.x, y=-self.vehicle_state.y, z=self.vehicle_state.z)
        
        # plan global route
        if self.global_planner.set_destination(start_loc, goal_location_carla):
            self.is_active = True
            
            # Pass Reference Path to Motion Planner
            route = self.global_planner.get_route()
            rx = [wp.transform.location.x for wp, _ in route]
            ry = [-wp.transform.location.y for wp, _ in route]
            self.motion_planner.set_reference_path(rx, ry)
            
            self._publish_global_route()
            self.get_logger().info('New Global Route Set.')

    def planning_loop(self):
        if not self.is_active or self.latest_costmap is None:
            return

        # 1. Update Global Planner progress
        current_loc = carla.Location(x=self.vehicle_state.x, y=-self.vehicle_state.y, z=self.vehicle_state.z)
        self.global_planner.advance(current_loc)
        
        # 2. Check Reached
        if self.global_planner.is_destination_reached(current_loc):
            self.is_active = False
            self.get_logger().info("Destination Reached.")
            return
        
        # 3. Get transform from map frame to costmap frame
        map_to_costmap = self._get_map_to_costmap_transform()
        if map_to_costmap is None:
            return
        
        t_sampling_start = time.perf_counter()
        # 4. Run Sampling based Planner
        trajectory = self.motion_planner.plan(
            self.vehicle_state,
            self.target_speed, 
            self.latest_costmap,
            self.costmap_origin_x,
            self.costmap_origin_y,
            self.costmap_resolution,
            map_to_costmap
        )
        self.get_logger().info(f'Planning Time: {(time.perf_counter() - t_sampling_start)*1000:.1f} ms')
        # 5. Publish
        if trajectory:
            self._publish_trajectory(trajectory)
        else:
            self.get_logger().warn("No valid path found!")
    
    def _get_map_to_costmap_transform(self):
        """
        Get transform from map frame to costmap frame.
        
        Returns:
            Tuple of (tx, ty, yaw) or None if lookup fails
        """
        try:
            transform = self.tf_buffer.lookup_transform(
                'map',
                self.costmap_frame_id,  # Target frame
                                   # Source frame
                rclpy.time.Time()        # Latest available
            )
            
            tx = transform.transform.translation.x
            ty = transform.transform.translation.y
            
            # Extract yaw from quaternion
            q = transform.transform.rotation
            # yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            
            return (tx, ty, yaw)
            
        except TransformException as e:
            self.get_logger().warn(f'TF lookup map->{self.costmap_frame_id} failed: {e}')
            return None

    def _publish_trajectory(self, points):
        msg = Trajectory()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.behavior_state = Trajectory.LANE_FOLLOWING # Generic state
        
        prev_pt = None
        for pt in points:
            t_pt = TrajectoryPoint()
            t_pt.x = float(pt.x)
            t_pt.y = float(pt.y)
            t_pt.yaw = float(pt.yaw)
            t_pt.speed = float(pt.speed)
            
            # Simple time calculation
            if prev_pt:
                dist = math.hypot(pt.x - prev_pt.x, pt.y - prev_pt.y)
                dt = dist / max(pt.speed, 0.1)
                t_pt.time_from_start = prev_pt.time_from_start + dt
            else:
                t_pt.time_from_start = 0.0
                
            msg.points.append(t_pt)
            prev_pt = t_pt
            
        self.traj_pub.publish(msg)

    def _publish_global_route(self):
        # (Same as before - strictly for visualization)
        route = self.global_planner.get_route()
        if not route: return
        msg = Path()
        msg.header.frame_id = 'map'
        for wp, _ in route:
            p = PoseStamped()
            p.pose.position.x = wp.transform.location.x
            p.pose.position.y = -wp.transform.location.y
            msg.poses.append(p)
        self.global_route_pub.publish(msg)

def main():
    rclpy.init()
    node = PlanningNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()