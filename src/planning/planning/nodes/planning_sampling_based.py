#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import math

from nav_msgs.msg import Path, Odometry, OccupancyGrid
from geometry_msgs.msg import PoseStamped, PointStamped
from std_msgs.msg import String, Int32
from autoiq_msgs.msg import Trajectory, TrajectoryPoint

from autoiq_common.types import VehicleState
from autoiq_common.converters import vehicle_state_from_odometry
import carla

# Import your planners
from planning.core.global_planner import GlobalPlanner
from planning.core.planner_unified import OptimizationMotionPlanner, PlannerConfig

class PlanningNode(Node):
    def __init__(self):
        super().__init__('planning_node')
        
        # --- Parameters ---
        self.declare_parameter('carla_host', 'localhost')
        self.declare_parameter('target_speed', 10.0) # km/h
        
        carla_host = self.get_parameter('carla_host').value
        self.target_speed = self.get_parameter('target_speed').value / 3.6 # m/s
        
        # --- CARLA Connection (Only for Global Route) ---
        self.get_logger().info('Connecting to CARLA...')
        client = carla.Client(carla_host, 2000)
        client.set_timeout(4.0)
        self.carla_map = client.get_world().get_map()
        
        # --- Initialize Planners ---
        self.global_planner = GlobalPlanner(self.carla_map, sampling_resolution=2.0)
        
        self.motion_planner = OptimizationMotionPlanner()
        
        # --- State Variables ---
        self.vehicle_state = VehicleState()
        self.latest_costmap = None # Holds numpy array of grid
        self.costmap_origin_x = 0.0  # World position of cell [0,0]
        self.costmap_origin_y = 0.0
        self.costmap_resolution = 0.2
        self.is_active = False
        
        # --- ROS2 Subscribers ---
        self.create_subscription(Odometry, '/carla/ego_vehicle/ground_truth/odometry', self.odom_cb, 10)
        self.create_subscription(PoseStamped, '/planning/goal_pose', self.goal_cb, 10)
        self.goal_index_sub = self.create_subscription(
            Int32, '/planning/goal_index',
            self.goal_index_callback, 10)
        self.create_subscription(OccupancyGrid, '/mapping/local_costmap', self.costmap_cb, 10)
        
        # --- ROS2 Publishers ---
        self.traj_pub = self.create_publisher(Trajectory, '/planning/trajectory', 10)
        self.global_route_pub = self.create_publisher(Path, '/planning/global_route', 10)
        
        # --- Timer Loop (10 Hz) ---
        self.create_timer(0.1, self.planning_loop)
        self.get_logger().info('Discrete Optimization Planner Ready.')

    def odom_cb(self, msg):
        self.vehicle_state = vehicle_state_from_odometry(msg)

    def costmap_cb(self, msg):
        # Store grid data and metadata for the planning loop
        grid = np.array(msg.data, dtype=np.int8)
        if msg.info.width > 0 and msg.info.height > 0:
            grid = grid.reshape((msg.info.height, msg.info.width))
        self.latest_costmap = grid
        self.costmap_origin_x = msg.info.origin.position.x
        self.costmap_origin_y = msg.info.origin.position.y
        self.costmap_resolution = msg.info.resolution

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

        # 3. Run Optimization Planner
        # This will sample 5 paths, check the grid, and return the best one
        trajectory = self.motion_planner.plan(
            self.vehicle_state,
            self.target_speed, 
            self.latest_costmap,
            self.costmap_origin_x,
            self.costmap_origin_y,
            self.costmap_resolution
        )
        
        # 4. Publish
        if trajectory:
            self._publish_trajectory(trajectory)
        else:
            self.get_logger().warn("No valid path found!")

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