import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Import common types
from autoiq_common.types import VehicleState, TrajectoryPoint
from planning.utils.frenet_utils import (
    WaypointConverter, 
    QuinticPolynomial, 
    QuarticPolynomial
)

@dataclass
class PlannerConfig:
    """Configuration for Discrete Sampling Planner."""
    dt: float = 0.1                # Time step
    prediction_horizon: float = 5.0 # Lookahead time
    
    # Lateral offset samples (final d values)
    d_samples: List[float] = field(default_factory=lambda: [0.0, -4.0, -2.0, 2.0, -1.0, 1.0, 4.0])
    
    # Lane change time samples (how fast to reach target d)
    t_samples: List[float] = field(default_factory=lambda: [2.0, 3.0, 5.0])
    
    # Cost Weights
    w_collision: float = 1000.0    # Safety is priority
    w_ref: float = 1.0             # Preference for lane center
    w_time: float = 0.1            # Prefer faster lane changes (when safe)
    
    lethal_thresh: int = 80        # Probability [0-100]

class SamplingBasedPlanner:
    # ==================== Planner ====================
    def __init__(self, config: PlannerConfig = None):
        self.config = config or PlannerConfig()
        self.wp_converter: Optional[WaypointConverter] = None
        
    def set_reference_path(self, path_x: List[float], path_y: List[float]):
        """Builds the Frenet converter from the Global Route."""
        if len(path_x) > 2:
            self.wp_converter = WaypointConverter(path_x, path_y)

    def plan(self, 
             vehicle_state: VehicleState,
             target_speed: float,
             occupancy_grid: np.ndarray,
             grid_origin_x: float,
             grid_origin_y: float,
             grid_resolution: float,
             map_to_costmap_tf: Tuple[float, float, float] = None) -> List[TrajectoryPoint]:
        """
        Main function:
        1. Generate candidate paths.
        2. Check each against the grid.
        3. Return the best one.
        
        Args:
            vehicle_state: Current vehicle state
            target_speed: Target speed in m/s
            occupancy_grid: 2D numpy array of occupancy values
            grid_origin_x: X position of cell [0,0] in costmap frame
            grid_origin_y: Y position of cell [0,0] in costmap frame
            grid_resolution: Cell size in meters
            map_to_costmap_tf: Transform from map to costmap frame as (tx, ty, yaw)
                               If None, assumes costmap is in map frame
        """
        if not self.wp_converter:
            return []

        # 1. Update State (Cartesian -> Frenet)
        cur_s, cur_d = self.wp_converter.xy_to_frenet(vehicle_state.x, vehicle_state.y)
        
        # Estimate lateral velocity (approximate)
        _, _, path_yaw = self.wp_converter.frenet_to_xy(cur_s, cur_d)
        diff_yaw = vehicle_state.yaw - path_yaw
        cur_d_dot = vehicle_state.speed * math.sin(diff_yaw)

        # 2. Generate Candidates (sample both target_d and lane change time)
        candidates = []
        for target_d in self.config.d_samples:
            for t_lc in self.config.t_samples:
                # Generate path from current (s, d) to target (s+dist, target_d)
                traj = self._generate_path(
                    vehicle_state,
                    cur_s, cur_d, cur_d_dot, 
                    target_d, t_lc, target_speed
                )
                if traj:
                    candidates.append((traj, t_lc))

        # 3. Evaluate Costs
        best_cost = float('inf')
        best_traj = []

        for traj, t_lc in candidates:
            cost = self._calculate_cost(
                traj, occupancy_grid,
                grid_origin_x, grid_origin_y, grid_resolution,
                map_to_costmap_tf
            )
            # Add time penalty (prefer faster lane changes when collision-free)
            cost += t_lc * self.config.w_time
            
            if cost < best_cost:
                best_cost = cost
                best_traj = traj
        #     print(f"Traj Cost: {cost:.2f} for target_d={traj[-1].d:.2f}")
        # print(f"Best Traj Cost: {best_cost:.2f} with {len(best_traj)} points.")
        if best_traj:
            return best_traj
        else:
            # Fallback: Emergency Stop if all paths are blocked
            return self._generate_stop(vehicle_state)

    def _generate_path(self, vehicle_state, s0, d0, d_dot0, target_d, t_lane_change, target_v):
        """
        Generates a smooth polynomial trajectory.
        
        Args:
            vehicle_state: Current vehicle state (used for first point)
            s0, d0, d_dot0: Current Frenet state
            target_d: Target lateral offset
            t_lane_change: Time to complete lane change
            target_v: Target longitudinal velocity
        """
        T = self.config.prediction_horizon
        dt = self.config.dt
        points = []
        
        # Lateral: Quintic Polynomial (lane change completes at t_lane_change)
        lat_qp = QuinticPolynomial(d0, d_dot0, 0.0, target_d, 0.0, 0.0, t_lane_change)
        
        # Longitudinal: Quartic Polynomial (Smooth velocity profile)
        lon_qp = QuarticPolynomial(s0, vehicle_state.speed, 0.0, target_v, 0.0, T)
        
        for i, t in enumerate(np.arange(0, T, dt)):
            s_t = lon_qp.calc_point(t)
            # After lane change time, d stays at target
            d_t = lat_qp.calc_point(min(t, t_lane_change))
            d_dot_t = lat_qp.calc_first_derivative(min(t, t_lane_change))
            v_t = lon_qp.calc_first_derivative(t)
            
            try:
                # Convert back to Global Coordinates
                x, y, ref_yaw = self.wp_converter.frenet_to_xy(s_t, d_t)
                
                # First point: use exact vehicle position to avoid shift
                if i == 0:
                    x = vehicle_state.x
                    y = vehicle_state.y
                    yaw = vehicle_state.yaw
                else:
                    # Correct yaw for lateral velocity
                    # True heading = ref_yaw + atan2(d_dot, s_dot)
                    s_dot_t = lon_qp.calc_first_derivative(t)
                    if abs(s_dot_t) > 0.1:
                        yaw = ref_yaw + math.atan2(d_dot_t, s_dot_t)
                    else:
                        yaw = ref_yaw
                
                points.append(TrajectoryPoint(
                    x=x,
                    y=y,
                    yaw=yaw,
                    speed=v_t,
                    s=s_t,
                    d=d_t,
                    time_from_start=t,
                ))
            except:
                break # Stop if we run out of reference path
                
        return points

    def _world_to_grid(self, world_x: float, world_y: float,
                        origin_x: float, origin_y: float, resolution: float,
                        costmap_to_map_tf: Tuple[float, float, float] = None) -> Tuple[int, int]:
        """
        Convert world (map frame) coordinates to grid indices.
        
        Args:
            world_x, world_y: Point in map/world frame
            origin_x, origin_y: Position of cell [0,0] in costmap frame
            resolution: Grid cell size in meters
            costmap_to_map_tf: Transform (tx, ty, yaw) from costmap to map frame
                               (i.e., costmap origin position and orientation in map frame)
        
        Returns:
            (row, col) grid indices
        """
        # Transform: costmap_to_map gives us P_map = R * P_costmap + t
        # We need the inverse: P_costmap = R^T * (P_map - t)
        tx, ty, yaw = costmap_to_map_tf
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        
        # Translate then rotate by inverse (transpose)
        dx = world_x - tx
        dy = world_y - ty
        costmap_x = dx * cos_yaw + dy * sin_yaw
        costmap_y = dx * (-sin_yaw) + dy * cos_yaw
        
        col = int((costmap_x - origin_x) / resolution)
        row = int((costmap_y - origin_y) / resolution)
        return row, col

    def _calculate_cost(self, trajectory, grid, origin_x, origin_y, resolution,
                         map_to_costmap_tf):
        """Calculates cost: Reference tracking + Smoothness + Collision.
        
        Args:
            trajectory: List of TrajectoryPoint in map frame
            grid: 2D occupancy grid
            origin_x, origin_y: Position of cell [0,0] in costmap frame
            resolution: Grid cell size in meters
            map_to_costmap_tf: Transform (tx, ty, yaw) from map to costmap frame
                               If None, assumes costmap is in map frame (identity transform)
        """
        cost = 0.0
        
        # 1. Reference Cost (Prefer staying on path)
        final_d = abs(trajectory[-1].d)
        cost += (final_d ** 2) * self.config.w_ref
        
        # 2. Collision Check
        for pt in trajectory:
            row, col = self._world_to_grid(
                pt.x, pt.y, origin_x, origin_y, resolution, map_to_costmap_tf
            )
            
            # Check Bounds + Occupancy
            if 0 <= col < grid.shape[1] and 0 <= row < grid.shape[0]:
                val = grid[row, col]
                if val > self.config.lethal_thresh:
                    return float('inf') # Hard Collision
                if val > 0:
                    cost += val * self.config.w_collision
        
        return cost

    def _generate_stop(self, state):
        """Generates a safe braking trajectory."""
        pts = []
        v = state.speed
        x, y = state.x, state.y
        yaw = state.yaw
        t = 0.0
        
        while v > 0:
            v -= 4.0 * self.config.dt # Decelerate at 4 m/s^2
            if v < 0: v = 0
            
            x += v * math.cos(yaw) * self.config.dt
            y += v * math.sin(yaw) * self.config.dt
            
            t += self.config.dt
            pts.append(TrajectoryPoint(
                x=x,
                y=y,
                yaw=yaw,
                speed=v,
                s=0.0,
                d=0.0,
                time_from_start=t,
            ))
            if v == 0: break
            
        return pts