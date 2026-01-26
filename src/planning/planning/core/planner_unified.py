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
    prediction_horizon: float = 10.0 # Lookahead time
    
    # The Discrete Set of Choices: [Center, Left Lane, Right Lane, Nudge Left, Nudge Right]
    d_samples: List[float] = field(default_factory=lambda: [0.0, -4.0, -2.0, 2.0,-1.0, 1.0,
                                                                4.0])
    
    # Cost Weights
    w_collision: float = 1000.0    # Safety is priority
    w_ref: float = 1.0             # Preference for lane center
    # w_smooth: float = 5.0          # Penalize lateral movement
    
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

        # 2. Generate Candidates
        candidates = []
        for target_d in self.config.d_samples:
            # Generate path from current (s, d) to target (s+dist, target_d)
            traj = self._generate_path(
                cur_s, cur_d, cur_d_dot, 
                target_d, target_speed, vehicle_state.speed
            )
            if traj:
                candidates.append(traj)

        # 3. Evaluate Costs
        best_cost = float('inf')
        best_traj = []

        for traj in candidates:
            cost = self._calculate_cost(
                traj, occupancy_grid,
                grid_origin_x, grid_origin_y, grid_resolution,
                map_to_costmap_tf
            )
            
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

    def _generate_path(self, s0, d0, d_dot0, target_d, target_v, current_v):
        """Generates a smooth polynomial trajectory."""
        T = self.config.prediction_horizon
        dt = self.config.dt
        points = []
        
        # Lateral: Quintic Polynomial (Smooth jerk-minimized lane change)
        lat_qp = QuinticPolynomial(d0, d_dot0, 0.0, target_d, 0.0, 0.0, T)
        
        # Longitudinal: Quartic Polynomial (Smooth velocity profile)
        lon_qp = QuarticPolynomial(s0, current_v, 0.0, target_v, 0.0, T)
        
        for t in np.arange(0, T, dt):
            s_t = lon_qp.calc_point(t)
            d_t = lat_qp.calc_point(t)
            v_t = lon_qp.calc_first_derivative(t)
            
            try:
                # Convert back to Global Coordinates
                x, y, yaw = self.wp_converter.frenet_to_xy(s_t, d_t)
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