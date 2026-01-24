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
    d_samples: List[float] = field(default_factory=lambda: [0.0, -4.0,
                                                                4.0])
    
    # Cost Weights
    w_collision: float = 1000.0    # Safety is priority
    w_ref: float = 1.0             # Preference for lane center
    w_smooth: float = 5.0          # Penalize lateral movement
    
    # Grid Parameters (SHOULD match mapping Node)
    grid_res: float = 0.2          # meters/cell
    grid_w: int = 200              # width in cells
    grid_h: int = 200              # height in cells
    lethal_thresh: int = 80        # Probability [0-100]

class OptimizationMotionPlanner:
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
             grid_origin_x: float = 0.0,
             grid_origin_y: float = 0.0,
             grid_resolution: float = 0.2) -> List[TrajectoryPoint]:
        """
        Main function:
        1. Generate 5 candidate paths.
        2. Check each against the grid.
        3. Return the best one.
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

        # Optimization Helpers (Pre-calc trig for coordinate transform)
        # We need to transform Global Points -> Ego Frame to check the local grid
        cos_yaw = math.cos(vehicle_state.yaw)
        sin_yaw = math.sin(vehicle_state.yaw)

        for traj in candidates:
            cost = self._calculate_cost(
                traj, occupancy_grid,
                grid_origin_x, grid_origin_y, grid_resolution
            )
            
            if cost < best_cost:
                best_cost = cost
                best_traj = traj
            # print(f"Traj Cost: {cost:.2f} for target_d={traj[-1].d:.2f}")
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

    def _calculate_cost(self, trajectory, grid, origin_x, origin_y, resolution):
        """Calculates cost: Reference + Smoothness + Collision.
        
        Grid is world-aligned (not ego-aligned). origin_x, origin_y is the
        world position of cell [0,0].
        """
        cost = 0.0
        
        # 1. Reference Cost (Prefer staying on path)
        final_d = abs(trajectory[-1].d)
        cost += (final_d ** 2) * self.config.w_ref
        
        # 2. Smoothness Cost (Penalize lateral offset magnitude)
        cost += abs(final_d) * self.config.w_smooth
        
        # 3. Collision Check (World coords -> Grid Index)
        for pt in trajectory:
            # Convert world coordinates to grid indices
            # Grid is world-aligned, so no rotation needed
            grid_x = int((pt.x - origin_x) / resolution)
            grid_y = int((pt.y - origin_y) / resolution)
            
            # Check Bounds + Occupancy (2D grid)
            if 0 <= grid_x < grid.shape[1] and 0 <= grid_y < grid.shape[0]:
                val = grid[grid_y, grid_x]
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