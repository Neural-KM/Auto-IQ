import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

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
    """Generates and evaluates discrete sampling-based trajectories and chooses the best traj"""
    
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
        Main planning function (vectorized).
        
        1. Generate candidate paths as NumPy arrays
        2. Evaluate costs vectorized
        3. Convert only the best trajectory to TrajectoryPoint list
        """
        if not self.wp_converter:
            return []

        # 1. Update State (Cartesian -> Frenet)
        cur_s, cur_d = self.wp_converter.xy_to_frenet(vehicle_state.x, vehicle_state.y)
        
        # Estimate lateral velocity
        _, _, path_yaw = self.wp_converter.frenet_to_xy(cur_s, cur_d)
        diff_yaw = vehicle_state.yaw - path_yaw
        cur_d_dot = vehicle_state.speed * math.sin(diff_yaw)

        # 2. Generate Candidates (as numpy array dictionaries)
        candidates = []
        for target_d in self.config.d_samples:
            for t_lc in self.config.t_samples:
                traj = self._generate_path_vectorized(
                    vehicle_state,
                    cur_s, cur_d, cur_d_dot, 
                    target_d, t_lc, target_speed
                )
                if traj is not None:
                    candidates.append((traj, t_lc))

        if not candidates:
            return self._generate_stop(vehicle_state)

        # 3. Evaluate Costs
        best_cost = float('inf')
        best_traj = None

        for traj, t_lc in candidates:
            cost = self._calculate_cost_vectorized(
                traj, occupancy_grid,
                grid_origin_x, grid_origin_y, grid_resolution,
                map_to_costmap_tf
            )
            cost += t_lc * self.config.w_time
            
            if cost < best_cost:
                best_cost = cost
                best_traj = traj

        # 4. Convert best trajectory to TrajectoryPoint list
        if best_traj is not None:
            return self._arrays_to_trajectory_points(best_traj)
        else:
            return self._generate_stop(vehicle_state)

    def _generate_path_vectorized(self, vehicle_state, s0, d0, d_dot0, 
                                   target_d, t_lane_change, target_v) -> Optional[Dict[str, np.ndarray]]:
        """
        Generates trajectory as NumPy arrays (vectorized).
        
        Returns:
            Dictionary with keys: 'x', 'y', 'yaw', 's', 'd', 'd_dot', 'speed', 't'
            or None if generation fails
        """
        T = self.config.prediction_horizon
        dt = self.config.dt
        
        # Time array
        t_arr = np.arange(0, T, dt)
        n_pts = len(t_arr)
        
        # Lateral polynomial (lane change completes at t_lane_change)
        lat_qp = QuinticPolynomial(d0, d_dot0, 0.0, target_d, 0.0, 0.0, t_lane_change)
        
        # Longitudinal polynomial
        lon_qp = QuarticPolynomial(s0, vehicle_state.speed, 0.0, target_v, 0.0, T)
        
        # Evaluate polynomials vectorized
        s_arr = lon_qp.calc_point(t_arr)
        v_arr = lon_qp.calc_first_derivative(t_arr)
        
        # After lane change time, d stays at target
        t_clamped = np.minimum(t_arr, t_lane_change)
        d_arr = lat_qp.calc_point(t_clamped)
        d_dot_arr = lat_qp.calc_first_derivative(t_clamped)
        
        # Vectorized Frenet to Cartesian conversion
        try:
            x_arr, y_arr, ref_yaw_arr = self.wp_converter.frenet_to_xy_vectorized(s_arr, d_arr)
        except:
            return None
        
        # Compute actual yaw (corrected for lateral velocity)
        s_dot_arr = lon_qp.calc_first_derivative(t_arr)
        yaw_arr = ref_yaw_arr.copy()
        
        # For points with sufficient forward velocity, correct yaw
        mask = np.abs(s_dot_arr) > 0.1
        yaw_arr[mask] = ref_yaw_arr[mask] + np.arctan2(d_dot_arr[mask], s_dot_arr[mask])
        
        # Override first point with exact vehicle state
        x_arr[0] = vehicle_state.x
        y_arr[0] = vehicle_state.y
        yaw_arr[0] = vehicle_state.yaw
        
        return {
            'x': x_arr,
            'y': y_arr,
            'yaw': yaw_arr,
            's': s_arr,
            'd': d_arr,
            'd_dot': d_dot_arr,
            'speed': v_arr,
            't': t_arr,
        }

    def _calculate_cost_vectorized(self, traj: Dict[str, np.ndarray], grid, 
                                    origin_x, origin_y, resolution,
                                    costmap_to_map_tf) -> float:
        """
        Calculates cost using vectorized operations.
        
        Args:
            traj: Dictionary of numpy arrays
            grid: 2D occupancy grid
            origin_x, origin_y: Grid origin in costmap frame
            resolution: Grid cell size
            costmap_to_map_tf: Transform (tx, ty, yaw) from costmap to map frame
        """
        cost = 0.0
        
        # 1. Reference Cost (prefer staying on path)
        final_d = abs(traj['d'][-1])
        cost += (final_d ** 2) * self.config.w_ref
        
        # 2. Collision Check (vectorized coordinate transform)
        x_arr = traj['x']
        y_arr = traj['y']
        
        if costmap_to_map_tf is not None:
            tx, ty, yaw = costmap_to_map_tf
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            
            # P_costmap = R^T * (P_map - t)
            dx = x_arr - tx
            dy = y_arr - ty
            costmap_x = dx * cos_yaw + dy * sin_yaw
            costmap_y = -dx * sin_yaw + dy * cos_yaw
        else:
            costmap_x = x_arr
            costmap_y = y_arr
        
        # Convert to grid indices (vectorized)
        col_arr = ((costmap_x - origin_x) / resolution).astype(np.int32)
        row_arr = ((costmap_y - origin_y) / resolution).astype(np.int32)
        
        # Bounds check
        h, w = grid.shape
        valid_mask = (col_arr >= 0) & (col_arr < w) & (row_arr >= 0) & (row_arr < h)
        
        # Get valid indices
        valid_rows = row_arr[valid_mask]
        valid_cols = col_arr[valid_mask]
        
        if len(valid_rows) == 0:
            return cost
        
        # Lookup occupancy values (vectorized)
        vals = grid[valid_rows, valid_cols]
        
        # Check for lethal collision
        if np.any(vals > self.config.lethal_thresh):
            return float('inf')
        
        # Sum collision costs for non-zero cells
        cost += np.sum(vals[vals > 0]) * self.config.w_collision
        
        return cost

    def _arrays_to_trajectory_points(self, traj: Dict[str, np.ndarray]) -> List[TrajectoryPoint]:
        """Convert trajectory arrays to list of TrajectoryPoint objects."""
        n_pts = len(traj['x'])
        points = []
        
        for i in range(n_pts):
            points.append(TrajectoryPoint(
                x=float(traj['x'][i]),
                y=float(traj['y'][i]),
                yaw=float(traj['yaw'][i]),
                speed=float(traj['speed'][i]),
                s=float(traj['s'][i]),
                d=float(traj['d'][i]),
                time_from_start=float(traj['t'][i]),
            ))
        
        return points

    def _world_to_grid(self, world_x: float, world_y: float,
                        origin_x: float, origin_y: float, resolution: float,
                        costmap_to_map_tf: Tuple[float, float, float] = None) -> Tuple[int, int]:
        """
        Convert world (map frame) coordinates to grid indices.
        """
        tx, ty, yaw = costmap_to_map_tf
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        
        dx = world_x - tx
        dy = world_y - ty
        costmap_x = dx * cos_yaw + dy * sin_yaw
        costmap_y = dx * (-sin_yaw) + dy * cos_yaw
        
        col = int((costmap_x - origin_x) / resolution)
        row = int((costmap_y - origin_y) / resolution)
        return row, col

    def _generate_stop(self, state) -> List[TrajectoryPoint]:
        """Generates a safe braking trajectory."""
        pts = []
        v = state.speed
        x, y = state.x, state.y
        yaw = state.yaw
        t = 0.0
        
        while v > 0:
            v -= 4.0 * self.config.dt
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