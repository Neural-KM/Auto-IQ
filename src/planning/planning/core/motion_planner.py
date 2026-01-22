"""
Motion Planner (Trajectory Generation)
=======================================

CARLA-agnostic local trajectory planner using Frenet coordinates.
Receives behavior decisions from BehaviorPlanner, generates trajectories.

Responsibilities:
- Convert between Cartesian and Frenet coordinates
- Generate smooth trajectories (lane following, lane change, stop)
- Track Frenet state (s, d, velocities)

Does NOT:
- Make behavior decisions (that's BehaviorPlanner)
- Access CARLA directly (receives data via arguments)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Import common types from autoiq_common
from autoiq_common.types import VehicleState, TrajectoryPoint

from planning.utils.frenet_utils import (
    WaypointConverter, 
    QuinticPolynomial, 
    QuarticPolynomial,
)
from planning.behavior_planner import BehaviorState, BehaviorOutput


# VehicleState and TrajectoryPoint are now imported from autoiq_common.types


@dataclass
class MotionPlannerConfig:
    """Configuration for motion planner."""
    lane_width: float = 3.5
    lane_change_time: float = 3.0  # seconds
    longitudinal_horizon: float = 4.0  # seconds
    dt: float = 0.1  # trajectory sampling interval
    # Smoothing parameters
    d_dot_filter_alpha: float = 0.2
    max_d_dot: float = 1.5  # m/s
    # Stop trajectory
    stop_deceleration: float = 3.0  # m/s^2
    min_stop_distance: float = 5.0  # meters ahead


class MotionPlanner:
    """
    Frenet-based trajectory generator.
    
    Receives behavior decisions, generates appropriate trajectories.
    """
    
    def __init__(self, config: Optional[MotionPlannerConfig] = None):
        self.config = config or MotionPlannerConfig()
        
        # Frenet converter
        self.wp_converter: Optional[WaypointConverter] = None
        
        # Frenet state (filtered)
        self.current_s: float = 0.0
        self.current_d: float = 0.0
        self.current_d_dot: float = 0.0
        
        # Raw state for velocity calculation
        self._prev_d_raw: float = 0.0
        
        # Lane change tracking (for trajectory generation)
        self._lane_change_start_d: float = 0.0
        self._prev_behavior_state: Optional[BehaviorState] = None
        
    def set_reference_path(self, path_x: List[float], path_y: List[float]) -> bool:
        """
        Set the reference path (centerline) for Frenet conversion.
        
        Args:
            path_x: List of x coordinates
            path_y: List of y coordinates
            
        Returns:
            True if successful
        """
        if len(path_x) < 2 or len(path_y) < 2:
            return False
            
        self.wp_converter = WaypointConverter(path_x, path_y)
        self.current_s = 0.0
        self.current_d = 0.0
        self.current_d_dot = 0.0
        self._prev_d_raw = 0.0
        return True
    
    def update_vehicle_state(self, state: VehicleState, dt: float = 0.1):
        """
        Update internal Frenet state from vehicle state.
        
        Args:
            state: Current vehicle state
            dt: Time since last update
        """
        if not self.wp_converter:
            return
        
        # Get filtered Frenet coordinates
        self.current_s, self.current_d = self.wp_converter.xy_to_frenet(
            state.x, state.y, filter_output=True
        )
        
        # Estimate and filter lateral velocity
        if dt > 0:
            d_dot_raw = (self.current_d - self._prev_d_raw) / dt
            d_dot_raw = np.clip(d_dot_raw, -self.config.max_d_dot, self.config.max_d_dot)
            
            alpha = self.config.d_dot_filter_alpha
            self.current_d_dot = alpha * d_dot_raw + (1 - alpha) * self.current_d_dot
        
        self._prev_d_raw = self.current_d
    
    def get_frenet_state(self) -> Tuple[float, float, float]:
        """Get current Frenet state (s, d, d_dot)."""
        return self.current_s, self.current_d, self.current_d_dot
    
    def generate_trajectory(
        self,
        behavior: BehaviorOutput,
        speed: float,
        target_speed: float = 8.33  # 30 km/h default
    ) -> List[TrajectoryPoint]:
        """
        Generate trajectory based on behavior decision.
        
        Args:
            behavior: Output from BehaviorPlanner
            speed: Current vehicle speed in m/s
            target_speed: Desired cruise speed in m/s
            
        Returns:
            List of trajectory points
        """
        if not self.wp_converter:
            return []
        
        # Track lane change start for smooth trajectory
        if behavior.state in [BehaviorState.LANE_CHANGING_LEFT, 
                               BehaviorState.LANE_CHANGING_RIGHT]:
            if self._prev_behavior_state not in [BehaviorState.LANE_CHANGING_LEFT,
                                                   BehaviorState.LANE_CHANGING_RIGHT]:
                # Just started lane change - record starting d
                self._lane_change_start_d = self.current_d
        
        self._prev_behavior_state = behavior.state
        
        # Common parameters
        T_lon = self.config.longitudinal_horizon
        dt = self.config.dt
        s0 = self.current_s
        v0 = max(speed, 0.1)
        
        # Generate trajectory based on state
        if behavior.should_stop:
            return self._generate_stop_trajectory(s0, v0, dt)
            
        elif behavior.state in [BehaviorState.LANE_CHANGING_LEFT, 
                                 BehaviorState.LANE_CHANGING_RIGHT]:
            return self._generate_lane_change_trajectory(
                s0, v0, target_speed, behavior.target_d, 
                behavior.lane_change_progress, T_lon, dt
            )
        else:
            # LANE_FOLLOWING or PREPARING
            return self._generate_lane_follow_trajectory(
                s0, v0, target_speed, behavior.target_d, T_lon, dt
            )
    
    def _generate_lane_follow_trajectory(
        self, 
        s0: float, 
        v0: float, 
        target_speed: float,
        target_d: float,
        T: float, 
        dt: float
    ) -> List[TrajectoryPoint]:
        """
        Generate trajectory for lane following.
        
        Simple path sampling at constant lateral offset.
        """
        trajectory = []
        
        # Longitudinal: smooth acceleration to target speed
        lon_qp = QuarticPolynomial(s0, v0, 0, target_speed, 0, T)
        
        t_values = [t * dt for t in range(int(T / dt) + 1)]
        
        for t in t_values:
            s_t = lon_qp.calc_point(t)
            v_t = lon_qp.calc_first_derivative(t)
            d_t = target_d
            
            x, y, yaw = self.wp_converter.frenet_to_xy(s_t, d_t)
            
            trajectory.append(TrajectoryPoint(
                x=x, y=y, yaw=yaw, s=s_t, d=d_t, speed=v_t
            ))
            
        return trajectory
    
    def _generate_lane_change_trajectory(
        self, 
        s0: float, 
        v0: float, 
        target_speed: float,
        target_d: float,
        progress: float,
        T: float, 
        dt: float
    ) -> List[TrajectoryPoint]:
        """
        Generate trajectory for lane change.
        
        Uses quintic polynomial for smooth lateral transition.
        """
        trajectory = []
        
        T_lat = self.config.lane_change_time
        
        # Remaining time in lane change
        elapsed = progress * T_lat
        t_remaining = max(0.1, T_lat - elapsed)
        
        # Current lateral state
        d0 = self.current_d
        d_dot0 = self.current_d_dot
        
        # Polynomials
        lat_qp = QuinticPolynomial(d0, d_dot0, 0, target_d, 0, 0, t_remaining)
        lon_qp = QuarticPolynomial(s0, v0, 0, target_speed, 0, T)
        
        t_values = [t * dt for t in range(int(T / dt) + 1)]
        
        for t in t_values:
            s_t = lon_qp.calc_point(t)
            v_t = lon_qp.calc_first_derivative(t)
            
            if t < t_remaining:
                d_t = lat_qp.calc_point(t)
            else:
                d_t = target_d
            
            x, y, yaw = self.wp_converter.frenet_to_xy(s_t, d_t)
            
            trajectory.append(TrajectoryPoint(
                x=x, y=y, yaw=yaw, s=s_t, d=d_t, speed=v_t
            ))
            
        return trajectory
    
    def _generate_stop_trajectory(
        self, 
        s0: float, 
        v0: float, 
        dt: float
    ) -> List[TrajectoryPoint]:
        """
        Generate emergency stop trajectory.
        
        Decelerates to zero speed while maintaining lane position.
        """
        trajectory = []
        
        # Calculate stopping distance and time
        # v^2 = v0^2 - 2*a*d  =>  d = v0^2 / (2*a)
        decel = self.config.stop_deceleration
        stop_distance = max(self.config.min_stop_distance, (v0 * v0) / (2 * decel))
        stop_time = v0 / decel if decel > 0 else 1.0
        
        # Generate points until stopped
        T = max(stop_time + 1.0, 2.0)  # Extra time after stop
        t_values = [t * dt for t in range(int(T / dt) + 1)]
        
        d_t = self.current_d  # Maintain current lane
        
        for t in t_values:
            if t < stop_time:
                # Decelerating
                v_t = max(0, v0 - decel * t)
                s_t = s0 + v0 * t - 0.5 * decel * t * t
            else:
                # Stopped
                v_t = 0.0
                s_t = s0 + stop_distance
            
            x, y, yaw = self.wp_converter.frenet_to_xy(s_t, d_t)
            
            trajectory.append(TrajectoryPoint(
                x=x, y=y, yaw=yaw, s=s_t, d=d_t, speed=v_t
            ))
            
        return trajectory
    
    def reset(self):
        """Reset planner state."""
        self.current_s = 0.0
        self.current_d = 0.0
        self.current_d_dot = 0.0
        self._prev_d_raw = 0.0
        self._lane_change_start_d = 0.0
        self._prev_behavior_state = None
        
        if self.wp_converter:
            self.wp_converter.reset_filter()
