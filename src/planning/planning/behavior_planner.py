"""
Behavior Planner
================

State machine for high-level driving behavior decisions.
CARLA-agnostic - receives data via function arguments.

Responsibilities:
- Decide WHEN to change lanes (not HOW)
- Track behavior states
- Handle obstacle avoidance logic
- Manage lane indices (not raw offsets)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum, auto


class BehaviorState(Enum):
    """Possible behavior states."""
    LANE_FOLLOWING = auto()
    PREPARING_LANE_CHANGE = auto()
    LANE_CHANGING_LEFT = auto()
    LANE_CHANGING_RIGHT = auto()
    EMERGENCY_STOP = auto()


@dataclass 
class ObstacleInfo:
    """Obstacle in ego-vehicle frame."""
    x: float  # Forward distance (+ ahead, - behind)
    y: float  # Lateral distance (+ left, - right)
    distance: float
    vx: float = 0.0  # Relative longitudinal velocity (+ approaching, - receding)
    vy: float = 0.0  # Relative lateral velocity


@dataclass
class LaneAvailability:
    """Lane change availability from map."""
    left: bool
    right: bool


@dataclass
class BehaviorPlannerConfig:
    """Configuration for behavior planner."""
    lane_width: float = 3.5
    lane_change_time: float = 3.0  # seconds to complete lane change
    # TTC-based triggering (replaces pure distance)
    ttc_threshold: float = 4.0  # seconds - trigger lane change if TTC < this
    ttc_critical: float = 2.0  # seconds - immediate action required
    min_trigger_distance: float = 10.0  # meters - always trigger if closer than this
    max_trigger_distance: float = 60.0  # meters - ignore obstacles beyond this
    safety_buffer: float = 5.0  # meters
    lane_threshold: float = 2.0  # lateral threshold for "in lane"
    lane_change_complete_threshold: float = 0.3  # meters (tighter than before)
    lane_change_complete_velocity_threshold: float = 0.2  # m/s lateral velocity
    # Timing
    prepare_timeout: float = 2.0  # max seconds in PREPARING state
    lane_change_cooldown: float = 3.0  # seconds before next lane change
    # Adjacent lane check
    adjacent_lane_check_distance: float = 20.0  # meters ahead
    adjacent_lane_check_behind: float = 8.0  # meters behind
    adjacent_lane_ttc_threshold: float = 3.0  # seconds - lane must be clear for this TTC


@dataclass
class BehaviorOutput:
    """Output from behavior planner to motion planner."""
    state: BehaviorState
    target_lane: int  # Lane index (0 = reference lane, +1 = left, -1 = right)
    target_d: float  # Lateral offset = target_lane * lane_width
    lane_change_progress: float  # 0.0 to 1.0 during lane change
    should_stop: bool  # Emergency stop requested


class BehaviorPlanner:
    """
    High-level behavior planning using state machine.
    
    Uses lane INDICES instead of raw d offsets for cleaner logic
    and no floating-point drift on multi-lane roads.
    """
    
    def __init__(self, config: Optional[BehaviorPlannerConfig] = None):
        self.config = config or BehaviorPlannerConfig()
        
        # Lane tracking (index-based)
        self.current_lane: int = 0  # 0 = reference lane
        self.target_lane: int = 0
        
        # Frenet state (updated externally)
        self.current_d: float = 0.0
        self.current_d_dot: float = 0.0
        
        # State machine
        self.behavior_state = BehaviorState.LANE_FOLLOWING
        
        # Timing
        self._lane_change_elapsed: float = 0.0
        self._prepare_elapsed: float = 0.0
        self._cooldown_remaining: float = 0.0
        
        # Hysteresis for obstacle detection
        self._consecutive_clear_count: int = 0
        self._consecutive_block_count: int = 0
        
    @property
    def target_d(self) -> float:
        """Compute target_d from lane index."""
        return self.target_lane * self.config.lane_width
    
    def update_frenet_state(self, current_d: float, current_d_dot: float):
        """Update internal state from motion planner's Frenet estimates."""
        self.current_d = current_d
        self.current_d_dot = current_d_dot
        
        # Update current lane estimate based on position
        # This handles cases where we've drifted or completed a lane change
        self.current_lane = round(current_d / self.config.lane_width)
    
    def compute_ttc(self, obstacle: ObstacleInfo, ego_speed: float) -> float:
        """
        Compute Time-to-Collision with an obstacle.
        
        Args:
            obstacle: Obstacle info with position and velocity
            ego_speed: Ego vehicle speed in m/s
            
        Returns:
            TTC in seconds. Returns float('inf') if no collision expected.
        """
        if obstacle.x <= 0:
            # Obstacle behind us
            # For now we expect the car behind to not ram into us
            return float('inf')
        
        # Closing velocity = ego_speed - obstacle_speed_in_ego_direction
        # obstacle.vx is obstacle's velocity projected onto ego's forward axis
        # If obstacle.vx > 0, obstacle is moving forward (away from us if ahead)
        # If obstacle.vx < 0, obstacle is moving backward (toward us)
        # 
        # closing_velocity > 0 means we're catching up to the obstacle
        closing_velocity = ego_speed - obstacle.vx
        
        # If obstacle velocity not provided (vx=0), assume stationary
        # closing_velocity = ego_speed (we're approaching stationary object)
        
        if closing_velocity <= 0.1:
            # Not closing or gap is opening
            return float('inf')
        
        ttc = obstacle.x / closing_velocity
        return max(0.0, ttc)
    
    def is_obstacle_threatening(
        self, 
        obstacle: ObstacleInfo, 
        ego_speed: float
    ) -> Tuple[bool, float]:
        """
        Determine if obstacle is a threat using TTC.
        
        Returns:
            (is_threat, ttc) tuple
        """
        # Distance-based bounds first
        if obstacle.x > self.config.max_trigger_distance:
            return False, float('inf')
        
        ttc = self.compute_ttc(obstacle, ego_speed)
        
        # Critical distance - always a threat
        if obstacle.x < self.config.min_trigger_distance:
            return True, ttc
        
        # TTC-based threat assessment
        is_threat = ttc < self.config.ttc_threshold
        
        return is_threat, ttc
    
    def update(
        self, 
        obstacles: List[ObstacleInfo],
        lane_availability: LaneAvailability,
        speed: float,
        dt: float = 0.1
    ) -> BehaviorOutput:
        """
        Update behavior state machine.
        
        Args:
            obstacles: List of all detected obstacles in ego frame
            lane_availability: Which lanes are available (from map)
            speed: Current vehicle speed in m/s
            dt: Time step
            
        Returns:
            BehaviorOutput with decisions for motion planner
        """
        # Decrement cooldown
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= dt
        
        # State machine
        if self.behavior_state == BehaviorState.LANE_FOLLOWING:
            self._handle_lane_following(obstacles, speed)
            
        elif self.behavior_state == BehaviorState.PREPARING_LANE_CHANGE:
            self._handle_preparing_lane_change(
                obstacles, lane_availability, speed, dt
            )
            
        elif self.behavior_state in [
            BehaviorState.LANE_CHANGING_LEFT,
            BehaviorState.LANE_CHANGING_RIGHT
        ]:
            self._handle_lane_changing(dt)
            
        # EMERGENCY_STOP: no state transition, wait for reset
        
        # Compute output
        lane_change_progress = 0.0
        if self.behavior_state in [BehaviorState.LANE_CHANGING_LEFT, 
                                    BehaviorState.LANE_CHANGING_RIGHT]:
            lane_change_progress = min(1.0, self._lane_change_elapsed / self.config.lane_change_time)
        
        return BehaviorOutput(
            state=self.behavior_state,
            target_lane=self.target_lane,
            target_d=self.target_d,
            lane_change_progress=lane_change_progress,
            should_stop=(self.behavior_state == BehaviorState.EMERGENCY_STOP)
        )
    
    def _handle_lane_following(
        self, 
        obstacles: List[ObstacleInfo], 
        ego_speed: float
    ):
        """Handle LANE_FOLLOWING state with TTC-based detection."""
        # Only consider lane change if cooldown expired
        if self._cooldown_remaining > 0:
            self._consecutive_block_count = 0
            return
        
        # Check for threatening obstacle in our target lane
        blocking, ttc = self._find_blocking_obstacle(obstacles, ego_speed)
        
        if blocking:
            self._consecutive_block_count += 1
            self._consecutive_clear_count = 0
            
            # Critical TTC = immediate action
            if ttc < self.config.ttc_critical:
                self.behavior_state = BehaviorState.PREPARING_LANE_CHANGE
                self._prepare_elapsed = 0.0
                self._consecutive_block_count = 0
            # Normal TTC threshold - require confirmation
            elif self._consecutive_block_count >= 3:
                self.behavior_state = BehaviorState.PREPARING_LANE_CHANGE
                self._prepare_elapsed = 0.0
                self._consecutive_block_count = 0
        else:
            self._consecutive_clear_count += 1
            self._consecutive_block_count = 0
    
    def _handle_preparing_lane_change(
        self,
        obstacles: List[ObstacleInfo],
        lane_availability: LaneAvailability,
        ego_speed: float,
        dt: float
    ):
        """Handle PREPARING_LANE_CHANGE state with timeout."""
        self._prepare_elapsed += dt
        
        # TIMEOUT - prevents getting stuck forever
        if self._prepare_elapsed > self.config.prepare_timeout:
            self.behavior_state = BehaviorState.EMERGENCY_STOP
            self._prepare_elapsed = 0.0
            return
        
        # Re-check if obstacle still there
        blocking, ttc = self._find_blocking_obstacle(obstacles, ego_speed)
        
        if not blocking:
            # Obstacle cleared - return to lane following with hysteresis
            self._consecutive_clear_count += 1
            if self._consecutive_clear_count >= 5:
                self.behavior_state = BehaviorState.LANE_FOLLOWING
                self._prepare_elapsed = 0.0
                self._consecutive_clear_count = 0
            return
        
        self._consecutive_clear_count = 0
        
        # Check which lane is available AND clear (using TTC for adjacent lanes too)
        left_clear = self._is_adjacent_lane_clear(obstacles, 'left', ego_speed)
        right_clear = self._is_adjacent_lane_clear(obstacles, 'right', ego_speed)
        
        if lane_availability.left and left_clear:
            self._start_lane_change('left')
        elif lane_availability.right and right_clear:
            self._start_lane_change('right')
        # else: keep waiting (timeout will trigger EMERGENCY_STOP)
    
    def _handle_lane_changing(self, dt: float):
        """Handle LANE_CHANGING states with proper completion logic."""
        self._lane_change_elapsed += dt
        
        time_complete = self._lane_change_elapsed >= self.config.lane_change_time
        position_complete = abs(self.current_d - self.target_d) < self.config.lane_change_complete_threshold
        velocity_stable = abs(self.current_d_dot) < self.config.lane_change_complete_velocity_threshold
        
        # FIX: Use AND logic for safety - must meet ALL conditions
        if time_complete and position_complete and velocity_stable:
            self._complete_lane_change()
        elif time_complete and position_complete:
            # Time and position OK but still have lateral velocity - wait a bit more
            # But don't wait forever
            if self._lane_change_elapsed > self.config.lane_change_time + 1.0:
                self._complete_lane_change()
    
    def _complete_lane_change(self):
        """Complete lane change and update lane index."""
        # Update current lane to match target
        self.current_lane = self.target_lane
        
        self.behavior_state = BehaviorState.LANE_FOLLOWING
        self._lane_change_elapsed = 0.0
        self._cooldown_remaining = self.config.lane_change_cooldown
    
    def _start_lane_change(self, direction: str):
        """Initialize lane change maneuver."""
        self._lane_change_elapsed = 0.0
        self._prepare_elapsed = 0.0
        
        if direction == 'left':
            self.behavior_state = BehaviorState.LANE_CHANGING_LEFT
            self.target_lane = self.current_lane + 1
        else:
            self.behavior_state = BehaviorState.LANE_CHANGING_RIGHT
            self.target_lane = self.current_lane - 1
    
    def _find_blocking_obstacle(
        self, 
        obstacles: List[ObstacleInfo], 
        ego_speed: float
    ) -> Tuple[Optional[ObstacleInfo], float]:
        """
        Find obstacle blocking our target lane using TTC.
        
        Returns:
            (obstacle, ttc) - obstacle if threatening, else (None, inf)
        """
        # Lateral offset from current position to target lane center
        lateral_to_target = self.target_d - self.current_d
        
        most_threatening: Optional[ObstacleInfo] = None
        min_ttc = float('inf')
        
        for obs in obstacles:
            if obs.x <= 0:  # Behind us
                continue
            if obs.x > self.config.max_trigger_distance:
                continue
            
            # Check if obstacle is in path to target lane
            obs_relative_to_target = abs(obs.y - lateral_to_target)
            
            if obs_relative_to_target < self.config.lane_threshold:
                # Obstacle is in our lane - check TTC
                is_threat, ttc = self.is_obstacle_threatening(obs, ego_speed)
                
                if is_threat and ttc < min_ttc:
                    min_ttc = ttc
                    most_threatening = obs
        
        return most_threatening, min_ttc
    
    def _is_adjacent_lane_clear(
        self, 
        obstacles: List[ObstacleInfo], 
        direction: str,
        ego_speed: float
    ) -> bool:
        """
        Check if adjacent lane is clear for merging using TTC.
        
        Accounts for current ego lateral position and closing velocity.
        """
        # Target lane center relative to EGO (not reference path)
        if direction == 'left':
            target_lane_y = self.config.lane_width  # Left of ego
        else:
            target_lane_y = -self.config.lane_width  # Right of ego
        
        for obs in obstacles:
            # Check ahead and behind
            if obs.x < -self.config.adjacent_lane_check_behind:
                continue
            if obs.x > self.config.adjacent_lane_check_distance:
                continue
            
            # Is obstacle in the target lane?
            lateral_offset = abs(obs.y - target_lane_y)
            
            if lateral_offset < self.config.lane_threshold:
                # Obstacle in target lane - check TTC
                if obs.x > 0:
                    # Ahead of us - check forward TTC
                    _, ttc = self.is_obstacle_threatening(obs, ego_speed)
                    if ttc < self.config.adjacent_lane_ttc_threshold:
                        return False
                else:
                    # Behind us - check if it's catching up
                    # If obstacle is behind (x < 0) and faster than us, it will catch up
                    # closing_vel = obstacle_speed - ego_speed (from their perspective)
                    # If obs.vx > ego_speed, they're faster and will catch us
                    closing_from_behind = obs.vx - ego_speed
                    if closing_from_behind > 1.0:  # Approaching from behind
                        rear_ttc = abs(obs.x) / closing_from_behind
                        if rear_ttc < self.config.adjacent_lane_ttc_threshold:
                            return False
                    # If not closing fast, distance check is enough
                    elif abs(obs.x) < self.config.adjacent_lane_check_behind:
                        return False
        
        return True
    
    def reset(self):
        """Reset to initial state."""
        self.behavior_state = BehaviorState.LANE_FOLLOWING
        self.current_lane = 0
        self.target_lane = 0
        self.current_d = 0.0
        self.current_d_dot = 0.0
        self._lane_change_elapsed = 0.0
        self._prepare_elapsed = 0.0
        self._cooldown_remaining = 0.0
        self._consecutive_clear_count = 0
        self._consecutive_block_count = 0
    
    def get_debug_info(self) -> dict:
        """Get diagnostic information."""
        return {
            'state': self.behavior_state.name,
            'current_lane': self.current_lane,
            'target_lane': self.target_lane,
            'current_d': round(self.current_d, 2),
            'target_d': round(self.target_d, 2),
            'cooldown': round(self._cooldown_remaining, 2),
            'prepare_elapsed': round(self._prepare_elapsed, 2),
            'lane_change_elapsed': round(self._lane_change_elapsed, 2),
        }
