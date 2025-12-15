"""
Behavior Planner
================

State machine for high-level behavior decisions.
"""

from enum import Enum, auto
from config.config import CONTROL


class BehaviorState(Enum):
    """Possible behavior states."""
    LANE_FOLLOWING = auto()
    PREPARING_LANE_CHANGE = auto()
    LANE_CHANGING_LEFT = auto()
    LANE_CHANGING_RIGHT = auto()
    EMERGENCY_STOP = auto()


class BehaviorPlanner:
    """
    High-level behavior planning using state machine.
    
    Decides when to follow lane, change lanes, or stop.
    """
    
    def __init__(self, local_planner):
        """
        Initialize behavior planner.
        
        Args:
            local_planner: LocalPlanner instance
        """
        self.local_planner = local_planner
        self.state = BehaviorState.LANE_FOLLOWING
        
        # Lane change state tracking
        self.lane_change_waypoints = []
        self.lane_change_index = 0
        self.original_lane_id = None
        
        # Road context
        self.lane_context = None
    
    def get_state(self):
        """Get current behavior state."""
        return self.state
    
    def get_state_name(self):
        """Get current state as string."""
        return self.state.name
    
    def update(self, obstacle_detection):
        """
        Update behavior state based on perception.
        
        Args:
            obstacle_detection: Result from ObstacleDetector.detect()
            
        Returns:
            Target waypoint for current behavior
        """
        # Update lane context
        self.lane_context = self.local_planner.get_lane_context()
        
        if self.state == BehaviorState.LANE_FOLLOWING:
            return self._handle_lane_following(obstacle_detection)
        
        elif self.state == BehaviorState.PREPARING_LANE_CHANGE:
            return self._handle_preparing_lane_change(obstacle_detection)
        
        elif self.state in [BehaviorState.LANE_CHANGING_LEFT, 
                            BehaviorState.LANE_CHANGING_RIGHT]:
            return self._handle_lane_changing()
        
        elif self.state == BehaviorState.EMERGENCY_STOP:
            return self._handle_emergency_stop()
        
        return self.local_planner.get_target_waypoint()
    
    def _handle_lane_following(self, obstacle_detection):
        """Handle lane following state."""
        if obstacle_detection and obstacle_detection.get('obstacle_detected'):
            distance = obstacle_detection.get('distance', float('inf'))
            position = obstacle_detection.get('position', '')
            
            # Obstacle in our lane and close
            if position == 'center' and distance < CONTROL['obstacle']['lane_change_trigger_distance']:
                self.state = BehaviorState.PREPARING_LANE_CHANGE
                self.original_lane_id = self.local_planner.get_current_waypoint().lane_id
                print(f"[Behavior] Obstacle detected at {distance:.1f}m - preparing lane change")
        
        return self.local_planner.get_target_waypoint()
    
    def _handle_preparing_lane_change(self, obstacle_detection):
        """Handle preparing for lane change."""
        # Determine which direction to change
        can_left = self.local_planner.can_change_lane('left')
        can_right = self.local_planner.can_change_lane('right')
        
        direction = None
        if can_left:
            direction = 'left'
        elif can_right:
            direction = 'right'
        
        if direction:
            # Generate lane change path
            self.lane_change_waypoints = self.local_planner.generate_lane_change_waypoints(direction)
            
            if self.lane_change_waypoints:
                self.lane_change_index = 0
                if direction == 'left':
                    self.state = BehaviorState.LANE_CHANGING_LEFT
                else:
                    self.state = BehaviorState.LANE_CHANGING_RIGHT
                print(f"[Behavior] Executing lane change {direction}")
            else:
                # Can't generate path, emergency stop
                self.state = BehaviorState.EMERGENCY_STOP
                print("[Behavior] Cannot change lanes - emergency stop")
        else:
            # No lane available, emergency stop
            self.state = BehaviorState.EMERGENCY_STOP
            print("[Behavior] No lane available - emergency stop")
        
        return self.local_planner.get_target_waypoint()
    
    def _handle_lane_changing(self):
        """Handle lane change execution."""
        if self.lane_change_index >= len(self.lane_change_waypoints):
            # Lane change complete - stay in new lane, don't auto-return
            self.state = BehaviorState.LANE_FOLLOWING
            self.lane_change_waypoints = []
            self.lane_change_index = 0
            print("[Behavior] Lane change complete - continuing in new lane")
            return self.local_planner.get_target_waypoint()
        
        # Get current target in lane change sequence
        target_wp = self.lane_change_waypoints[self.lane_change_index]
        
        # Check if we've reached this waypoint
        vehicle_location = self.local_planner.vehicle.get_location()
        distance = vehicle_location.distance(target_wp.transform.location)
        
        # Advance to next waypoint when close enough
        # Use smaller threshold for smoother path following
        if distance < 2.5:
            self.lane_change_index += 1
            
            # Skip waypoints that are behind the vehicle (already passed)
            while self.lane_change_index < len(self.lane_change_waypoints):
                next_wp = self.lane_change_waypoints[self.lane_change_index]
                next_dist = vehicle_location.distance(next_wp.transform.location)
                if next_dist > 2.0:
                    break
                self.lane_change_index += 1
        
        # Return current target or next available
        if self.lane_change_index < len(self.lane_change_waypoints):
            return self.lane_change_waypoints[self.lane_change_index]
        else:
            return target_wp
    
    def _handle_emergency_stop(self):
        """Handle emergency stop state."""
        # Stay stopped until manually reset
        return self.local_planner.get_target_waypoint()
    
    def reset(self):
        """Reset to lane following state."""
        self.state = BehaviorState.LANE_FOLLOWING
        self.lane_change_waypoints = []
        self.lane_change_index = 0
        print("[Behavior] Reset to lane following")
