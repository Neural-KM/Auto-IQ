"""
World State (Blackboard Pattern).

Central repository for all simulation state.
Modules read from and write to this shared state.
"""

import threading
import copy
from dataclasses import dataclass, field
from typing import List, Optional

from state.obstacle import Obstacle
from state.lane import LaneDetection
from state.vehicle import VehicleState

@dataclass
class WorldState:
    """Thread-safe container for all simulation state."""
    
    # Lock for thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    
    # 1. Ground Truth (from FakePerception)
    ground_truth_obstacles: List[Obstacle] = field(default_factory=list)
    
    # 2. Perception Output (from Real Perception)
    perceived_obstacles: List[Obstacle] = field(default_factory=list)
    
    # 3. Lane Detection
    lane_detection: Optional[LaneDetection] = None
    
    # 4. Vehicle State
    vehicle_state: Optional[VehicleState] = None
    
    # 5. Planning (Trajectory)
    # Storing as list of (x, y) or (x, y, speed) tuples for now
    trajectory: List[tuple] = field(default_factory=list)
    
    def update_ground_truth(self, obstacles: List[Obstacle]):
        with self._lock:
            self.ground_truth_obstacles = obstacles

    def update_perception(self, obstacles: List[Obstacle]):
        with self._lock:
            self.perceived_obstacles = obstacles
            
    def update_lane_detection(self, detection: LaneDetection):
        with self._lock:
            self.lane_detection = detection
            
    def update_vehicle_state(self, state: VehicleState):
        with self._lock:
            self.vehicle_state = state
            
    def update_trajectory(self, trajectory: List[tuple]):
        with self._lock:
            self.trajectory = trajectory

    def get_snapshot(self) -> 'WorldState':
        """Returns a consistent deep copy of the state for visualization/logging."""
        with self._lock:
            # Create a shallow copy of the object, but deep copy the mutable lists/objects
            # to ensure thread safety for the consumer
            snapshot = WorldState()
            snapshot.ground_truth_obstacles = list(self.ground_truth_obstacles)
            snapshot.perceived_obstacles = list(self.perceived_obstacles)
            snapshot.lane_detection = copy.deepcopy(self.lane_detection)
            snapshot.vehicle_state = copy.deepcopy(self.vehicle_state)
            snapshot.trajectory = list(self.trajectory)
            return snapshot

# Global instance (optional, but useful for singleton access if preferred)
# world_state = WorldState()
