# Planned trajectory with behavior state
#
# Published by planning, consumed by control.
# Contains a sequence of waypoints with kinematic information.

std_msgs/Header header

# Trajectory waypoints
autoiq_msgs/TrajectoryPoint[] points

# Current behavior state
uint8 behavior_state

# Behavior state constants
uint8 LANE_FOLLOWING = 0
uint8 PREPARING_LANE_CHANGE = 1
uint8 LANE_CHANGING = 2
uint8 EMERGENCY_STOP = 3
uint8 DESTINATION_REACHED = 4

# Target lane index (0 = current, +1 = left, -1 = right)
int8 target_lane
