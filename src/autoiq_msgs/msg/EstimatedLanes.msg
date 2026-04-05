# Estimated lanes from road boundary detection
# Published by mapping, consumed by planning and visualization
#
# This represents ESTIMATED lane structure, not detected lane markings.
# Uses a priori knowledge (lane count, road width) to divide detected
# road boundaries into individual lanes.

std_msgs/Header header

# Overall validity
bool valid
uint8 status_code

# Status codes
uint8 STATUS_OK = 0
uint8 STATUS_NO_BOUNDARIES = 1
uint8 STATUS_INVALID_WIDTH = 2
uint8 STATUS_INSUFFICIENT_POINTS = 3

# Road boundaries in base_link frame (ordered by increasing x = forward)
# Left = higher y (divider side), Right = lower y (wall/kerb side)
geometry_msgs/Point[] left_boundary
geometry_msgs/Point[] right_boundary

# Estimated lanes (index 0 = rightmost, toward center)
autoiq_msgs/EstimatedLane[] lanes

# Road geometry metadata
float64 road_width_avg          # Average measured road width (meters)
uint8 lane_count                # Number of estimated lanes
