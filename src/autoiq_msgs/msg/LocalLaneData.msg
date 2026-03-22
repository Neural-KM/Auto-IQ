# Structured local lane data with lane information
# Published by mapping, consumed by planning
# Replaces raw boundary paths with pre-computed lane structure

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

# Computed lanes (index 0 = rightmost, toward center)
autoiq_msgs/Lane[] lanes

# Road geometry metadata
float64 road_width_avg          # Average measured road width (meters)
uint8 lane_count                # Number of computed lanes
