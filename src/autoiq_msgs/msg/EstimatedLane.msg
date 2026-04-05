# Single estimated lane within EstimatedLanes message
# Part of EstimatedLanes message for mapping -> planning communication

# Lane index: 0 = rightmost (closest to wall/kerb), higher = toward center
uint8 lane_index

# Lane boundaries in base_link frame (ordered by increasing x = forward)
geometry_msgs/Point[] left_boundary
geometry_msgs/Point[] right_boundary

# Pre-computed lane centerline in base_link frame
geometry_msgs/Point[] centerline

# Lane width statistics (meters)
float64 width_avg

# Validity flags
bool is_valid                   # True if lane meets width requirements
bool is_partial                 # True if computed from single boundary (fallback)
