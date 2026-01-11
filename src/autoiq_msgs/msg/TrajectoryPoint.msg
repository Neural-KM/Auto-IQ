# Single trajectory waypoint with full kinematic state
#
# Used in Trajectory message for planning → control communication.

# Position in world frame (CARLA coordinates)
float64 x
float64 y

# Heading angle (radians, CARLA convention)
float64 yaw

# Target speed at this point (m/s)
float64 speed

# Curvature at this point (1/m, positive = left turn)
float64 curvature

# Time from trajectory start (seconds)
float64 time_from_start
