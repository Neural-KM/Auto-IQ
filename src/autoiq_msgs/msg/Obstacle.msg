# Single obstacle detection
#
# Represents a detected or ground-truth obstacle in the environment.
# Used for perception → planning communication.

std_msgs/Header header

# Unique identifier for tracking
uint32 id

# Object classification: "vehicle", "pedestrian", "cyclist", "static"
string type

# Pose in world frame (CARLA coordinates)
# position.x, position.y, position.z = center of object
# orientation = yaw as quaternion
geometry_msgs/Pose pose

# Velocity in world frame (m/s)
# x = forward, y = lateral, z = vertical
geometry_msgs/Vector3 velocity

# Bounding box dimensions (meters)
# x = length (front-to-back)
# y = width (side-to-side)  
# z = height
geometry_msgs/Vector3 size

# Detection confidence [0.0, 1.0]
# 1.0 for ground truth
float32 confidence
