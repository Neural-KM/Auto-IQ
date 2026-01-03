# Vehicle control command message for CARLA ego vehicle
#
# This message is used to send control commands from the control node
# to the CARLA bridge.

std_msgs/Header header

# Throttle command [0.0, 1.0]
float32 throttle

# Brake command [0.0, 1.0]  
float32 brake

# Steering command [-1.0, 1.0] (left positive, right negative)
float32 steer

# Hand brake engaged
bool hand_brake

# Reverse gear engaged
bool reverse

# Manual gear shift mode
bool manual_gear_shift

# Current gear (only used if manual_gear_shift is true)
int32 gear
