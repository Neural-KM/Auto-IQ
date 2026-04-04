# Lane detection result from polynomial fitting on semantic segmentation.
#
# Fits are in BEV (bird's eye view) pixel coordinates with model:
#   x = a*y^2 + b*y + c

std_msgs/Header header

# Detection validity
bool valid
float32 confidence

# Number of fitted lanes and flattened polynomial coefficients.
# lane_count = N
# poly_coeffs_flat = [a0,b0,c0, a1,b1,c1, ... aN-1,bN-1,cN-1]
int32 lane_count
float64[] poly_coeffs_flat

# BEV parameters for interpreting polynomial coefficients.
# In BEV image coordinates: image X is lateral (vehicle Y), image Y is forward (vehicle X).
int32 bev_width
int32 bev_height
float64 bev_x_min
float64 bev_x_max
float64 bev_y_min
float64 bev_y_max
