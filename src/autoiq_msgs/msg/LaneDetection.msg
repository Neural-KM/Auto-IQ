# Lane detection result from polynomial fitting on semantic segmentation
#
# Polynomial fits are 2nd-order: x = a*y^2 + b*y + c
# Coefficients are in BEV (bird's eye view) pixel space.

std_msgs/Header header

# Detection validity
bool valid
float32 confidence            # 0.0 - 1.0

# Left lane polynomial coefficients [a, b, c]  (empty if not detected)
float64[] left_poly_coeffs
# Right lane polynomial coefficients [a, b, c] (empty if not detected)
float64[] right_poly_coeffs
# Center lane polynomial coefficients [a, b, c] (derived from left+right)
float64[] center_poly_coeffs

# Lane geometry in image pixels (original image space)
int32 lane_center             # Lane center x-pixel
int32 left_lane_x             # Left lane boundary x-pixel  (-1 if unknown)
int32 right_lane_x            # Right lane boundary x-pixel (-1 if unknown)
int32 lane_width_px           # Lane width in pixels        (-1 if unknown)

# Real-world lane width (meters), derived from polynomial fits in BEV
float64 lane_width_m          # -1.0 if unknown
