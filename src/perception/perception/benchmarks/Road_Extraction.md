'''
Inference time:::

Initializing RoadExtractor (road_grayscale_id=90)...
[TIMING] Image load + BGR2RGB: 5.173 ms
Processing: 20251208_165834_584_spawn0.jpg (1280x720)
[TIMING] Segmentation map load: 4.339 ms
Loaded segmentation map from: /home/autoiq/nakul/Auto-IQ/data/rosbag_images_15jan/inference/20251208_165834_584_spawn0_inference_b2_segmented.png
DEBUG: Unique pixel values found in segmentation map: [  7  46  70  84  90 108 117 119 153 164 194]
DEBUG: Using road_grayscale_id=90
[TIMING] extract_road_mask: 0.223 ms
DEBUG: Extracted road mask has 387119 pixels (42.01% of image)
Extracted road mask from segmentation map
Pipeline: Road Mask → Separate Road → IPM → Lane Detection
IPM range: X=[3.0, 30.0]m, Y=[-10.0, 10.0]m
Resolution: 30.0 pixels/meter
Step 1: Converting original image to IPM...
[TIMING] convert_to_ipm: 0.957 ms
IPM size: 600x810 pixels (auto-calculated for equal scale)
[TIMING] warpPerspective (road mask to IPM): 0.378 ms
DEBUG: Road pixels in IPM view: 360458
Step 2: Detecting lane markings in IPM...
[TIMING] _extract_lanes_from_ipm: 13.009 ms
[TIMING] bitwise_and (road_only extraction): 0.379 ms
[TIMING] === TOTAL extract_lanes_ipm: 14.784 ms ===
[TIMING] === FPS (extract_lanes_ipm): 67.6 ===
Final IPM size: 600x810 pixels
'''
