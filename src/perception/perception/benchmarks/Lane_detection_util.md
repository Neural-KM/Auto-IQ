'''
Time of each step (total: 13.067 ms | ~76.5 FPS):

[TIMING] bitwise_and (extract road ROI): 0.529 ms
[TIMING] cvtColor (RGB2GRAY): 0.107 ms
[TIMING] CLAHE: 0.315 ms
[TIMING] Fill non-road areas: 0.711 ms
[TIMING] GaussianBlur (light + medium): 0.130 ms
[TIMING] adaptiveThreshold (small, block=11): 1.416 ms
[TIMING] adaptiveThreshold (medium, block=21): 1.518 ms
[TIMING] adaptiveThreshold (large, block=31): 2.071 ms
[TIMING] Otsu threshold: 0.236 ms
[TIMING] Top-hat transform: 0.318 ms
[TIMING] Top-hat thresholding: 1.639 ms
[TIMING] Weighted voting combination: 1.599 ms
[TIMING] Morphological cleanup (open + close): 0.218 ms
[TIMING] Connected components + blob removal: 2.260 ms
'''
