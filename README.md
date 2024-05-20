# WildPose Applications

## Respiration of Lion

1. Make the bounding boxes of the target animals
2. Predict the masks of each individuals with Segment Anything (`segment_anything.py`)
<!-- 3. Make the labeled, textured point cloud data (`vis_lidar_rgb_cam_lion.py`) -->
3. Estimate the transition of the body size (`body_size_estimator.py`)

## The files in data directory

### `textured_pcds/`

1. `*.pcd` files contain point locations and colors.
2. `*_mask.pcd` files contain point locations and masks for lions.
For instance, points having rgb values `1` are lion `1`(the left in image) and `2` are lion `2`( the right one in image)

### `masks/`

This directory has the output from Segment Anything.

1. `*.npy` files contain the mask numpy-array with the shape `(I, 1, H, W)`, where `I` is the number of object IDs and `H` and `W` show the image size.