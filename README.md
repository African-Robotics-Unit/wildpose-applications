# WildPose Applications

## Lion Lug Size

0. change `config.hjson`
1. Make the bounding boxes of the target animals
2. Predict the masks of each individuals with Segment Anything (`segment_anything.py`)
3. Make the labeled, textured point cloud data (`vis_lidar_rgb_cam_lion.py`)
4. Estimate the transition of the body size (`body_size_estimator.py`)

## The data format of textured point cloud

There are two types of pcd files in `textured_pcds` directory.
1. `*.pcd` files contain point locations and colors.
2. `*_mask.pcd` files contain point locations and masks for lions.
For instance, points having rgb values `1` are lion `1`(the left in image) and `2` are lion `2`( the right one in image)