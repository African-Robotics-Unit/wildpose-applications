import cv2
import numpy as np
import open3d as o3d
import quaternion
import glob
import os
import json
from tqdm import tqdm

from projection_functions import extract_rgb_from_image_pure
from utils.camera import make_intrinsic_mat, make_extrinsic_mat


CONFIG = {
    "scene_dir": "data/giraffe_stand",
    "pcd_dir": "data/giraffe_stand/lidar",
    "merged_pcd_dir": "data/giraffe_stand/merged_pcd",
}
MERGE_SIZE = 5


def main():
    # arguments
    data_dir = CONFIG['scene_dir']
    lidar_dir = CONFIG['pcd_dir']
    output_dir = CONFIG['merged_pcd_dir']
    os.makedirs(output_dir, exist_ok=True)

    # load the texture image
    lidar_list = sorted(glob.glob(os.path.join(lidar_dir, '*.pcd')))

    # accumulate all the point cloud
    for i in range(len(lidar_list) - MERGE_SIZE + 1):
        accumulated_points = None
        j = i + MERGE_SIZE
        for pcd_fpath in lidar_list[i:j]:
            # NOTE: you need to write the parser of pcd files if you get the intensity.
            pcd_in_lidar = o3d.io.read_point_cloud(pcd_fpath)
            pcd_points = np.asarray(pcd_in_lidar.points)  # [N, 3]

            if accumulated_points is None:
                accumulated_points = pcd_points
            else:
                accumulated_points = np.vstack((accumulated_points, pcd_points))

        # make the new pcd file
        output_pcd = o3d.geometry.PointCloud()
        output_pcd.points = o3d.utility.Vector3dVector(accumulated_points)

        o3d.io.write_point_cloud(
            os.path.join(
                output_dir,
                str(i).zfill(3) + '_' + str(j-1).zfill(3) + '.pcd'
            ),
            output_pcd)


if __name__ == '__main__':
    main()