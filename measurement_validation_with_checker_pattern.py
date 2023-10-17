import os
import glob
import json
import open3d as o3d

from utils.file_loader import load_config_file, load_pcd, load_camera_parameters


ECAL_FOLDER = '/Volumes/Expansion/Calibration/ecal_meas/2023-02-04_15-20-34.496_wildpose_v1.1'
CAMERA_PARAM_FILENAME = 'manual_calibration.json'
FRAME_START_INDEX = 0
FRAME_END_INDEX = 10


def main():
    # load the data
    all_rgb_fpaths = sorted(glob.glob(os.path.join(ECAL_FOLDER, 'sync_rgb', '*.jpeg')))
    all_pcd_fpaths = sorted(glob.glob(os.path.join(ECAL_FOLDER, 'lidar', '*.pcd')))
    assert len(all_rgb_fpaths) == len(all_pcd_fpaths)

    rgb_fpaths = all_rgb_fpaths[FRAME_START_INDEX:FRAME_END_INDEX]
    pcd_fpaths = all_pcd_fpaths[FRAME_START_INDEX:FRAME_END_INDEX]

    merged_pcd = o3d.geometry.PointCloud()
    for pcd_fpath in pcd_fpaths:
        merged_pcd += load_pcd(pcd_fpath, mode='open3d')

    # load the camera parameters
    calib_fpath = os.path.join(ECAL_FOLDER, CAMERA_PARAM_FILENAME)
    fx, fy, cx, cy, rot_mat, translation = load_camera_parameters(calib_fpath)

    # get the checker pattern points from the image

    # project the 3D points onto the image

    # get 3D point indices corresponding with checker pattern

    # show the 3D lengths
    pass


if __name__ == '__main__':
    main()
