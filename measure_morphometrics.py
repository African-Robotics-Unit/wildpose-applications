import os
import glob
import numpy as np
import cv2
import open3d as o3d
from itertools import combinations

from utils.file_loader import load_pcd, load_camera_parameters
from utils import camera as camera_utils
from projection_functions import closest_point


ECAL_FOLDER = '/Users/ikuta/Documents/Projects/wildpose-self-calibrator/data/giraffe_stand'
CAMERA_PARAM_FILENAME = 'manual_calibration.json'
FRAME_START_INDEX = 0
FRAME_END_INDEX = 50
PATTERN_SIZE = (7, 10)  # for example
KEYPOINTS = {
    'nose': [838, 159],
    'r_eye': [774, 112],
    'neck': [697, 453],
    'tail': [540, 568],
}
COMBINATIONS = [
    ['nose', 'r_eye'],
    ['r_eye', 'neck'],
    ['neck', 'tail'],
]


def main():
    # load the data
    rgb_fpaths = sorted(glob.glob(os.path.join(ECAL_FOLDER, 'merged_rgb', '*.jpeg')))
    pcd_fpaths = sorted(glob.glob(os.path.join(ECAL_FOLDER, 'merged_pcd', '*.pcd')))

    # load the camera parameters
    calib_fpath = os.path.join(ECAL_FOLDER, CAMERA_PARAM_FILENAME)
    fx, fy, cx, cy, rot_mat, translation = load_camera_parameters(calib_fpath)
    intrinsic_mat = camera_utils.make_intrinsic_mat(fx, fy, cx, cy)
    extrinsic_mat = camera_utils.make_extrinsic_mat(rot_mat, translation)

    for pcd_fpath in pcd_fpaths:
        pcd_o3d = load_pcd(pcd_fpath, mode='open3d')
        pts_in_ptc = np.array(pcd_o3d.points)

        # project the point cloud to camera and its image sensor
        pts_in_cam = camera_utils.lidar2cam_projection(pts_in_ptc, extrinsic_mat)
        pts_in_img = camera_utils.cam2image_projection(pts_in_cam, intrinsic_mat)
        pts_in_img = pts_in_img.T[:, :-1]   # [N, 3]

        # get 3D point indices corresponding with checker pattern
        for kp_a, kp_b in COMBINATIONS:
            _, pt_idx_a = closest_point(KEYPOINTS[kp_a], pts_in_img[:, :2])
            _, pt_idx_b = closest_point(KEYPOINTS[kp_b], pts_in_img[:, :2])
            pt3d_a = pts_in_ptc[pt_idx_a, :]
            pt3d_b = pts_in_ptc[pt_idx_b, :]
            distance = np.linalg.norm(pt3d_a - pt3d_b)
            print(f'{distance}m between {kp_a} and {kp_b}')

        # break   # DEBUG


if __name__ == '__main__':
    main()
