import os
import glob
import numpy as np
import pandas as pd

from utils.file_loader import load_pcd, load_camera_parameters
from utils import camera as camera_utils
from projection_functions import closest_point


ECAL_FOLDER = '/Users/ikuta/Documents/Projects/wildpose-self-calibrator/data/giraffe_stand'
CAMERA_PARAM_FILENAME = 'manual_calibration.json'
FRAME_START_INDEX = 0
FRAME_END_INDEX = 50
PATTERN_SIZE = (7, 10)  # for example
KEYPOINTS = {
    '000_004.jpeg': {
        'nose': [838, 159],
        'r_eye': [774, 112],
        'neck': [697, 453],
        'tail': [540, 568],
    },
    '001_005.jpeg': {
        'nose': [843, 162],
        'r_eye': [782, 115],
        'neck': [696, 458],
        'tail': [536, 580],
    },
    '002_006.jpeg': {
        'nose': [840, 160],
        'r_eye': [778, 115],
        'neck': [480, 472],
        'tail': [528, 575],
    },
    '003_007.jpeg': {
        'nose': [0, 0],
        'r_eye': [0, 0],
        'neck': [0, 0],
        'tail': [0, 0],
    },
    '004_008.jpeg': {
        'nose': [0, 0],
        'r_eye': [0, 0],
        'neck': [0, 0],
        'tail': [0, 0],
    },
    '005_009.jpeg': {
        'nose': [0, 0],
        'r_eye': [0, 0],
        'neck': [0, 0],
        'tail': [0, 0],
    },
    '006_010.jpeg': {
        'nose': [0, 0],
        'r_eye': [0, 0],
        'neck': [0, 0],
        'tail': [0, 0],
    },
    '007_011.jpeg': {
        'nose': [0, 0],
        'r_eye': [0, 0],
        'neck': [0, 0],
        'tail': [0, 0],
    },
    '008_012.jpeg': {
        'nose': [0, 0],
        'r_eye': [0, 0],
        'neck': [0, 0],
        'tail': [0, 0],
    },
    '009_013.jpeg': {
        'nose': [0, 0],
        'r_eye': [0, 0],
        'neck': [0, 0],
        'tail': [0, 0],
    },
    '010_014.jpeg': {
        'nose': [0, 0],
        'r_eye': [0, 0],
        'neck': [0, 0],
        'tail': [0, 0],
    },
    '011_015.jpeg': {
        'nose': [0, 0],
        'r_eye': [0, 0],
        'neck': [0, 0],
        'tail': [0, 0],
    },
    '012_016.jpeg': {
        'nose': [0, 0],
        'r_eye': [0, 0],
        'neck': [0, 0],
        'tail': [0, 0],
    },
    '013_017.jpeg': {
        'nose': [0, 0],
        'r_eye': [0, 0],
        'neck': [0, 0],
        'tail': [0, 0],
    },
    '014_018.jpeg': {
        'nose': [0, 0],
        'r_eye': [0, 0],
        'neck': [0, 0],
        'tail': [0, 0],
    },
    '015_019.jpeg': {
        'nose': [0, 0],
        'r_eye': [0, 0],
        'neck': [0, 0],
        'tail': [0, 0],
    },
    '016_020.jpeg': {
        'nose': [0, 0],
        'r_eye': [0, 0],
        'neck': [0, 0],
        'tail': [0, 0],
    },
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

    result = []
    for rgb_fpath, pcd_fpath in zip(rgb_fpaths, pcd_fpaths):
        pcd_o3d = load_pcd(pcd_fpath, mode='open3d')
        pts_in_ptc = np.array(pcd_o3d.points)

        # load keypoints
        key = os.path.basename(rgb_fpath)
        keypoints = KEYPOINTS[key]

        # project the point cloud to camera and its image sensor
        pts_in_cam = camera_utils.lidar2cam_projection(pts_in_ptc, extrinsic_mat)
        pts_in_img = camera_utils.cam2image_projection(pts_in_cam, intrinsic_mat)
        pts_in_img = pts_in_img.T[:, :-1]   # [N, 3]

        # get 3D point indices corresponding with checker pattern
        result_row = []
        for kp_a, kp_b in COMBINATIONS:
            _, pt_idx_a = closest_point(keypoints[kp_a], pts_in_img[:, :2])
            _, pt_idx_b = closest_point(keypoints[kp_b], pts_in_img[:, :2])
            pt3d_a = pts_in_ptc[pt_idx_a, :]
            pt3d_b = pts_in_ptc[pt_idx_b, :]
            distance = np.linalg.norm(pt3d_a - pt3d_b)
            result_row.append(distance)
        result.append(result_row)

    df = pd.DataFrame(
        result,
        columns =[f'{a}–{b}' for a, b in COMBINATIONS]
    )
    df.to_csv('results/output.csv', encoding='utf-8')
    print(df)


if __name__ == '__main__':
    main()