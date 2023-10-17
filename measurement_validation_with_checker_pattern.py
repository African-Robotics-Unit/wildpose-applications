import os
import glob
import numpy as np
import cv2
import open3d as o3d

from utils.file_loader import load_pcd, load_camera_parameters


ECAL_FOLDER = '/Volumes/Expansion/Calibration/ecal_meas/2023-02-04_15-20-34.496_wildpose_v1.1'
# ECAL_FOLDER = '/Volumes/Expansion/Calibration/ecal_meas/2023-02-04_15-27-37.535_wildpose_v1.1'
CAMERA_PARAM_FILENAME = 'manual_calibration.json'
FRAME_START_INDEX = 0
FRAME_END_INDEX = 10
PATTERN_SIZE = (7, 10)  # for example


def find_checkerboard_corners(image_path, pattern_size):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(
        gray, pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    return corners if ret else None

def project_3d_points(points, fx, fy, cx, cy, rot_mat, translation):
    transformed_points = points.dot(rot_mat.T) + translation
    x, y, z = transformed_points[:, 0], transformed_points[:, 1], transformed_points[:, 2]
    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    return np.column_stack((u, v))


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

    for image_path in rgb_fpaths:
        # get the checker pattern points from the image
        corners = find_checkerboard_corners(image_path, pattern_size=PATTERN_SIZE)
        if corners is None:
            print(f"Error: failed to find corners in {image_path}")
            continue

        # Assuming merged_pcd contains the points
        points = np.asarray(merged_pcd.points)

        # project the 3D points onto the image
        projected_points = project_3d_points(points, fx, fy, cx, cy, rot_mat, translation)

        # get 3D point indices corresponding with checker pattern
        indices = []  # This depends on your exact correspondence method
        # Populate 'indices' here with the indices of 3D points that match the 2D checkerboard corners

        # show the 3D lengths between each pattern points
        # Assuming indices contain the indices of the 3D points that match the 2D checkerboard corners
        for i, idx1 in enumerate(indices):
            for j, idx2 in enumerate(indices[i + 1:]):
                distance = np.linalg.norm(points[idx1] - points[idx2])
                print(f"Distance between point {idx1} and {idx2}: {distance}")



if __name__ == '__main__':
    main()
