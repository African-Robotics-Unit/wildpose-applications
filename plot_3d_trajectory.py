import os
import re
import glob
import pandas as pd
import numpy as np
import json
import cv2
from tqdm import tqdm
import scipy.ndimage

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import scienceplots

from utils.file_loader import load_camera_parameters, load_rgb_img, load_pcd
from utils.camera import make_intrinsic_mat, make_extrinsic_mat
from utils.projection import lidar2cam_projection, cam2image_projection
from utils.format_conversion import get_timestamp_from_img_fpath
from projection_functions import extract_rgb_from_image
from config import COLORS, colors_indices

from projection_functions import closest_point


# plt.style.use(['science', 'nature', 'no-latex'])
# figure(figsize=(10, 6))
matplotlib.rc('font', family='sans-serif')
matplotlib.rc('font', serif='Arial')
plt.rcParams.update({
    'legend.frameon': False,
    "pdf.fonttype": 42,
})


IMG_WIDTH, IMG_HEIGHT = 1280,720


def get_timestamp_from_pcd_fpath(fpath: str) -> float:
    fname = os.path.splitext(os.path.basename(fpath))[0]
    # get timestamp
    fname = fname.split('_')
    msg_id = '_'.join(fname[:-2])
    timestamp = float(fname[-2] + '.' + fname[-1])

    return timestamp


def median_filter_3d_positions(dfs, filter_size=3):
    filtered_dfs = {}
    for key, df in dfs.items():
        filtered_df = df.copy()
        filtered_df['x'] = scipy.ndimage.median_filter(
            df['x'], size=filter_size)
        filtered_df['y'] = scipy.ndimage.median_filter(
            df['y'], size=filter_size)
        filtered_df['z'] = scipy.ndimage.median_filter(
            df['z'], size=filter_size)
        filtered_dfs[key] = filtered_df
    return filtered_dfs


def get_2D_gt(gt_json_path):
    gt_json = json.load(open(gt_json_path, 'r'))
    img_dict = {}

    annotations = gt_json['annotations']
    images = gt_json['images']

    for anno in annotations:
        bbox = anno['bbox']
        img_id = anno['image_id']
        obj_id = anno['category_id']
        img_filename = os.path.basename(images[img_id]['file_name'])

        if img_filename not in img_dict:
            img_dict[img_filename] = []
        img_dict[img_filename].append([bbox,obj_id])
    return img_dict


def erode_mask(mask, kernel_size=(5,5), iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    eroded_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=iterations)
    return eroded_mask


def calculate_precision(positions_3d):
    precision_results = {}
    all_deviations = []

    for id, positions in positions_3d.items():
        if len(positions) > 1:
            # Extracting just the x, y, z coordinates
            coordinates = np.array([
                [pos['x'], pos['z']]
                for i, pos in positions.iterrows()
            ])

            # Calculate mean position
            mean_position = np.mean(coordinates, axis=0)

            # Calculate deviations from the mean
            deviations = np.linalg.norm(coordinates - mean_position, axis=1)
            all_deviations += deviations.tolist()

            # Calculate standard deviation (precision)
            precision = np.std(deviations)

            precision_results[id] = precision

    precision_result_total = np.std(all_deviations)

    return precision_results, precision_result_total


def main():
    data_dir = '/Users/ikuta/Documents/Projects/wildpose-applications/data/springbok_herd2/'
    lidar_dir = os.path.join(data_dir, 'lidar')
    rgb_dir = os.path.join(data_dir, 'sync_rgb')
    mask_dir = os.path.join(data_dir, 'masks2_fixed')
    calib_fpath = os.path.join(data_dir, 'manual_calibration.json')

    # load data
    img_fpaths = sorted(glob.glob(os.path.join(rgb_dir, '*.jpeg')))
    pcd_fpaths = sorted(glob.glob(os.path.join(lidar_dir, '*.pcd')))
    assert len(img_fpaths) == len(pcd_fpaths)
    n_frame = len(img_fpaths)
    mask_fpaths = [os.path.join(mask_dir, '{0}.npy'.format(i)) for i in range(n_frame)]
    mask_id_fpaths = [os.path.join(mask_dir, '{0}_obj_ids.npy'.format(i)) for i in range(n_frame)]
    fx, fy, cx, cy, rot_mat, translation = load_camera_parameters(calib_fpath)
    intrinsic = make_intrinsic_mat(fx, fy, cx, cy)
    extrinsic = make_extrinsic_mat(rot_mat, translation)

    # collect the 3D positions with Segment Anything Model
    timestamp0 = get_timestamp_from_img_fpath(img_fpaths[0])
    positions_3d = {}
    for img_fpath, pcd_fpath, mask_fpath, mask_id_fpath in tqdm(zip(img_fpaths, pcd_fpaths, mask_fpaths, mask_id_fpaths)):
        # load the frame
        rgb_img = load_rgb_img(img_fpath)
        pcd_open3d = load_pcd(pcd_fpath, mode='open3d')
        pts_in_lidar = np.asarray(pcd_open3d.points)
        seg_mask = np.load(mask_fpath)  # [n_id, 1, H, W]
        obj_ids = np.load(mask_id_fpath)
        timestamp = get_timestamp_from_img_fpath(img_fpath)
        img_key = os.path.basename(img_fpath).replace('.jpeg', '_3.jpeg')

        # reprojection
        pcd_in_cam = lidar2cam_projection(pts_in_lidar, extrinsic)
        pcd_in_img = cam2image_projection(pcd_in_cam, intrinsic)
        pcd_in_cam = pcd_in_cam.T[:, :-1]
        pcd_in_img = pcd_in_img.T[:, :-1]

        # # eroded_2d_mask -> median 3D point
        # # erode the segmentation mask to reduce the error of estimated 3d positions
        # for i in range(seg_mask.shape[0]):
        #     # seg_mask.shape should be (n, 1, H, W)
        #     seg_mask[i, 0, :, :] = erode_mask(seg_mask[i, 0, :, :], kernel_size=(5,5), iterations=4)

        # colors, valid_mask, obj_points, obj_mask_from_color = extract_rgb_from_image(
        #     pcd_in_img, pcd_in_cam, rgb_img, seg_mask, obj_ids,
        #     width=IMG_WIDTH, height=IMG_HEIGHT
        # )

        # # store the position data
        # for id, points in obj_points.items():
        #     position_3d = np.median(points, axis=0)
        #     if id not in positions_3d.keys():
        #         positions_3d[id] = []
        #     positions_3d[id].append([timestamp] + position_3d.tolist())

        # median_2d -> 3d point
        n_id = seg_mask.shape[0]
        assert n_id == len(obj_ids)
        for idx, obj_id in enumerate(obj_ids):
            mask = seg_mask[idx, 0]
            mask_ys, mask_xs = np.where(mask)
            target_2d_pt = np.array([
                np.median(mask_xs),
                np.median(mask_ys),
            ])
            _, pt_idx = closest_point(target_2d_pt, pcd_in_img[:, :2])
            pt3d = pcd_in_cam[pt_idx, :]
            if obj_id not in positions_3d.keys():
                positions_3d[obj_id] = []
            positions_3d[obj_id].append([timestamp] + pt3d.tolist())

    # array to dataframe
    dfs = {}
    for key in positions_3d.keys():
        dfs[key] = pd.DataFrame(
            positions_3d[key],
            columns =['time', 'x', 'y', 'z']
        )

    # filter the positions
    # dfs = median_filter_3d_positions(dfs, filter_size=5)

    # calculate the precision with stationary individuals
    # ID 4&8
    # Frame: 39--99
    precisions, total_precision = calculate_precision({
        4: dfs[4][39:99+1],
        8: dfs[8][39:99+1],
    })
    print(precisions)
    print(f'total precision: {total_precision}')

    # plot the data
    ax = plt.figure(layout='constrained').add_subplot(projection='3d')
    for k, v in dfs.items():
        # define color
        rgb = COLORS[colors_indices[int(k)]]['color']
        norm_rgb = [x / 255. for x in rgb]
        # plot
        ax.plot(
            v['x'], v['z'], v['time'] - timestamp0,
            linewidth=1,
            label=int(k), color=norm_rgb
        )
    # for k, v in dfs.items():
    #     if k=='06' or k=='07':
    #         # define color
    #         rgb = COLORS[colors_indices[int(k)]]['color']
    #         norm_rgb = [x / 255. for x in rgb]
    #         # plot
    #         ax.plot(
    #             v['x'], v['z'], time,
    #             linewidth=3 if k=='06' or k=='07' else 1,
    #             label=k, color=norm_rgb
    #         )
    ax.legend()
    # ax.set_box_aspect([1.0, 1.0, 1.0])
    ax.set_xticks(np.arange(-6, 5, 2), minor=False)
    ax.set_yticks(np.arange(110, 170, 10), minor=False)
    ax.set_zticks(np.arange(0, 30, 5), minor=False)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)')
    ax.set_zlabel('Time (s)')

    plt.show()


if __name__ == '__main__':
    main()