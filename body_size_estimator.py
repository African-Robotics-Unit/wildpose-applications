import os
import re
import json
import cv2
import glob
from typing import Any
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import scipy as sp

from utils.file_loader import load_config_file, load_pcd, load_camera_parameters
from utils.format_conversion import get_timestamp_from_pcd_fpath
from utils import camera as camera_utils
import projection_functions


def get_bbox_labels(gt_json_path):
    gt_json = json.load(open(gt_json_path, 'r'))
    img_dict = {}

    annotations = gt_json['annotations']
    images = gt_json['images']

    for idx in range(len(annotations)):

        anno = annotations[idx]
        bbox = anno['bbox']
        img_id = anno['image_id']
        obj_id = anno['category_id']
        img_filename = os.path.basename(images[img_id]['file_name'])

        if img_filename not in img_dict:
            img_dict[img_filename] = []
        img_dict[img_filename].append([bbox, obj_id])
    return img_dict


def plane2point_distance(plane, point):
    pt_vec = np.ones(4)
    pt_vec[:3] = point
    plane = np.array(plane)
    return np.abs(plane @ pt_vec) / np.linalg.norm(pt_vec[:3])


class PointCloudPainter:
    def __init__(self) -> None:
        pass

    def __call__(self, index: int, *args: Any, **kwds: Any) -> Any:
        pass


class KeyEvent:
    def __init__(self,
                 img_fpaths, pcd_fpaths, mask_fpaths,
                 labels,
                 timestamps,
                 bbox_dict,
                 img_data,
                 intrinsic_mat, extrinsic_mat,
                 pcd_mode,
                 init_geometry=None):
        self.img_fpaths = img_fpaths
        self.pcd_fpaths = pcd_fpaths
        self.mask_fpaths = mask_fpaths
        self.labels = labels
        self.timestamps = timestamps
        self.idx = 0
        self.pcd_mode = pcd_mode
        self.current_pcd = init_geometry
        self.bbox_dict = bbox_dict
        self.intrinsic_mat = intrinsic_mat
        self.extrinsic_mat = extrinsic_mat
        self.imu_data = img_data

        self.record_values = [0] * len(self.pcd_fpaths)

    def get_plane_mesh(self, plane_model):
        """
        plane_model: [a, b, c, d] for plane equation ax + by + cz + d = 0
        """

        # Create a mesh grid for the plane
        x = np.linspace(0, 500, 100)
        y = np.linspace(-10, 10, 100)
        x, y = np.meshgrid(x, y)
        z = (-plane_model[3] - plane_model[0] *
             x - plane_model[1] * y) / plane_model[2]

        # Create a point cloud from the mesh grid
        plane_pcd = o3d.geometry.PointCloud()
        plane_points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
        plane_pcd.points = o3d.utility.Vector3dVector(plane_points)

        # Create a TriangleMesh from the mesh grid
        vertices = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
        triangles = []
        for i in range(x.shape[0] - 1):
            for j in range(x.shape[1] - 1):
                # Original triangles
                triangles.append(
                    [i * x.shape[1] + j, i * x.shape[1] + j + 1, (i + 1) * x.shape[1] + j])
                triangles.append([i * x.shape[1] + j + 1,
                                  (i + 1) * x.shape[1] + j + 1,
                                  (i + 1) * x.shape[1] + j])
                # Reversed triangles
                triangles.append([i * x.shape[1] + j, (i + 1)
                                 * x.shape[1] + j, i * x.shape[1] + j + 1])
                triangles.append([i * x.shape[1] + j + 1,
                                  (i + 1) * x.shape[1] + j,
                                  (i + 1) * x.shape[1] + j + 1])

        plane_mesh = o3d.geometry.TriangleMesh()
        plane_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        plane_mesh.triangles = o3d.utility.Vector3iVector(triangles)
        plane_mesh.paint_uniform_color([1, 0.6, 0.75])

        return plane_mesh

    def get_plot(self, vis):
        for i in tqdm(range(len(self.pcd_fpaths))):
            self.idx = i
            self.update_pcd(vis, verb=False)

        # gather the data values
        xs = self.timestamps
        ys = []
        yerr_negative = []
        yerr_positive = []
        for values in self.record_values:
            tmp = [v for v in values if v > 0.001]
            y = np.average(tmp)
            ys.append(y)
            yerr_negative.append(y - np.min(tmp))
            yerr_positive.append(np.max(tmp) - y)

        # gather the IMU data
        imu_xs = []
        angle_velocity = []
        for data in self.imu_data:
            if xs[0] <= data['timestamp_sec'] <= xs[-1]:
                imu_xs.append(
                    float(str(data['timestamp_sec']) + '.' +
                          str(data['timestamp_nanosec']))
                )
                angle_velocity.append(data['angular_velocity'])
        angle_velocity = [y for x, y in sorted(zip(imu_xs, angle_velocity))]
        imu_xs = sorted(imu_xs)
        imu_ys = [y[0] for y in angle_velocity]

        angle_change = [0, 0, 0]
        for i in range(len(imu_xs) - 1):
            for j in range(len(angle_change)):
                angle_change[j] += angle_velocity[i][j] * \
                    (imu_xs[i + 1] - imu_xs[i])
        print(angle_change)

        # plot
        fig, ax = plt.subplots()
        ax.plot(imu_xs, imu_ys, '.')
        vplot_data = [values for values in self.record_values]
        ax.violinplot(
            vplot_data, positions=xs,
            widths=0.1,
            showmeans=True
        )
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Diff of Body Volume')
        ymin, ymax = ax.get_ylim()
        ax.vlines(
            x=[t for i, t in enumerate(self.timestamps)
               if self.labels[i] == 1],
            ymin=ymin, ymax=ymax,
            colors='red', ls='--'
        )
        ax.vlines(
            x=[t for i, t in enumerate(self.timestamps)
               if self.labels[i] == -1],
            ymin=ymin, ymax=ymax,
            colors='blue', ls='--'
        )
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Diff of Body Volume')

        return True

    def update_pcd(self, vis, verb=True):
        # reset the scene
        viewpoint_param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        if self.current_pcd is not None:
            self.current_pcd.points = o3d.utility.Vector3dVector([])
            self.current_pcd.colors = o3d.utility.Vector3dVector([])

        # load target files
        img_fpath = self.img_fpaths[self.idx]
        bgr_img = cv2.imread(img_fpath)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pcd = load_pcd(self.pcd_fpaths[self.idx], mode=self.pcd_mode)
        pts = np.asarray(pcd.points)
        # 0 - background
        # 1... - animal IDs
        # [L, 1, H, W] where L is the number of object IDs
        seg_mask = np.load(self.mask_fpaths[self.idx])

        # erode the mask
        num_label = seg_mask.shape[0]
        for i in range(num_label):
            img2d = seg_mask[i, 0, :, :]
            seg_mask[i, 0, :, :] = sp.ndimage.binary_erosion(
                img2d, iterations=10)

        # project the point cloud to camera and its image sensor
        pts_in_cam = camera_utils.lidar2cam_projection(
            pts, self.extrinsic_mat)
        pts_in_img = camera_utils.cam2image_projection(
            pts_in_cam, self.intrinsic_mat)
        pts_in_cam = pts_in_cam.T[:, :-1]
        pts_in_img = pts_in_img.T[:, :-1]

        # make the colored point cloud
        img_height, img_width, _ = rgb_img.shape
        pcd_colors, valid_mask, pcd_seg, pcd_mask = projection_functions.get_coloured_point_cloud(
            pts_in_img, rgb_img, seg_mask,
            self.bbox_dict[os.path.basename(img_fpath)],
            width=img_width, height=img_height)
        rgb_pcd = np.concatenate([pts, pcd_colors], axis=1)
        # rgb_pcd = rgb_pcd[valid_mask]  # [N, 6]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            rgb_pcd[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(
            rgb_pcd[:, 3:])

        # Remove ground plane
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                                 ransac_n=3,
                                                 num_iterations=1000)
        num_points = np.array(pcd.points).shape[0]
        ground_mask = np.ones(num_points)
        ground_mask[inliers] = 0
        ground_plane_mesh = self.get_plane_mesh(plane_model)

        # Compute rotation matrix to align the normal of the ground plane to
        # Y-axis
        normal = plane_model[:3]
        up = np.array([0.0, 1.0, 0.0])
        rotation_axis = np.cross(normal, up)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        rotation_angle = np.arccos(
            np.dot(normal, up) / (np.linalg.norm(normal) * np.linalg.norm(up)))
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(
            rotation_angle * rotation_axis)

        # pick the target dots up and change the color
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        combined_mask = (ground_mask == 1) & (pcd_seg == 1)
        range_mask = (31.07 < points[:, 0]) & (points[:, 0] < 31.66) & \
                     (-0.166 < points[:, 1]) & (points[:, 1] < 0.495)
        animal_points = points[combined_mask & range_mask, :]
        colors[combined_mask, :] = [1, 0, 0]
        self.current_pcd.colors = o3d.utility.Vector3dVector(colors)
        self.record_values[self.idx] = []
        for i in range(animal_points.shape[0]):
            v = plane2point_distance(plane_model, animal_points[i, :])
            self.record_values[self.idx].append(v)

        # update the scene
        # pcd.rotate(rotation_matrix)
        # ground_plane_mesh.rotate(rotation_matrix)
        self.current_pcd = pcd
        vis.add_geometry(ground_plane_mesh)
        vis.add_geometry(self.current_pcd)
        vis.get_view_control().convert_from_pinhole_camera_parameters(viewpoint_param)
        if verb:
            print(os.path.basename(self.pcd_fpaths[self.idx]))

        self.increment_pcd_index()
        return True

    def increment_pcd_index(self,):
        self.idx += 1
        if len(self.pcd_fpaths) <= self.idx:
            self.idx %= len(self.pcd_fpaths)


def main():
    # args
    config_fpath = 'config.hjson'
    pcd_mode = 'open3d'

    # load the config file
    config = load_config_file(config_fpath)

    # load data file paths
    pcd_fpaths = sorted(glob.glob(
        os.path.join(config['pcd_dir'], '*.pcd')))
    img_fpaths = sorted(glob.glob(
        os.path.join(config['sync_rgb_dir'], '*.jpeg')))
    mask_fpaths = sorted(
        [
            os.path.join(config['mask_dir'], f)
            for f in os.listdir(config['mask_dir'])
            if re.search(r'\d+\.npy$', f)
        ],
        key=lambda x: int(os.path.basename(x).split('.')[0])
    )
    timestamps = sorted([
        get_timestamp_from_pcd_fpath(f)
        for f in pcd_fpaths
    ])
    img_data = json.load(open(config['imu_fpath'], 'r'))
    df = pd.read_excel(os.path.join(config['scene_dir'], 'body_state.xlsx'))
    labels = df['state']
    labels = labels.where(pd.notnull(labels), None).tolist()
    assert len(pcd_fpaths) == len(img_fpaths) == len(
        labels) == len(mask_fpaths)

    # load camera parameters
    calib_fpath = os.path.join(config['scene_dir'], 'manual_calibration.json')
    fx, fy, cx, cy, rot_mat, translation = load_camera_parameters(calib_fpath)

    # get the bounding box labels
    bbox_dict = get_bbox_labels(config['bbox_info_fpath'])

    # project the point cloud to camera and its image sensor
    intrinsic_mat = camera_utils.make_intrinsic_mat(fx, fy, cx, cy)
    extrinsic_mat = camera_utils.make_extrinsic_mat(rot_mat, translation)

    # prepare the open3d viewer
    init_geometry = load_pcd(pcd_fpaths[0], mode=pcd_mode)
    event_handler = KeyEvent(
        img_fpaths,
        pcd_fpaths,
        mask_fpaths,
        labels,
        timestamps,
        bbox_dict,
        img_data,
        intrinsic_mat, extrinsic_mat,
        pcd_mode=pcd_mode,
        init_geometry=init_geometry
    )
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    # register key callback functions with GLFW_KEY
    vis.register_key_callback(77, event_handler.get_plot)  # m
    vis.register_key_callback(32, event_handler.update_pcd)  # space
    vis.add_geometry(init_geometry)
    opt = vis.get_render_option()
    opt.show_coordinate_frame = True
    opt.background_color = np.asarray([0.7, 0.7, 0.7])
    vis.poll_events()
    vis.run()

    vis.destroy_window()


if __name__ == '__main__':
    main()
