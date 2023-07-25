import os
import json
import glob
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

from utils.file_loader import load_config_file, load_pcd, load_camera_parameters
from utils.format_conversion import get_timestamp_from_pcd_fpath


def plane2point_distance(plane, point):
    pt_vec = np.ones(4)
    pt_vec[:3] = point
    plane = np.array(plane)
    return np.abs(plane @ pt_vec) / np.linalg.norm(pt_vec[:3])


class KeyEvent:
    def __init__(self, pcd_fpaths, pcd_mask_fpaths,
                 labels,
                 timestamps, pcd_mode, init_geometry=None):
        self.pcd_fpaths = pcd_fpaths
        self.pcd_mask_fpaths = pcd_mask_fpaths
        self.labels = labels
        self.timestamps = timestamps
        self.pcd_idx = 0
        self.pcd_mode = pcd_mode
        self.current_pcd = init_geometry

        self.record_values = [0] * len(self.pcd_fpaths)

    def get_plot(self, vis):
        for i in tqdm(range(len(self.pcd_fpaths))):
            self.pcd_idx = i
            self.update_pcd(vis)

        fig, ax = plt.subplots()
        ax.plot(self.timestamps, self.record_values, '-o')
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
        plt.show()
        return True

    def update_pcd(self, vis):
        # reset the scene
        viewpoint_param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        if self.current_pcd is not None:
            self.current_pcd.points = o3d.utility.Vector3dVector([])
            self.current_pcd.colors = o3d.utility.Vector3dVector([])

        # load new pcd file
        pcd = load_pcd(self.pcd_fpaths[self.pcd_idx], mode=self.pcd_mode)
        # 0 - background
        # 1... - animal IDs
        pcd_mask = np.load(self.pcd_mask_fpaths[self.pcd_idx])  # [N,]

        # Remove ground plane
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                                 ransac_n=3,
                                                 num_iterations=1000)
        ground_mask = np.ones_like(pcd_mask)
        ground_mask[inliers] = 0

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
        combined_mask = (ground_mask == 1) & (pcd_mask == 1)
        animal_points = points[combined_mask, :]
        colors[combined_mask, :] = [1, 0, 0]
        self.current_pcd.colors = o3d.utility.Vector3dVector(colors)
        self.record_values[self.pcd_idx] = 0
        for i in range(animal_points.shape[0]):
            self.record_values[self.pcd_idx] += plane2point_distance(
                plane_model, animal_points[i, :])

        # update the scene
        pcd.rotate(rotation_matrix)
        self.current_pcd = pcd
        vis.add_geometry(self.current_pcd)
        vis.get_view_control().convert_from_pinhole_camera_parameters(viewpoint_param)
        print(os.path.basename(self.pcd_fpaths[self.pcd_idx]))

        self.increment_pcd_index()
        return True

    def increment_pcd_index(self,):
        self.pcd_idx += 1
        if len(self.pcd_fpaths) <= self.pcd_idx:
            self.pcd_idx %= len(self.pcd_fpaths)


def main():
    # args
    config_fpath = 'config.hjson'
    pcd_mode = 'open3d'

    # load the config file
    config = load_config_file(config_fpath)

    # load data file paths
    pcd_fpaths = sorted(glob.glob(
        os.path.join(config['scene_dir'], config['pcd_dir'], '*.pcd')))
    img_fpaths = sorted(glob.glob(
        os.path.join(config['scene_dir'], config['sync_rgb_dir'], '*.jpeg')))
    timestamps = sorted([
        get_timestamp_from_pcd_fpath(f)
        for f in pcd_fpaths
    ])
    df = pd.read_excel(os.path.join(config['scene_dir'], 'body_state.xlsx'))
    labels = df['state']
    labels = labels.where(pd.notnull(labels), None).tolist()
    assert len(pcd_fpaths) == len(img_fpaths) == len(labels)

    # load camera parameters
    calib_fpth = os.path.join(config['scene_dir'], 'manual_calibration.json')
    fx, fy, cx, cy, rot_mat, translation = load_camera_parameters(calib_fpth)

    # TODO

    # prepare the open3d viewer
    init_geometry = load_pcd(pcd_fpaths[0], mode=pcd_mode)
    event_handler = KeyEvent(
        pcd_fpaths,
        pcd_mask_fpaths,
        labels,
        timestamps,
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
