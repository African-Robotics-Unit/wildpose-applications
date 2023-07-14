import os
import glob
import time
import open3d as o3d
import numpy as np

from utils.file_loader import load_config_file, load_pcd
from utils.format_conversion import get_timestamp_from_pcd_fpath


class KeyEvent:
    def __init__(self, pcd_fpaths, pcd_mask_fpaths,
                 timestamps, pcd_mode, init_geometry=None):
        self.pcd_fpaths = pcd_fpaths
        self.pcd_mask_fpaths = pcd_mask_fpaths
        self.timestamps = timestamps
        self.pcd_idx = 0
        self.pcd_mode = pcd_mode
        self.current_pcd = init_geometry

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
        # Z-axis
        normal = plane_model[:3]
        up = np.array([0.0, 1.0, 0.0])
        rotation_axis = np.cross(normal, up)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        rotation_angle = np.arccos(
            np.dot(normal, up) / (np.linalg.norm(normal) * np.linalg.norm(up)))
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(
            rotation_angle * rotation_axis)
        pcd.rotate(rotation_matrix)

        # pick the target dots up
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        combined_mask = (ground_mask == 1) & (pcd_mask == 1)
        animal_points = points[combined_mask, :]
        colors[combined_mask, :] = [1, 0, 0]
        self.current_pcd.colors = o3d.utility.Vector3dVector(colors)

        # Get the principal line of the non-ground plane
        pcd_np = animal_points
        mean = np.mean(pcd_np, axis=0)
        centered_data = pcd_np - mean
        u, s, vh = np.linalg.svd(centered_data, full_matrices=True)
        first_principal_component = vh[0, :]

        # generate points along the principal line
        line_points = mean + first_principal_component * \
            np.mgrid[-1:2:2][:, np.newaxis]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(line_points),
            lines=o3d.utility.Vector2iVector([[0, 1]])
        )

        # update the scene
        self.current_pcd = pcd
        vis.add_geometry(self.current_pcd)
        vis.add_geometry(line_set)
        vis.get_view_control().convert_from_pinhole_camera_parameters(viewpoint_param)

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

    # get file paths of textured point cloud
    all_pcd_fpaths = sorted(glob.glob(
        os.path.join(config['textured_pcd_folder'], '*.pcd')))
    pcd_fpaths = sorted([f for f in all_pcd_fpaths if '_mask.pcd' not in f])
    pcd_mask_fpaths = sorted(glob.glob(
        os.path.join(config['textured_pcd_folder'], '*_mask.npy')))
    timestamps = sorted([
        get_timestamp_from_pcd_fpath(f)
        for f in pcd_fpaths
    ])
    assert len(pcd_fpaths) == len(pcd_mask_fpaths) == len(timestamps)

    # prepare the open3d viewer
    init_geometry = load_pcd(pcd_fpaths[0], mode=pcd_mode)
    event_handler = KeyEvent(
        pcd_fpaths,
        pcd_mask_fpaths,
        timestamps,
        pcd_mode=pcd_mode,
        init_geometry=init_geometry
    )
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.register_key_callback(32, event_handler.update_pcd)
    vis.add_geometry(init_geometry)
    opt = vis.get_render_option()
    opt.show_coordinate_frame = True
    opt.background_color = np.asarray([0.7, 0.7, 0.7])
    vis.poll_events()
    vis.run()

    vis.destroy_window()


if __name__ == '__main__':
    main()
