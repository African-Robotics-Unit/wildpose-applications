import os
import glob
import time
import open3d as o3d
import numpy as np

from utils.file_loader import load_config_file, load_pcd
from utils.format_conversion import get_timestamp_from_pcd_fpath


class KeyEvent:
    def __init__(self, pcd_fpaths, pcd_mask_fpaths, timestamps, pcd_mode):
        self.pcd_fpaths = pcd_fpaths
        self.pcd_mask_fpaths = pcd_mask_fpaths
        self.timestamps = timestamps
        self.current_pcd = None
        self.pcd_idx = 0
        self.pcd_mode = pcd_mode

    def update_pcd(self, vis):
        # reset the scene
        viewpoint_param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        if self.current_pcd is not None:
            vis.remove_geometry(self.current_pcd)

        # load new pcd file
        pcd = load_pcd(self.pcd_fpaths[self.pcd_idx], mode=self.pcd_mode)
        pcd_mask = np.load(self.pcd_mask_fpaths[self.pcd_idx])

        # pick the target dots up
        self.current_pcd = pcd

        # get the principal line
        pcd_np = np.asarray(pcd.points)
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
    event_handler = KeyEvent(
        pcd_fpaths,
        pcd_mask_fpaths,
        timestamps,
        pcd_mode=pcd_mode,
    )
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.register_key_callback(32, event_handler.update_pcd)
    vis.add_geometry(load_pcd(pcd_fpaths[0], mode=pcd_mode))
    vis.run()

    vis.destroy_window()


if __name__ == '__main__':
    main()
