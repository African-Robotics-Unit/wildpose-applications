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
        pcd_mask = load_pcd(
            self.pcd_mask_fpaths[self.pcd_idx], mode=self.pcd_mode)
        self.current_pcd = pcd

        # update the scene
        vis.add_geometry(self.current_pcd)
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
    pcd_mask_fpaths = sorted([f for f in all_pcd_fpaths if '_mask.pcd' in f])
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
