import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import open3d as o3d
# import pyvista as pv

from utils.file_loader import load_config_file, load_pcd
from utils.format_conversion import get_timestamp_from_pcd_fpath


class KeyEvent:
    def __init__(self, pcd_fpaths,
                 timestamps, pcd_mode, init_geometry=None):
        self.pcd_fpaths = pcd_fpaths
        self.timestamps = timestamps
        self.pcd_idx = 0
        self.pcd_mode = pcd_mode
        self.current_pcd = init_geometry

        self.body_heights = [0] * len(self.pcd_fpaths)

    def update_pcd(self, vis):
        # reset the scene
        viewpoint_param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        if self.current_pcd is not None:
            self.current_pcd.points = o3d.utility.Vector3dVector([])
            self.current_pcd.colors = o3d.utility.Vector3dVector([])

        # load new pcd file
        pcd = load_pcd(self.pcd_fpaths[self.pcd_idx], mode=self.pcd_mode)

        # update the scene
        self.current_pcd = pcd
        vis.add_geometry(self.current_pcd)
        vis.get_view_control().convert_from_pinhole_camera_parameters(viewpoint_param)

        self.increment_pcd_index()
        return True

    def increment_pcd_index(self,):
        self.pcd_idx += 10
        if len(self.pcd_fpaths) <= self.pcd_idx:
            self.pcd_idx %= len(self.pcd_fpaths)


def main():
    # args
    pcd_mode = 'open3d'

    # load the config file
    pcd_fpaths = [
        os.path.join(
            '/Users/ikuta/Documents/Projects/wildpose-self-calibrator/data/martial_eagle/lidar',
            x)
        for x in [
            'livox_frame_1670316371_416378416.pcd',
            'livox_frame_1670316371_517465712.pcd',
            'livox_frame_1670316371_617302160.pcd',
            'livox_frame_1670316371_717487632.pcd',
            'livox_frame_1670316371_817358256.pcd',
            'livox_frame_1670316371_916403504.pcd',
            'livox_frame_1670316372_017466128.pcd',
            'livox_frame_1670316372_117501424.pcd',
            'livox_frame_1670316372_217646640.pcd',
            'livox_frame_1670316372_317414576.pcd',
            'livox_frame_1670316372_416408496.pcd',
            'livox_frame_1670316372_517364912.pcd',
            'livox_frame_1670316372_617444208.pcd',
            'livox_frame_1670316372_717515888.pcd',
            'livox_frame_1670316372_817410960.pcd',
            'livox_frame_1670316372_916385232.pcd',
            'livox_frame_1670316373_017311792.pcd',
            'livox_frame_1670316373_116479760.pcd',
            'livox_frame_1670316373_217650416.pcd',
            'livox_frame_1670316373_317481392.pcd',
            'livox_frame_1670316373_417421264.pcd',
            'livox_frame_1670316373_516967536.pcd',
            'livox_frame_1670316373_616309744.pcd',
            'livox_frame_1670316373_717420880.pcd',
            'livox_frame_1670316373_817867952.pcd',
        ]
    ]

    # Iterate through the list of point cloud data and combine
    combined_pcd = o3d.geometry.PointCloud()
    for pcd_fpath in pcd_fpaths:
        # Add the points from each point cloud to the combined point cloud
        combined_pcd += load_pcd(pcd_fpath, mode=pcd_mode)

    o3d.io.write_point_cloud("results/accumulation.pcd", combined_pcd)

    # visualize
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    # register key callback functions with GLFW_KEY
    vis.add_geometry(combined_pcd)
    opt = vis.get_render_option()
    opt.show_coordinate_frame = True
    opt.background_color = np.asarray([0.7, 0.7, 0.7])
    vis.poll_events()
    vis.run()

    vis.destroy_window()


if __name__ == '__main__':
    main()
