import os
import glob
import time
import json
import open3d as o3d
import numpy as np

from utils.file_loader import load_config_file, load_pcd
from utils.format_conversion import get_timestamp_from_pcd_fpath


def save_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()


def load_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(filename)
    vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    vis.destroy_window()


def main():
    # args
    config_fpath = 'config.hjson'
    pcd_mode = 'open3d'
    viewpoint_fpath = 'viewpoint.json'

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

    # set the viewpoint
    pcd = load_pcd(pcd_fpaths[0], mode=pcd_mode)
    if not os.path.exists(viewpoint_fpath):
        save_view_point(pcd, viewpoint_fpath)

    # prepare the open3d viewer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    viewpoint_param = o3d.io.read_pinhole_camera_parameters(viewpoint_fpath)

    # prepare the coordinate frame
    max_values = np.array(pcd.points).max(axis=0)
    coord_frame = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([
            [0, 0, 0],
            [max_values[0], 0, 0],
            [0, max_values[1], 0],
            [0, 0, max_values[2]],
        ]),
        lines=o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]]),
    )
    coord_frame.paint_uniform_color(np.array([1, 0, 0]))

    for i, (pcd_fpath, pcd_mask_fpath, timestamp) in enumerate(zip(
        pcd_fpaths, pcd_mask_fpaths, timestamps
    )):
        # load pcd files
        pcd = load_pcd(pcd_fpath, mode=pcd_mode)
        pcd_mask = load_pcd(pcd_mask_fpath, mode=pcd_mode)

        # pick the target dots up
        pass

        # show the PCD in the viewer
        vis.add_geometry(coord_frame)
        vis.add_geometry(pcd)
        ctr.convert_from_pinhole_camera_parameters(viewpoint_param)
        vis.poll_events()
        vis.update_renderer()

        time.sleep(0.1)  # Sleep for a while to simulate video frame rate

        o3d.io.write_pinhole_camera_parameters(
            viewpoint_fpath,
            ctr.convert_to_pinhole_camera_parameters()
        )
        vis.remove_geometry(pcd)

    vis.destroy_window()


if __name__ == '__main__':
    main()
