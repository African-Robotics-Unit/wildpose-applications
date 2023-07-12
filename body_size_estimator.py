import os
import glob
from pprint import pprint

from utils.file_loader import load_config_file, load_pcd
from utils.format_conversion import get_timestamp_from_pcd_fpath


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

    for (pcd_fpath, pcd_mask_fpath, timestamp) in zip(
        pcd_fpaths, pcd_mask_fpaths, timestamps
    ):
        # load pcd files
        pcd = load_pcd(pcd_fpath, mode=pcd_mode)
        pcd_mask = load_pcd(pcd_mask_fpath, mode=pcd_mode)

        # pick the target dots up
        pass

        # show the PCD in the viewer
        pass


if __name__ == '__main__':
    main()
