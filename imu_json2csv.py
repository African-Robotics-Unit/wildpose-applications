import os
import re
import glob
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from utils.format_conversion import get_timestamp_from_pcd_fpath


def main():
    data_dir = '/Users/ikuta/Documents/Projects/wildpose-applications/data/springbok_herd2/'

    # load data
    lidar_fpaths = sorted(glob.glob(os.path.join(data_dir, 'lidar', '*.pcd')))
    imu_json_fpath = os.path.join(data_dir, 'imu.json')
    with open(imu_json_fpath) as f:
        imu_data = json.load(f)

    # get the timestamp range
    start_time = get_timestamp_from_pcd_fpath(lidar_fpaths[0])
    end_time = get_timestamp_from_pcd_fpath(lidar_fpaths[-1])

    # extract the IMU data within the time
    imu_frames = []
    for imu_frame in imu_data:
        timestamp = float(
            str(imu_frame['timestamp_sec']) + '.' + str(imu_frame['timestamp_nanosec']))
        if start_time <= timestamp <= end_time:
            imu_frames.append(imu_frame)

    # make the csv file
    column_names = \
        [f'linear_acceleration_{x}' for x in ['x', 'y', 'z']] + \
        [f'orientation_{x}' for x in ['x', 'y', 'z', 'w']] + \
        [f'linear_acceleration_{x}' for x in ['x', 'y', 'z']] + \



if __name__ == '__main__':
    main()
