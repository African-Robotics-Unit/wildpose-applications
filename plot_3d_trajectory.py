import os
import re
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def get_timestamp_from_pcd_fpath(fpath: str) -> float:
    fname = os.path.splitext(os.path.basename(fpath))[0]
    # get timestamp
    fname = fname.split('_')
    msg_id = '_'.join(fname[:-2])
    timestamp = float(fname[-2] + '.' + fname[-1])

    return timestamp


def main():
    data_dir = '/Users/ikuta/Documents/Projects/wildpose-applications/data/springbok_herd2/trajectory_raw'

    # load data
    all_csv_fpaths = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    csv_fpaths = [
        f for f in all_csv_fpaths
        if re.fullmatch(r'\d+\.csv', os.path.basename(f))
    ]

    dfs = {}
    for csv_fpath in csv_fpaths:
        key = os.path.splitext(os.path.basename(csv_fpath))[0]
        df = pd.read_csv(
            csv_fpath,
            names=['time', 'x', 'y', 'z'], header=0
        )
        df = df.where(df != -1e-6, other=np.nan)
        dfs[key] = df

    # load timestamp
    csv_fpath = os.path.join(data_dir, 'lidar_frames.csv')
    df = pd.read_csv(csv_fpath, names=['file_name'], header=0, index_col=0)
    timestamps = np.array(df['file_name'].apply(
        get_timestamp_from_pcd_fpath).tolist())
    time = timestamps - timestamps[0]

    # plot the data
    ax = plt.figure().add_subplot(projection='3d')
    for k, v in dfs.items():
        ax.plot(v['x'], v['z'], time, label=k)
    ax.legend()
    # ax.set_box_aspect([1.0, 1.0, 1.0])
    # ax.set_xlim(0, 400)
    # ax.set_ylim(0, 400)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)')
    ax.set_zlabel('Time (s)')

    plt.show()


if __name__ == '__main__':
    main()
