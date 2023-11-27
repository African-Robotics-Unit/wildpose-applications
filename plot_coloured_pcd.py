import os
import re
import glob
import pandas as pd
import numpy as np
import scipy.ndimage
import open3d as o3d

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import scienceplots


# plt.style.use(['science', 'nature', 'no-latex'])
# figure(figsize=(10, 6))
plt.rcParams.update({
    'legend.frameon': False,
    "pdf.fonttype": 42,
})


def equal_3d_aspect(ax):
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    ax.set_box_aspect((xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]))


def main():
    pcd_fpath = '/Users/ikuta/Documents/Projects/wildpose-self-calibrator/data/giraffe_stand/textured_pcds/coloured_accumulation.pcd'
    # xlim = (-2, 2)
    # ylim = (-2, 3)
    # zlim = (40, 50)
    xlim = (-100, 100)
    ylim = (-100, 100)
    zlim = (85, 95)

    # load data
    pcd = o3d.io.read_point_cloud(pcd_fpath)
    points = np.asarray(pcd.points)  # [N, 3]
    colors = np.asarray(pcd.colors)  # [N, 3]

    # make the mask
    mask = (
        (xlim[0] < points[:, 0]) & (points[:, 0] < xlim[1]) &
        (ylim[0] < points[:, 1]) & (points[:, 1] < ylim[1]) &
        (zlim[0] < points[:, 2]) & (points[:, 2] < zlim[1])
    )
    masked_points = points[mask]
    masked_colors = colors[mask]

    # plot the data
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(
        masked_points[:, 0], masked_points[:, 1], masked_points[:, 2],
        c=masked_colors, s=1)  # s is the size of the points
    equal_3d_aspect(ax=ax)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('Depth (m)')

    plt.show()


if __name__ == '__main__':
    main()
