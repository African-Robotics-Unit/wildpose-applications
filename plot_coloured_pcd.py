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
import plotly.graph_objs as go


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
    # pcd_fpath = '/Users/ikuta/Documents/Projects/wildpose-self-calibrator/data/giraffe_stand/textured_pcds/coloured_accumulation.pcd'
    # xlim = (-100, 100)
    # ylim = (-100, 100)
    # zlim = (85, 95)
    pcd_fpath = '/Users/ikuta/Documents/Projects/wildpose-self-calibrator/data/martial_eagle_stand/textured_pcds/coloured_accumulation.pcd'
    xlim = (-100, -0.1)
    ylim = (-100, 100)
    zlim = (18.6, 18.9)

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
    fig = go.Figure()
    point_cloud_scatter = go.Scatter3d(
        x=masked_points[:, 0],
        y=masked_points[:, 1],
        z=masked_points[:, 2],
        mode='markers',
        marker=dict(size=2, color=masked_colors)
    )
    fig.add_trace(point_cloud_scatter)

    def _axis_dict(title, range=None):
        return dict(
            title=title,
            ticks='outside',
            tickangle=0,
            backgroundcolor='rgb(230, 230, 230)',
            tickformat='.1f',
            range=None
        )

    fig.update_layout(
        font_family='Arial',
        font_size=14,
        scene=dict(
            xaxis=_axis_dict('x (m)', range=[-2, 1]),
            yaxis=_axis_dict('y (m)'),
            zaxis=_axis_dict('Depth (m)'),
            aspectmode='data',
        ),
    )
    # fig.layout.scene.camera.projection.type = "orthographic"

    fig.show()


if __name__ == '__main__':
    main()
