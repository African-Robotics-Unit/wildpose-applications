import os
import numpy as np
import pickle
import pandas as pd
from collections import OrderedDict
from scipy.signal import lombscargle, butter, filtfilt, detrend
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import scienceplots

from utils.file_loader import load_config_file


plt.style.use(['science', 'nature', 'no-latex'])
plt.rcParams.update({
    "pdf.fonttype": 42,
})
# figure(figsize=(3.5, 2.5))   # max width is 3.5 for single column

DATA = {
    'Giraffe': {
        'height': [
            1,
        ],
        'body length': [1],
        'neck length': [1],
    },
    'Martial eagle': {
        'height': [
            0.494,  # 000_004.pcd
            0.501,  # 001_005.pcd
            0.493,  # 002_006.pcd
            0.493,  # 003_007.pcd
            0.493,  # 004_008.pcd
            0.488,  # 005_009.pcd
            0.489,  # 006_010.pcd
            0.492,  # 007_011.pcd
            0.497,  # 008_012.pcd
            0.497,  # 009_013.pcd
            0.497,  # 010_014.pcd
            0.497,  # 011_015.pcd
            0.497,  # 012_016.pcd
            0.493,  # 013_017.pcd
            0.493,  # 014_018.pcd
            0.493,  # 015_019.pcd
            0.490,  # 016_020.pcd
        ],
        'body length': [2],
        'neck length': [2],
    },
}


def main():
    length_kinds = list(DATA[list(DATA.keys())[0]].keys())
    x = np.arange(len(length_kinds))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(
        figsize=(3.5, 2.5),
        layout='constrained'
    )
    for animal, data in DATA.items():
        # collect values
        ys = [np.mean(data[l]) for l in length_kinds]
        # draw bars
        offset = width * multiplier
        rects = ax.bar(
            x + offset,
            height=ys,
            width=width,
            label=animal)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('Length (m)')
    ax.set_xticks(x + width, length_kinds)
    ax.legend(loc='upper left')
    # ax.set_ylim(0, 250)

    # for fmt in ['svg', 'pdf']:
    #     plt.savefig(f"results/output.{fmt}", format=fmt, bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    main()
