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
        'height': [1],
        'body length': [1],
        'neck length': [1],
    },
    'Martial eagle': {
        'height': [2],
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
