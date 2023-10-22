import os
import numpy as np
import hjson
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


CONFIG = {
    "data_fpath": "data/calibration/validation_data.hjson",
    "result_dir": "results",
}


def main():
    # load the data
    with open(CONFIG['data_fpath'], 'r') as f:
        dataset = hjson.loads(f.read())

    # make the values for plot
    xs = []
    y_points = []
    y_bars = []
    for data in dataset:
        distance = data['distance (m)']
        gt = data['true lengths (m)']
        measurements = np.array(data['measured lengths (m)'])
        xs.append(distance)

        points = np.abs(measurements - gt) / gt
        y_points.append(points)
        y_bars.append(np.average(points))

    # Plotting
    # bars = plt.bar(range(len(xs)), y_bars, alpha=0.7)
    # for i, bar in enumerate(bars):
    #     yerr = y_points[i]
    #     plt.scatter(
    #         [bar.get_x() + bar.get_width() / 2] * len(yerr), yerr,
    #         facecolors='none', edgecolors='black')
    for i, x in enumerate(xs):
        plt.scatter(
            [x] * len(y_points[i]), y_points[i],
            facecolors='none', edgecolors='black')

    plt.xlabel('Distance (m)') # Set appropriate label for x-axis
    plt.ylabel('Absolute measurement error ratio') # Set appropriate label for y-axis
    # plt.xticks(range(len(xs)), np.round(xs))


    for fmt in ['svg', 'pdf']:
        plt.savefig(
            os.path.join(CONFIG['result_dir'], f"validation_plot.{fmt}"),
            format=fmt, bbox_inches="tight"
        )
    plt.show()


if __name__ == '__main__':
    main()
