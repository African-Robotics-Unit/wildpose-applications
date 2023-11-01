import os
import numpy as np
import hjson
import pandas as pd

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.ticker import StrMethodFormatter
import scienceplots


plt.style.use(['science', 'nature', 'no-latex'])
figure(figsize=(10, 6))
plt.rcParams.update({
    "pdf.fonttype": 42,
})

CONFIG = {
    "data_fpath": "data/calibration/validation_data.hjson",
    "result_dir": "results",
}

def polynomial(x, a, b, c):
    return a * x**2 + b * x + c


def main():
    # Load the data
    with open(CONFIG['data_fpath'], 'r') as f:
        dataset = hjson.loads(f.read())

    # Prepare the values for plot
    xs = []
    y_points = []
    y_bars = []
    for data in dataset:
        distance = data['distance (m)']
        gt = data['true lengths (m)']
        measurements = np.array(data['measured lengths (m)'])
        xs.append(distance)

        points = np.abs(measurements - gt) / gt * 100.0
        y_points.append(points)
        y_bars.append(np.average(points))

    # calculate the error bars
    unique_xs = np.unique(xs)
    y_means = []
    y_stds = []
    for x in unique_xs:
        indices = np.argwhere(x == np.array(xs)).flatten()
        data = np.array(y_points)[indices].flatten()
        y_means.append(np.mean(data))
        y_stds.append(np.std(data))

    # Perform curve fitting
    popt, _ = curve_fit(polynomial, unique_xs, y_means)
    # Generate x values for the fit curve
    x_fit = np.linspace(min(unique_xs), max(unique_xs), 500)
    # Generate y values based on the fit
    y_fit = polynomial(x_fit, *popt)

    # Plotting with error bars
    plt.errorbar(
        unique_xs, y_means, yerr=y_stds,
        fmt='o', markersize=5, capsize=5,
        label='Mean ± 1SD'
    )
    plt.plot(x_fit, y_fit, 'r--', label='Fit Curve')

    plt.xlabel('Distance (m)')  # Label for x-axis
    plt.ylabel('Absolute percentage error (%)')  # Label for y-axis
    plt.legend()

    for fmt in ['svg', 'pdf']:
        plt.savefig(
            os.path.join(CONFIG['result_dir'], f"validation_plot_with_error_bars.{fmt}"),
            format=fmt, bbox_inches="tight"
        )
    plt.show()


if __name__ == '__main__':
    main()
