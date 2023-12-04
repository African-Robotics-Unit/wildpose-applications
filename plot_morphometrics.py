import os
import numpy as np
import pickle
import pandas as pd
from scipy.signal import lombscargle, butter, filtfilt, detrend
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import scienceplots

from utils.file_loader import load_config_file


plt.style.use(['science', 'nature', 'no-latex'])
figure(figsize=(10, 6))
plt.rcParams.update({
    "pdf.fonttype": 42,
})
figure(figsize=(3.5 * 2, 2.5))   # max width is 3.5 for single column

CONFIG = {
    "scene_dir": "data/lion_sleep3",
    "pcd_dir": "data/lion_sleep3/lidar",
    "sync_rgb_dir": "data/lion_sleep3/sync_rgb",
    "mask_dir": "data/lion_sleep3/masks_lion2",
    "textured_pcd_dir": "data/lion_sleep3/textured_pcds",
    "bbox_info_fpath": "data/lion_sleep3/train.json",
    "imu_fpath": "data/lion_sleep3/imu.json",
}


def normalize_data(data, new_min=-1, new_max=1):
    min_val = np.min(data)
    max_val = np.max(data)
    return ((data - min_val) / (max_val - min_val)) * (new_max - new_min) + new_min


def main():
    config = CONFIG

    # load the data
    with open('saved_data.pkl', 'rb') as f:
        input_data = pickle.load(f)
    timestamps = np.array(input_data['timestamp'])
    data = input_data['data']

    df = pd.read_excel(os.path.join(config['scene_dir'], 'body_state.xlsx'))
    labels = df['state']
    labels = labels.where(pd.notnull(labels), None).tolist()

    # set the start from t=0
    timestamps = timestamps - timestamps[0]

    # make the y values
    ys = [np.average(v) for v in data]

    # Interpolate ys onto a uniform grid
    interp_func = interp1d(timestamps, ys, kind='linear')
    uniform_timestamps = np.linspace(
        min(timestamps),
        max(timestamps),
        len(timestamps))
    uniform_ys = interp_func(uniform_timestamps)

    # Design a band-pass filter for the frequency range
    fs = 1 / np.mean(np.diff(uniform_timestamps))  # Sampling frequency
    lowcut = 0.75
    highcut = 1.4
    order = 6
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y_filtered = filtfilt(b, a, uniform_ys)

    # show the filtered plot
    normalized_y_filtered = normalize_data(np.array(y_filtered))
    # plt.subplot(3, 1, 2)
    plt.plot(uniform_timestamps, normalized_y_filtered)
    ta = None
    tb = None
    color = None
    for attribute, measurement in penguin_means.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # calculate the Lomb-Scargle periodogram
    f = np.linspace(0.01, 2, 1000)  # Frequency range
    pgram = lombscargle(np.array(timestamps), np.array(ys), f)

    for fmt in ['svg', 'pdf']:
        plt.savefig(f"results/output.{fmt}", format=fmt, bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    main()
