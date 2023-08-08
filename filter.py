import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.signal import lombscargle, butter, filtfilt
from scipy.interpolate import interp1d


def main():
    # load the data
    with open('saved_data.pkl', 'rb') as f:
        input_data = pickle.load(f)
    timestamps = input_data['timestamp']
    data = input_data['data']

    # make the y values
    ys = [np.average(v) for v in data]

    # Interpolate ys onto a uniform grid
    interp_func = interp1d(timestamps, ys, kind='linear')
    uniform_timestamps = np.linspace(
        min(timestamps),
        max(timestamps),
        len(timestamps))
    uniform_ys = interp_func(uniform_timestamps)

    # Design a band-pass filter for the frequency range from 0.75 to 1.5 Hz
    fs = 1 / np.mean(np.diff(uniform_timestamps))  # Sampling frequency
    lowcut = 0.75
    highcut = 1.5
    order = 6
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y_filtered = filtfilt(b, a, uniform_ys)

    # show the original plot
    plt.subplot(3, 1, 1)
    plt.plot(timestamps, ys)
    plt.title('Original Plot')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    # show the filtered plot
    plt.subplot(3, 1, 2)
    plt.plot(uniform_timestamps, y_filtered)
    plt.title('Filtered Plot')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    # calculate the Lomb-Scargle periodogram
    f = np.linspace(0.01, 2, 1000)  # Frequency range
    pgram = lombscargle(np.array(timestamps), np.array(ys), f)

    # show the spectral plot
    plt.subplot(3, 1, 3)
    plt.plot(f, pgram)
    plt.title('Lomb-Scargle Periodogram')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.show()


if __name__ == '__main__':
    main()
