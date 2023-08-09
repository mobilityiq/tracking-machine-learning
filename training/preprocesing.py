import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter

def apply_gaussian_filter(data, sigma=1):
    return gaussian_filter(data, sigma=sigma)

def spline_interpolate(data, original_sample_rate=100, desired_sample_rate=20):
    x = np.arange(len(data))
    s = UnivariateSpline(x, data, s=0)
    new_length = int(len(data) * desired_sample_rate / original_sample_rate)
    new_x = np.linspace(0, len(data) - 1, new_length)
    return s(new_x)

def compute_magnitude(d):
    return np.sqrt(d[0]**2 + d[1]**2 + d[2]**2)

def compute_jerk(data):
    return [data[i+1] - data[i] for i in range(len(data)-1)] + [0]

def preprocess_data_for_prediction(data):
    # Extracting individual columns
    timestamps = data[:, 0]
    acc_data = data[:, 1:4]
    mag_data = data[:, 4:7]

    # Compute jerk
    acc_jerks_x, acc_jerks_y, acc_jerks_z = compute_jerk(acc_data[:, 0]), compute_jerk(acc_data[:, 1]), compute_jerk(acc_data[:, 2])
    mag_jerks_x, mag_jerks_y, mag_jerks_z = compute_jerk(mag_data[:, 0]), compute_jerk(mag_data[:, 1]), compute_jerk(mag_data[:, 2])

    # Compute magnitude
    acc_magnitudes = np.array([compute_magnitude(acc) for acc in acc_data])
    mag_magnitudes = np.array([compute_magnitude(mag) for mag in mag_data])


    # Constructing reshaped input
    X_acc = np.column_stack([acc_data])[..., np.newaxis]
    X_jerk = np.column_stack([acc_jerks_x, acc_jerks_y, acc_jerks_z])[..., np.newaxis]
    X_acc_mag = acc_magnitudes[..., np.newaxis]
    X_mag_jerk = np.column_stack([mag_jerks_x, mag_jerks_y, mag_jerks_z])[..., np.newaxis]
    X_mag_mag = mag_magnitudes[..., np.newaxis]
    X_magnetic = mag_data[..., np.newaxis]

    return [X_acc, X_jerk, X_acc_mag, X_mag_jerk, X_mag_mag, X_magnetic]