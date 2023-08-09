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