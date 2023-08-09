import os
import numpy as np
from scipy.ndimage import gaussian_filter
from preprocesing import apply_gaussian_filter, compute_jerk, compute_magnitude

def read_motion_data(file_path):
    return np.genfromtxt(file_path, delimiter=' ', dtype=float, usecols=[0, 1, 2, 3, 7, 8, 9])

def read_label_file(file_path):
    return np.genfromtxt(file_path, delimiter=' ', dtype=int, usecols=[1])