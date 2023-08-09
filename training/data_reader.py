import os
import numpy as np
from scipy.ndimage import gaussian_filter
from preprocesing import apply_gaussian_filter, compute_jerk, compute_magnitude

class DataReader :
    #  Still=1, Walking=2, Run=3, Bike=4, Car=5, Bus=6, Train=7, Subway=8
    fine_label_mapping = {
        0: "unknown",
        1: "stationary",
        2: "walking",
        3: "running",
        4: "cycling",
        5: "driving",
        6: "bus",
        7: "train",
        8: "metro"
    }

    def read_motion_data(file_path):
        return np.genfromtxt(file_path, delimiter=' ', dtype=float, usecols=[0, 1, 2, 3, 7, 8, 9])

    def read_label_file(file_path):
        return np.genfromtxt(file_path, delimiter=' ', dtype=int, usecols=[1])

