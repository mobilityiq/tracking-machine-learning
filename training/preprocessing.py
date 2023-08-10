import os
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter

class Preprocessing:
    LABEL_MAP = {
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
    
    @staticmethod
    def data_for_classification_model(users, motion_files):
        all_data = []
        for user in users:
            user_folder = os.path.join("files", user)
            dates_folders = [folder for folder in os.listdir(user_folder) if not folder.startswith('.')]

            for date_folder in dates_folders:
                label_file_path = os.path.join(user_folder, date_folder, "Label.txt")

                try:
                    labels = Preprocessing.read_label_file(label_file_path)
                except FileNotFoundError:
                    print(f"Warning: {label_file_path} not found. Skipping...")
                    continue

                # Downsample the labels from 100Hz to 20Hz
                labels = labels[::5]  # Take every 5th label

                for motion_file in motion_files:
                    motion_file_path = os.path.join(user_folder, date_folder, motion_file)

                    if os.path.exists(motion_file_path):
                        data = Preprocessing.read_motion_data(motion_file_path)
                        print("Shape of data after reading:", data.shape)

                        data = data[::5]
                        print("Shape of data after interpolation:", data.shape)

                        timestamps = [row[0] for row in data]
                        acc_data = [row[1:4] for row in data]
                        mag_data = [row[4:7] for row in data]   

                        for idx, (acc_values, mag_values) in enumerate(zip(acc_data, mag_data)):
                            mode = labels[idx]
                            if mode != 0:
                                modeString = Preprocessing.LABEL_MAP[mode]
                                # Append the data in the desired order
                                row = ([timestamps[idx]] +
                                    list(acc_values) +  # Acceleration x,y,z -> Channel 1
                                    list(mag_values) +  # Magnetic x,y,z -> Channel 6
                                    [modeString]
                                )
                                all_data.append(row)
        return all_data

    @staticmethod
    def data_for_cnn_bilstm(users, motion_files):
        all_data = []

        for user in users:
            user_folder = os.path.join("files", user)
            dates_folders = [folder for folder in os.listdir(user_folder) if not folder.startswith('.')]

            for date_folder in dates_folders:
                label_file_path = os.path.join(user_folder, date_folder, "Label.txt")

                try:
                    labels = Preprocessing.read_label_file(label_file_path)
                except FileNotFoundError:
                    print(f"Warning: {label_file_path} not found. Skipping...")
                    continue

                # Downsample the labels from 100Hz to 20Hz
                labels = labels[::5]  # Take every 5th label

                for motion_file in motion_files:
                    motion_file_path = os.path.join(user_folder, date_folder, motion_file)

                    if os.path.exists(motion_file_path):
                        data = Preprocessing.read_motion_data(motion_file_path)
                        print("Shape of data after reading:", data.shape)

                        data = Preprocessing.apply_gaussian_filter(data=data)
                        print("Shape of data after gaussian filter:", data.shape)

                        data = data[::5]
                        print("Shape of data after interpolation:", data.shape)

                        timestamps = [row[0] for row in data]
                        acc_data = [row[1:4] for row in data]
                        mag_data = [row[4:7] for row in data]   

                        # Compute jerk 
                        acc_jerks_x, acc_jerks_y, acc_jerks_z = Preprocessing.compute_jerk([row[0] for row in acc_data]), Preprocessing.compute_jerk([row[1] for row in acc_data]), Preprocessing.compute_jerk([row[2] for row in acc_data])
                        mag_jerks_x, mag_jerks_y, mag_jerks_z = Preprocessing.compute_jerk([row[0] for row in mag_data]), Preprocessing.compute_jerk([row[1] for row in mag_data]), Preprocessing.compute_jerk([row[2] for row in mag_data])

                        # Compute magnitude
                        acc_magnitudes = [Preprocessing.compute_magnitude(acc) for acc in acc_data]
                        mag_magnitudes = [Preprocessing.compute_magnitude(mag) for mag in mag_data]

                        for idx, (acc_values, mag_values) in enumerate(zip(acc_data, mag_data)):
                            mode = labels[idx]
                            if mode != 0:
                                # Append the data in the desired order
                                row = (
                                    list(acc_values) +  # Acceleration x,y,z -> Channel 1
                                    [acc_jerks_x[idx], acc_jerks_y[idx], acc_jerks_z[idx]] +  # Acceleration jerk x,y,z -> Channel 2
                                    [acc_magnitudes[idx]] +  # Acceleration magnitude -> Channel 3
                                    [mag_jerks_x[idx], mag_jerks_y[idx], mag_jerks_z[idx]] +  # Magnetic jerk x,y,z -> Channel 4
                                    [mag_magnitudes[idx]] +  # Magnetic magnitude -> Channel 5
                                    list(mag_values) +  # Magnetic x,y,z -> Channel 6
                                    [mode]
                                )
                                all_data.append(row)
        return all_data