import os
import math
import numpy as np
import random
from collections import Counter
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter
from collections import Counter
from keras import backend as K


class Preprocessing:
    LABEL_MAP = {
        4: "cycling",
        5: "driving",
        6: "bus",
        7: "train",
        8: "metro"
    }

    @staticmethod
    def f1_metric(y_true, y_pred):
        """
        Compute the F1 score, also known as balanced F-score or F-measure.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        
        f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
        return f1_val

    def read_motion_data(file_path):
        return np.genfromtxt(file_path, delimiter=' ', dtype=float, usecols=[0, 1, 2, 3, 7, 8, 9])
    
    def read_3_0_motion_data(file_path):
        return np.genfromtxt(file_path, delimiter=' ', dtype=float, usecols=[0, 1, 2, 3, 10, 11, 12, 13])
    
    def read_3_0_gps_data(file_path):
        return np.genfromtxt(file_path, delimiter=' ', dtype=float, usecols=[0, 4, 5])

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
    def calculate_speed_and_course(row1, row2):
        # Extract the relevant data from the rows
        timestamp1, lat1, lon1 = row1[0], row1[1], row1[2]
        timestamp2, lat2, lon2 = row2[0], row2[1], row2[2]

        # Calculate speed
        distance = Preprocessing.haversine(lat1, lon1, lat2, lon2)  # Assuming you've a haversine formula method
        time_diff = (timestamp2 - timestamp1) / 1000.0  # Convert to seconds
        speed = distance / time_diff

        # Calculate course (trajectory)
        course = math.atan2(math.sin(lon2 - lon1) * math.cos(lat2), 
                        math.cos(lat1) * math.sin(lat2) - 
                        math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1))
        course = math.degrees(course)
        course = (course + 360) % 360  # Ensure it's between 0 and 360

        return speed, course
    
    # Haversine function to calculate distance between two lat-lon points
    @staticmethod
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371000  # Radius of Earth in meters
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
            math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
            math.sin(dlon / 2) * math.sin(dlon / 2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        return distance

    @staticmethod
    def get_motion_data_between(timestamps, start_time, end_time):
        return [idx for idx, ts in enumerate(timestamps) if start_time <= ts <= end_time]


    @staticmethod
    def data_for_3_0(users, motion_files):
        all_data = []
        for user in users:
            print(f"Processing data for: {user}") 
            user_folder = os.path.join("files", user)
            dates_folders = [folder for folder in os.listdir(user_folder) if not folder.startswith('.')]
            
            for date_folder in dates_folders:
                print(f"\tProcessing date: {date_folder}") 
                label_file_path = os.path.join(user_folder, date_folder, "Label.txt")

                try:
                    labels = Preprocessing.read_label_file(label_file_path)
                except FileNotFoundError:
                    print(f"Warning: {label_file_path} not found. Skipping...")
                    continue

                # Calculate the minimum mode count for random sampling
                label_counts = Counter(labels)
                min_mode_count = min(label_counts.values())

                # Filter and downsample labels
                downsampled_labels = labels[::5]
                indices_to_use = random.sample(range(len(downsampled_labels)), min_mode_count)
                
                for motion_file in motion_files:
                    print(f"\t\tProcessing motion type: {motion_file}") 
                    motion_file_path = os.path.join(user_folder, date_folder, motion_file)

                    if os.path.exists(motion_file_path):
                        data = Preprocessing.read_3_0_motion_data(motion_file_path)[::5]
                        gps = Preprocessing.read_3_0_gps_data(motion_file_path)

                        timestamps = data[:, 0]
                        acc_data = data[:, 1:4]
                        qua_data = data[:, 4:8]
                        gps_data = gps[:, :3]

                        for i in range(len(gps_data) - 1):  
                            speed, course = Preprocessing.calculate_speed_and_course(gps_data[i], gps_data[i+1])
                            motion_indices = Preprocessing.get_motion_data_between(timestamps, gps_data[i][0], gps_data[i+1][0])

                            for idx in motion_indices:
                                if idx in indices_to_use:
                                    acc_values = acc_data[idx]
                                    qua_values = qua_data[idx]
                                    mode = downsampled_labels[idx]
                                    if mode != 0:
                                        modeString = Preprocessing.LABEL_MAP[mode]
                                        row = ([timestamps[idx]] + [speed] + [course] + list(acc_values) + list(qua_values) + [modeString])
                                        all_data.append(row)
        return all_data
    
    @staticmethod
    def data_for_bi_lstm_model(users, motion_files):
        all_data = np.empty((0, 1, 8))
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
                downsampled_labels = labels[::5]  # Take every 5th label
                print(f"Shape of downsampled labels for {label_file_path}: {len(downsampled_labels)}")


                for motion_file in motion_files:
                    motion_file_path = os.path.join(user_folder, date_folder, motion_file)

                    if os.path.exists(motion_file_path):
                        data = Preprocessing.read_motion_data(motion_file_path)
                        print(f"Shape of data after reading {motion_file_path}:", data.shape)

                        data = data[::5]
                        print("Shape of data after interpolation:", data.shape)

                        timestamps = [row[0] for row in data]
                        acc_data = [row[1:4] for row in data]
                        mag_data = [row[4:7] for row in data]   

                        for idx, (acc_values, mag_values) in enumerate(zip(acc_data, mag_data)):
                            mode = downsampled_labels[idx]
                            if mode >3:
                                # Check for NaN values in acc_values, mag_values, and mode
                                if np.isnan(acc_values).any() or np.isnan(mag_values).any() or np.isnan(mode):
                                    print(f"Skipping idx {idx} due to NaN values")
                                    continue

                                modeString = Preprocessing.LABEL_MAP[mode]
                                # Append the data in the desired order
                                row = np.array([
                                    timestamps[idx],
                                    acc_values[0],  # x acceleration
                                    acc_values[1],  # y acceleration
                                    acc_values[2],  # z acceleration
                                    mag_values[0],  # mx magnetometer
                                    mag_values[1],  # my magnetometer
                                    mag_values[2],   # mz magnetometer
                                    modeString
                                ])[np.newaxis, np.newaxis, :]

                                all_data = np.append(all_data, row, axis=0)

        return all_data


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
                            if mode > 3:
                                modeString = Preprocessing.LABEL_MAP[mode]
                                # Append the data in the desired order
                                row = ([timestamps[idx]] +
                                    list(acc_values) +  # Acceleration x,y,z -> Channel 1
                                    list(mag_values) +  # Magnetic x,y,z -> Channel 6
                                    [modeString]
                                )

                                all_data.append(row)

        # Extract the labels (last column) from all_data
        labels = [row[-1] for row in all_data]

        # Count the occurrences of each label
        label_counts = Counter(labels)

        print(label_counts)
        return all_data
    
    # MARK: - CNN-BiLSTM
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
                    print("Labels: ",len(labels))
                except FileNotFoundError:
                    print(f"Warning: {label_file_path} not found. Skipping...")
                    continue

                # Downsample the labels from 100Hz to 20Hz
                labels = labels[::5]  # Take every 5th label
                print("Downsampled Labels: ",len(labels))

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

                        # acc_data =  Preprocessing.apply_gaussian_filter(data=acc_data, sigma=1)
                        # mag_data = Preprocessing.apply_gaussian_filter(data=mag_data, sigma=1)

                        # Compute jerk 
                        acc_jerks_x, acc_jerks_y, acc_jerks_z = Preprocessing.compute_jerk([row[0] for row in acc_data]), Preprocessing.compute_jerk([row[1] for row in acc_data]), Preprocessing.compute_jerk([row[2] for row in acc_data])
                        mag_jerks_x, mag_jerks_y, mag_jerks_z = Preprocessing.compute_jerk([row[0] for row in mag_data]), Preprocessing.compute_jerk([row[1] for row in mag_data]), Preprocessing.compute_jerk([row[2] for row in mag_data])

                        # Compute magnitude
                        acc_magnitudes = [Preprocessing.compute_magnitude(acc) for acc in acc_data]
                        mag_magnitudes = [Preprocessing.compute_magnitude(mag) for mag in mag_data]

                        

                        for idx, (acc_values, mag_values) in enumerate(zip(acc_data, mag_data)):
                            mode = labels[idx]

                            if mode > 3:
                                # Append the data in the desired order
                                row = (
                                    [timestamps[idx]] +
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