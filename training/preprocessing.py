import os
import csv
import math
import pickle
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
from keras import backend as K
from datetime import datetime
from tqdm import tqdm

class Preprocessing:
    LABEL_MAP = {
        4: "cycling",
        5: "driving",
        6: "bus",
        7: "train",
        8: "metro"
    }
    

    @staticmethod
    def load_and_process_data(locations, is_validation=False):
        data = Preprocessing.data_from_phone_locations(locations=locations, is_validation=is_validation)
        
        # Check if the data is empty
        if not data:
            print("Warning: Data is empty!")
            return None, None

        try:
            data = np.array(data)
        except ValueError as e:
            print(f"Error when converting data to numpy array: {e}")
            return None, None

        print(data.shape)
        
        # Splitting data into X and y
        try:
            X = data[:, :-1]
            y = data[:, -1].astype(int)
        except IndexError as e:
            print(f"Error when splitting data into X and y: {e}")
            return None, None

        return X, y


    @staticmethod
    # A helper function to segment data into sliding windows
    def segment_data(data, window_size, step_size):
        segments = []
        labels = []

        for start_pos in range(0, len(data) - window_size, step_size):
            end_pos = start_pos + window_size
            segment = [row[:-1] for row in data[start_pos:end_pos]]  # exclude the last column (label)
            segment_labels = [row[-1] for row in data[start_pos:end_pos]]  # only the last column (label)

            # Determine the majority label for this segment
            segment_label = max(set(segment_labels), key=segment_labels.count)

            segments.append(segment)
            labels.append(segment_label)

        return segments, labels

    @staticmethod
    def normalize_data(X_train, X_test):
        means = np.mean(X_train, axis=0)
        stds = np.std(X_train, axis=0)
        X_train_norm = (X_train - means) / stds
        X_test_norm = (X_test - means) / stds
        return X_train_norm, X_test_norm, means, stds

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

    def read_motion_accel_data(file_path):
        return np.genfromtxt(file_path, delimiter=' ', dtype=float, usecols=[0, 1, 2, 3])
    
    def read_motion_data(file_path):
        return np.genfromtxt(file_path, delimiter=' ', dtype=float, usecols=[0, 1, 2, 3, 7, 8, 9])
    
    def read_2023_label_file(file_path):
        return np.genfromtxt(file_path, delimiter=' ', dtype=int, usecols=[1])
    
    def read_2023_gps_file(file_path):
        return np.genfromtxt(file_path, delimiter=' ', dtype=int, usecols=[0, 4, 5, 7, 8])

    def read_label_file(file_path):
        return np.genfromtxt(file_path, delimiter=' ', dtype=int, usecols=[1])

    def apply_gaussian_filter(data, sigma=1):
        return gaussian_filter(data, sigma=sigma)
    
    def apply_savitzky_golay(data, window_length=5, polynomial_order=2):
        """
        Apply Savitzky-Golay filter to data.
        
        Parameters:
        - data: The input data (e.g., a list or numpy array)
        - window_length: The length of the filter window (should be an odd integer). Default is 5.
        - polynomial_order: The order of the polynomial used to fit the samples. Default is 2.
        
        Returns:
        - Smoothed data
        """
        return savgol_filter(data, window_length, polynomial_order)


    def compute_magnitude(d):
        return np.sqrt(d[0]**2 + d[1]**2 + d[2]**2)

    def compute_jerk(data):
        return [data[i+1] - data[i] for i in range(len(data)-1)] + [0]

    def compute_speed(point1, point2):
        distance = Preprocessing.haversine(point1["latitude"], point1["longitude"], point2["latitude"], point2["longitude"]) # In kilometers
        time = (point2["timestamp"] - point1["timestamp"]) / (1000 * 60 * 60) # Convert milliseconds to hours
        speed = distance / time # In km/h
        return speed
    
    def calculate_bearing(lat1, lon1, lat2, lon2):
        """
        Calculate the bearing between two points on the earth specified in decimal degrees.
        Returns bearing in degrees (0 to 360).
        """
        dLon = math.radians(lon2 - lon1)
        y = math.sin(dLon) * math.cos(math.radians(lat2))
        x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(dLon)
        initial_bearing = math.degrees(math.atan2(y, x))
        # Normalize bearing to lie between 0-360 degrees
        compass_bearing = (initial_bearing + 360) % 360
        return compass_bearing
    
    def compute_speed_and_bearing(timestamps, latitudes, longitudes):
        speeds = []
        bearings = []

        for i in tqdm(range(1, len(timestamps)), desc="Computing speeds and bearings"):
            dt = (timestamps[i] - timestamps[i-1]) / 1000  # Convert from milliseconds to seconds

            if dt == 0:  # Check for zero denominator
                speed = -1
                bearing = bearings[-1] if bearings else -1  # Use the last value or default to 0
            else:
                distance = Preprocessing.haversine(latitudes[i-1], longitudes[i-1], latitudes[i], longitudes[i])
                
                # Speed in m/s
                speed = distance / dt

                bearing = Preprocessing.calculate_bearing(latitudes[i-1], longitudes[i-1], latitudes[i], longitudes[i])
            
            speeds.append(speed)
            bearings.append(bearing)
        
        # Append the first value to the front of speeds and bearings lists to make their length equal to the length of timestamps
        speeds.insert(0, speeds[0])
        bearings.insert(0, bearings[0])

        return speeds, bearings
    
    # Haversine function to calculate distance between two lat-lon points
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
    
    def convert_to_seconds(timestamps):
        if isinstance(timestamps, (list, np.ndarray)):
            return [int(t/1000) for t in timestamps]
        else:
            return int(timestamps/1000)

    def create_gps_timestamps_dict(gps_timestamps):
        """Convert GPS timestamps to seconds and create a lookup dictionary."""
        gps_timestamps_seconds = Preprocessing.convert_to_seconds(gps_timestamps)
        gps_timestamps_dict = {time: idx for idx, time in enumerate(gps_timestamps_seconds)}
        return gps_timestamps_dict

    def find_nearest_gps_index(acc_timestamp, gps_timestamps_dict):
        """Find matched indices of accelerometer data in the GPS data."""    
        # Convert acc_timestamp from milliseconds to seconds
        acc_timestamp_seconds = int(acc_timestamp / 1000)

        # Check if acc_timestamp_seconds matches any gps_timestamp
        if acc_timestamp_seconds in gps_timestamps_dict:
            return gps_timestamps_dict[acc_timestamp_seconds]
        else:
            return None


    # MARK: - DATA FROM PHONE LOCATIONS
    @staticmethod
    def data_from_phone_locations(locations,is_validation=False):
        is_short = False
        all_data = []
        print(os.getcwd())
        print(locations)
        
    
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"{current_time} - Start processing data")

        for location in locations:
            # Create the filename
            filename_suffix = "_validation" if is_validation else ""
            filename_suffix += "_short" if is_short else ""
            filename = f"{location}_processed_mag_gyro_accel_gps{filename_suffix}.txt"

            # Check if file already exists
            if os.path.exists(filename):
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"{current_time} - Looking for existing file")
                print(f"{filename} already exists. Loading data from the file.")
                with open(filename, 'r') as f:
                    reader = csv.reader(f)
                    all_data = list(reader)
                    return all_data

            # locations_folder = os.path.join("files/SHL-2023", location)
            locations_folder = os.path.join("files/SHL-2023", location)

            # Use the short file names if is_short is True
            mag_suffix = "_short.txt" if is_short else ".txt"
            accel_suffix = "_short.txt" if is_short else ".txt"
            gyro_suffix = "_short.txt" if is_short else ".txt"
            
            label_suffix = "_short.txt" if is_short else ".txt"

            if is_validation:
                mag_file = os.path.join("files/SHL-2023", "validate", location, "Mag" + mag_suffix)
                accel_file = os.path.join("files/SHL-2023", "validate", location, "Acc" + accel_suffix)
                gyro_file = os.path.join("files/SHL-2023", "validate", location, "Gyr" + gyro_suffix)
                label_file = os.path.join("files/SHL-2023", "validate", location, "Label" + label_suffix)
                # gps_file = os.path.join("files/SHL-2023", "validate", location, "Location.txt")
            else:
                mag_file = os.path.join(locations_folder, "Mag" + mag_suffix)
                accel_file = os.path.join(locations_folder, "Acc" + accel_suffix)
                gyro_file = os.path.join(locations_folder, "Gyr" + gyro_suffix)
                label_file = os.path.join(locations_folder, "Label" + label_suffix)
                # gps_file = os.path.join(locations_folder, "Location.txt")

            try:
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"{current_time} - Loading: ",(mag_file))
                mag = Preprocessing.read_motion_accel_data(mag_file)
                print("Magnetometer: ",len(mag))
                # Downsample from 100Hz to 20Hz
                mag = mag[::5]  # Take every 5th label
            except FileNotFoundError:
                print(f"Warning: {mag_file} not found. Skipping...")
                continue

            try:
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"{current_time} - Loading: ",(accel_file))
                accel = Preprocessing.read_motion_accel_data(accel_file)
                print("Accelerometer: ",len(accel))
                # Downsample from 100Hz to 20Hz
                accel = accel[::5]  # Take every 5th label
            except FileNotFoundError:
                print(f"Warning: {accel_file} not found. Skipping...")
                continue

            try:
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"{current_time} - Loading: ",(gyro_file))
                gyro = Preprocessing.read_motion_accel_data(gyro_file)
                print("Gyro: ",len(gyro))
                # Downsample from 100Hz to 20Hz
                gyro = gyro[::5]  # Take every 5th label
            except FileNotFoundError:
                print(f"Warning: {gyro_file} not found. Skipping...")
                continue

            try:
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"{current_time} - Loading: ",(label_file))
                labels = Preprocessing.read_2023_label_file(label_file)
                print("Labels: ",len(labels))
                # Downsample from 100Hz to 20Hz
                labels = labels[::5]  # Take every 5th label
                unique_modes = set(labels)
                print("Unique modes before filtering:", unique_modes)
            except FileNotFoundError:
                print(f"Warning: {label_file} not found. Skipping...")
                continue

            # try:
            #     current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            #     print(f"{current_time} - Loading: ",(gps_file))
            #     gps = Preprocessing.read_2023_gps_file(gps_file)
            #     gps = gps[::5]  # Take every 5th location
            # except FileNotFoundError:
            #     print(f"Warning: {gps_file} not found. Skipping...")
            #     continue

            if len(accel) != len(mag) or len(accel) != len(gyro):
                print(f"Warning: Data lengths do not match for location {location}. Skipping...")
                continue

            acc_data = [row[0:4] for row in accel]
            if not acc_data:
                print("acc_data is empty!, Skipping...")
                continue

            gyro_data = [row[1:4] for row in gyro]
            if not gyro_data:
                print("gyro_data is empty!, Skipping...")
                continue

            mag_data = [row[1:4] for row in mag] 
            if not mag_data:
                print("mag_data is empty!, Skipping...")
                continue

            # gps_data = [row for row in gps] 
            # if not gps_data:
            #     print("gps_data is empty!, Skipping...")
            #     continue

            
            # Extract each dimension for accelerometer, magnetometer, gyro and gps
            acc_timestamps = [row[0] for row in acc_data]
            acc_data_x     = [row[1] for row in acc_data]
            acc_data_y     = [row[2] for row in acc_data]
            acc_data_z     = [row[3] for row in acc_data]

            mag_data_x = [row[0] for row in mag_data]
            mag_data_y = [row[1] for row in mag_data]
            mag_data_z = [row[2] for row in mag_data]

            gyr_data_x = [row[0] for row in gyro_data]
            gyr_data_y = [row[1] for row in gyro_data]
            gyr_data_z = [row[2] for row in gyro_data]

            # gps_data_timestamps = [row[0] for row in gps_data]
            # # gps_data_latitude   = [row[1] for row in gps_data]
            # # gps_data_longitude  = [row[2] for row in gps_data] 
            # gps_data_speed      = [row[3] for row in gps_data]
            # gps_data_bearing    = [row[4] for row in gps_data]
            

            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"{current_time} - Smoothing data using savitzky_golay...")
            # current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # print(f"{current_time} - Applying filters...")
            # Apply the filter to accelerometer data
            smoothed_acc_x = Preprocessing.apply_savitzky_golay(acc_data_x)
            smoothed_acc_y = Preprocessing.apply_savitzky_golay(acc_data_y)
            smoothed_acc_z = Preprocessing.apply_savitzky_golay(acc_data_z)

            # Apply the filter to magnetometer data
            smoothed_mag_x = Preprocessing.apply_savitzky_golay(mag_data_x)
            smoothed_mag_y = Preprocessing.apply_savitzky_golay(mag_data_y)
            smoothed_mag_z = Preprocessing.apply_savitzky_golay(mag_data_z)

            # Apply the filter to magnetometer data
            smoothed_gyr_x = Preprocessing.apply_savitzky_golay(gyr_data_x)
            smoothed_gyr_y = Preprocessing.apply_savitzky_golay(gyr_data_y)
            smoothed_gyr_z = Preprocessing.apply_savitzky_golay(gyr_data_z)

            # Group the smoothed data back together
            smoothed_acc_data = list(zip(smoothed_acc_x, smoothed_acc_y, smoothed_acc_z))
            smoothed_mag_data = list(zip(smoothed_mag_x, smoothed_mag_y, smoothed_mag_z))
            smoothed_gyr_data = list(zip(smoothed_gyr_x, smoothed_gyr_y, smoothed_gyr_z))

            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"{current_time} - Computing jerks and magnitudes...")
            # Compute jerk
            acc_jerks_x = Preprocessing.compute_jerk(smoothed_acc_x)
            acc_jerks_y = Preprocessing.compute_jerk(smoothed_acc_y)
            acc_jerks_z = Preprocessing.compute_jerk(smoothed_acc_z)

            mag_jerks_x = Preprocessing.compute_jerk(smoothed_mag_x)
            mag_jerks_y = Preprocessing.compute_jerk(smoothed_mag_y)
            mag_jerks_z = Preprocessing.compute_jerk(smoothed_mag_z)

            gyr_jerks_x = Preprocessing.compute_jerk(smoothed_gyr_x)
            gyr_jerks_y = Preprocessing.compute_jerk(smoothed_gyr_y)
            gyr_jerks_z = Preprocessing.compute_jerk(smoothed_gyr_z)
            
            # Compute magnitude
            acc_magnitudes = [Preprocessing.compute_magnitude(acc) for acc in acc_data]
            mag_magnitudes = [Preprocessing.compute_magnitude(mag) for mag in mag_data]
            gyr_magnitudes = [Preprocessing.compute_magnitude(gyr) for gyr in gyro_data]
            
            # gps_timestamps_dict = Preprocessing.create_gps_timestamps_dict(gps_data_timestamps)

            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"{current_time} - Putting all together...")
            for idx, (acc_values, mag_values, gyr_values) in tqdm(enumerate(zip(acc_data, mag_data, gyro_data)), total=len(acc_data), desc="Processing last step"):
                mode = labels[idx]

                acc_timestamp = acc_timestamps[idx]  # Assuming you have a list called acc_timestamps which holds the timestamps for the accelerometer data

                # nearest_gps_idx = Preprocessing.find_nearest_gps_index(acc_timestamp, gps_data_timestamps)


                # if nearest_gps_idx is not None:
                #     current_speed = gps_data_speed[nearest_gps_idx]
                #     current_bearing = gps_data_bearing[nearest_gps_idx]
                # else:
                #     current_speed = -1
                #     current_bearing = -1

                if mode != 0:
                    # Append the data in the desired order
                    row = (
                        # [acc_timestamp] + # Timestamp
                        # [current_speed] + # Speed
                        # [current_bearing] + # Bearing
                        list(acc_values) +  # Acceleration x,y,z 
                        [acc_jerks_x[idx], acc_jerks_y[idx], acc_jerks_z[idx]] +  # Acceleration jerk x,y,z 
                        [acc_magnitudes[idx]] +  # Acceleration magnitude
                        list(mag_values) +  # Magnetic x,y,z 
                        [mag_jerks_x[idx], mag_jerks_y[idx], mag_jerks_z[idx]] +  # Magnetic jerk x,y,z 
                        [mag_magnitudes[idx]] +  # Magnetic magnitude 
                        list(gyr_values) + # Gyro x,y,z
                        [gyr_jerks_x[idx], gyr_jerks_y[idx], gyr_jerks_z[idx]] + # Gyro jerk x,y,z
                        [gyr_magnitudes[idx]] + # Gyro magnitude
                        [mode]
                    )
                    all_data.append(row)


        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"{current_time} - Finished processing data")
        print(len(all_data))

        # with open(filename, 'w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerows(all_data)
        
        # print(f"Data saved to {filename}")

        return all_data
    
    def compute_magnitude(arrays):
        squares = [np.square(array) for array in arrays]
        sum_of_squares = np.sum(squares, axis=0)
        return np.sqrt(sum_of_squares)

    def compute_jerk(data):
        return [data[i+1] - data[i] for i in range(len(data)-1)] + [0]

    @staticmethod
    def load_and_extract_features(file_path):
        data = np.genfromtxt(file_path, delimiter=',', dtype=str)
        timestamp = data[:, 0].astype(float)  # This is the new timestamp column
        speed = data[:, 1].astype(float)
        course = data[:, 2].astype(float)
        x = data[:, 3].astype(float)
        y = data[:, 4].astype(float)
        z = data[:, 5].astype(float)
        mx = data[:, 6].astype(float)
        my = data[:, 7].astype(float)
        mz = data[:, 8].astype(float)
        modes = data[:, -1]
        
        return timestamp, speed, course, x, y, z, mx, my, mz, modes

    def compute_statistics(data):
        return np.mean(data), np.std(data)

    def normalize_data(data, mean, std):
        # To handle zero standard deviation
        std = std if std != 0 else 1e-10  # 1e-10 or some small number to avoid division by zero
        return (data - mean) / std


    def check_data_types(*arrays):
        for arr in arrays:
            if not np.issubdtype(arr.dtype, np.number):
                print(f"Found non-numeric data: {arr[arr != arr.astype(float).astype(str)]}")

    def preprocess_data(speed, course, x, y, z, jerk_ax, jerk_ay, jerk_az, acc_magnitude, mx, my, mz, jerk_mx, jerk_my, jerk_mz, mag_magnitude):
        mean_speed, std_speed = Preprocessing.compute_statistics(speed)
        mean_course, std_course = Preprocessing.compute_statistics(course)
        mean_x, std_x = Preprocessing.compute_statistics(x)
        mean_y, std_y = Preprocessing.compute_statistics(y)
        mean_z, std_z = Preprocessing.compute_statistics(z)
        mean_mx, std_mx = Preprocessing.compute_statistics(mx)
        mean_my, std_my = Preprocessing.compute_statistics(my)
        mean_mz, std_mz = Preprocessing.compute_statistics(mz)
        mean_acc_magnitude, std_acc_magnitude = Preprocessing.compute_statistics(acc_magnitude)
        mean_mag_magnitude, std_mag_magnitude = Preprocessing.compute_statistics(mag_magnitude)

        # Add the new jerk statistics
        mean_jerk_ax, std_jerk_ax = Preprocessing.compute_statistics(jerk_ax)
        mean_jerk_ay, std_jerk_ay = Preprocessing.compute_statistics(jerk_ay)
        mean_jerk_az, std_jerk_az = Preprocessing.compute_statistics(jerk_az)
        mean_jerk_mx, std_jerk_mx = Preprocessing.compute_statistics(jerk_mx)
        mean_jerk_my, std_jerk_my = Preprocessing.compute_statistics(jerk_my)
        mean_jerk_mz, std_jerk_mz = Preprocessing.compute_statistics(jerk_mz)
        
        # This part was repeated, so removing the redundant calculations
        normalized_speed = Preprocessing.normalize_data(speed, mean_speed, std_speed)
        normalized_course = Preprocessing.normalize_data(course, mean_course, std_course)
        # normalized_speed = speed
        # normalized_course = course
        normalized_x = Preprocessing.normalize_data(x, mean_x, std_x)
        normalized_y = Preprocessing.normalize_data(y, mean_y, std_y)
        normalized_z = Preprocessing.normalize_data(z, mean_z, std_z)
        normalized_mx = Preprocessing.normalize_data(mx, mean_mx, std_mx)
        normalized_my = Preprocessing.normalize_data(my, mean_my, std_my)
        normalized_mz = Preprocessing.normalize_data(mz, mean_mz, std_mz)
        normalized_acc_magnitude = Preprocessing.normalize_data(acc_magnitude, mean_acc_magnitude, std_acc_magnitude)
        normalized_mag_magnitude = Preprocessing.normalize_data(mag_magnitude, mean_mag_magnitude, std_mag_magnitude)
        normalized_jerk_ax = Preprocessing.normalize_data(jerk_ax, mean_jerk_ax, std_jerk_ax)
        normalized_jerk_ay = Preprocessing.normalize_data(jerk_ay, mean_jerk_ay, std_jerk_ay)
        normalized_jerk_az = Preprocessing.normalize_data(jerk_az, mean_jerk_az, std_jerk_az)
        normalized_jerk_mx = Preprocessing.normalize_data(jerk_mx, mean_jerk_mx, std_jerk_mx)
        normalized_jerk_my = Preprocessing.normalize_data(jerk_my, mean_jerk_my, std_jerk_my)
        normalized_jerk_mz = Preprocessing.normalize_data(jerk_mz, mean_jerk_mz, std_jerk_mz)

        features = np.column_stack((normalized_speed, normalized_course, normalized_x, normalized_y, normalized_z, normalized_jerk_ax, normalized_jerk_ay, normalized_jerk_az, normalized_acc_magnitude, 
                                    normalized_mx, normalized_my, normalized_mz, normalized_jerk_mx, normalized_jerk_my, normalized_jerk_mz, normalized_mag_magnitude))    
            
        statistics = {
            'mean_speed': mean_speed, 'std_speed': std_speed,
            'mean_course': mean_course, 'std_course': std_course,
            'mean_x': mean_x, 'std_x': std_x,
            'mean_y': mean_y, 'std_y': std_y,
            'mean_z': mean_z, 'std_z': std_z,
            'mean_mx': mean_mx, 'std_mx': std_mx,
            'mean_my': mean_my, 'std_my': std_my,
            'mean_mz': mean_mz, 'std_mz': std_mz,
            'mean_acc_magnitude': mean_acc_magnitude, 
            'std_acc_magnitude': std_acc_magnitude,
            'mean_mag_magnitude': mean_mag_magnitude, 
            'std_mag_magnitude': std_mag_magnitude,
            'mean_jerk_ax': mean_jerk_ax, 'std_jerk_ax': std_jerk_ax,
            'mean_jerk_ay': mean_jerk_ay, 'std_jerk_ay': std_jerk_ay,
            'mean_jerk_az': mean_jerk_az, 'std_jerk_az': std_jerk_az,
            'mean_jerk_mx': mean_jerk_mx, 'std_jerk_mx': std_jerk_mx,
            'mean_jerk_my': mean_jerk_my, 'std_jerk_my': std_jerk_my,
            'mean_jerk_mz': mean_jerk_mz, 'std_jerk_mz': std_jerk_mz
        }
        
        return features, statistics
    
    def preprocess_data_no_jerks(timestamp, speed, course, x, y, z, mx, my, mz, acc_magnitude, mag_magnitude):
        mean_timestamp, std_timestamp = Preprocessing.compute_statistics(timestamp)
        mean_speed, std_speed = Preprocessing.compute_statistics(speed)
        mean_course, std_course = Preprocessing.compute_statistics(course)
        mean_x, std_x = Preprocessing.compute_statistics(x)
        mean_y, std_y = Preprocessing.compute_statistics(y)
        mean_z, std_z = Preprocessing.compute_statistics(z)
        mean_mx, std_mx = Preprocessing.compute_statistics(mx)
        mean_my, std_my = Preprocessing.compute_statistics(my)
        mean_mz, std_mz = Preprocessing.compute_statistics(mz)
        mean_acc_magnitude, std_acc_magnitude = Preprocessing.compute_statistics(acc_magnitude)
        mean_mag_magnitude, std_mag_magnitude = Preprocessing.compute_statistics(mag_magnitude)
        
        normalized_speed = Preprocessing.normalize_data(speed, mean_speed, std_speed)
        normalized_course = Preprocessing.normalize_data(course, mean_course, std_course)

        normalized_timestamp = Preprocessing.normalize_data(timestamp, mean_timestamp, std_timestamp)
        normalized_x = Preprocessing.normalize_data(x, mean_x, std_x)
        normalized_y = Preprocessing.normalize_data(y, mean_y, std_y)
        normalized_z = Preprocessing.normalize_data(z, mean_z, std_z)
        normalized_mx = Preprocessing.normalize_data(mx, mean_mx, std_mx)
        normalized_my = Preprocessing.normalize_data(my, mean_my, std_my)
        normalized_mz = Preprocessing.normalize_data(mz, mean_mz, std_mz)


        features = np.column_stack((normalized_timestamp, normalized_speed, normalized_course, normalized_x, normalized_y, normalized_z, 
                                    normalized_mx, normalized_my, normalized_mz, acc_magnitude, mag_magnitude))    
            
        statistics = {
            'mean_timestamp': mean_timestamp, 'std_timestamp': std_timestamp,
            'mean_speed': mean_speed, 'std_speed': std_speed,
            'mean_course': mean_course, 'std_course': std_course,
            'mean_x': mean_x, 'std_x': std_x,
            'mean_y': mean_y, 'std_y': std_y,
            'mean_z': mean_z, 'std_z': std_z,
            'mean_mx': mean_mx, 'std_mx': std_mx,
            'mean_my': mean_my, 'std_my': std_my,
            'mean_mz': mean_mz, 'std_mz': std_mz,
            'mean_acc_magnitude': mean_acc_magnitude, 
            'std_acc_magnitude': std_acc_magnitude,
            'mean_mag_magnitude': mean_mag_magnitude, 
            'std_mag_magnitude': std_mag_magnitude,
        }
        
        return features, statistics
    
    @staticmethod
    def preprocess_test_data(speed, course, x, y, z, jerk_ax, jerk_ay, jerk_az, acc_magnitude, mx, my, mz, jerk_mx, jerk_my, jerk_mz, mag_magnitude, statistics):    

        normalized_speed = Preprocessing.normalize_data(speed, statistics["mean_speed"], statistics["std_speed"])
        normalized_course = Preprocessing.normalize_data(course, statistics["mean_course"], statistics["std_course"])
        normalized_x = Preprocessing.normalize_data(x, statistics["mean_x"], statistics["std_x"])
        normalized_y = Preprocessing.normalize_data(y, statistics["mean_y"], statistics["std_y"])
        normalized_z = Preprocessing.normalize_data(z, statistics["mean_z"], statistics["std_z"])
        normalized_mx = Preprocessing.normalize_data(mx, statistics["mean_mx"], statistics["std_mx"])
        normalized_my = Preprocessing.normalize_data(my, statistics["mean_my"], statistics["std_my"])
        normalized_mz = Preprocessing.normalize_data(mz, statistics["mean_mz"], statistics["std_mz"])
        normalized_acc_magnitude = Preprocessing.normalize_data(acc_magnitude, statistics["mean_acc_magnitude"], statistics["std_acc_magnitude"])
        normalized_mag_magnitude = Preprocessing.normalize_data(mag_magnitude, statistics["mean_mag_magnitude"], statistics["std_mag_magnitude"])
        normalized_jerk_ax = Preprocessing.normalize_data(jerk_ax, statistics["mean_jerk_ax"], statistics["std_jerk_ax"])
        normalized_jerk_ay = Preprocessing.normalize_data(jerk_ay, statistics["mean_jerk_ay"], statistics["std_jerk_ay"])
        normalized_jerk_az = Preprocessing.normalize_data(jerk_az, statistics["mean_jerk_az"], statistics["std_jerk_az"])
        normalized_jerk_mx = Preprocessing.normalize_data(jerk_mx, statistics["mean_jerk_mx"], statistics["std_jerk_mx"])
        normalized_jerk_my = Preprocessing.normalize_data(jerk_my, statistics["mean_jerk_my"], statistics["std_jerk_my"])
        normalized_jerk_mz = Preprocessing.normalize_data(jerk_mz, statistics["mean_jerk_mz"], statistics["std_jerk_mz"])


        features = np.column_stack((normalized_speed, normalized_course, normalized_x, normalized_y, normalized_z, normalized_jerk_ax, normalized_jerk_ay, normalized_jerk_az, normalized_acc_magnitude, 
                                    normalized_mx, normalized_my, normalized_mz, normalized_jerk_mx, normalized_jerk_my, normalized_jerk_mz, normalized_mag_magnitude))    

        return features
    
    @staticmethod
    def preprocess_test_data_no_jerks(timestamp, speed, course, x, y, z, acc_magnitude, mx, my, mz, mag_magnitude, statistics):    

        normalized_speed = Preprocessing.normalize_data(speed, statistics["mean_speed"], statistics["std_speed"])
        normalized_course = Preprocessing.normalize_data(course, statistics["mean_course"], statistics["std_course"])
        normalized_x = Preprocessing.normalize_data(x, statistics["mean_x"], statistics["std_x"])
        normalized_y = Preprocessing.normalize_data(y, statistics["mean_y"], statistics["std_y"])
        normalized_z = Preprocessing.normalize_data(z, statistics["mean_z"], statistics["std_z"])
        normalized_mx = Preprocessing.normalize_data(mx, statistics["mean_mx"], statistics["std_mx"])
        normalized_my = Preprocessing.normalize_data(my, statistics["mean_my"], statistics["std_my"])
        normalized_mz = Preprocessing.normalize_data(mz, statistics["mean_mz"], statistics["std_mz"])
        normalized_acc_magnitude = Preprocessing.normalize_data(acc_magnitude, statistics["mean_acc_magnitude"], statistics["std_acc_magnitude"])
        normalized_mag_magnitude = Preprocessing.normalize_data(mag_magnitude, statistics["mean_mag_magnitude"], statistics["std_mag_magnitude"])


        features = np.column_stack((timestamp, normalized_speed, normalized_course, normalized_x, normalized_y, normalized_z,  normalized_acc_magnitude, 
                                    normalized_mx, normalized_my, normalized_mz, normalized_mag_magnitude))    

        return features

    def apply_savitzky_golay(data, window_length=5, polynomial_order=2):
        """
        Apply Savitzky-Golay filter to data.
        
        Parameters:
        - data: The input data (e.g., a list or numpy array)
        - window_length: The length of the filter window (should be an odd integer). Default is 5.
        - polynomial_order: The order of the polynomial used to fit the samples. Default is 2.
        
        Returns:
        - Smoothed data
        """
        return savgol_filter(data, window_length, polynomial_order)
    
    @staticmethod
    def apply_savitzky_golay_to_all(data, window_length=5, polyorder=2):
        """
        Apply Savitzky-Golay smoothing to each series in the provided data.
        
        :param data: List of 1D arrays representing different series.
        :param window_length: The length of the filter window (must be a positive odd integer).
        :param polyorder: The order of the polynomial used to fit the samples (must be less than window_length).
        :return: List of smoothed 1D arrays.
        """
        if any((window_length % 2 == 0) or (window_length <= polyorder) for _ in data):
            raise ValueError("window_length must be an odd positive integer and greater than polyorder.")
            
        smoothed_data = [savgol_filter(series, window_length, polyorder) for series in data]
        return smoothed_data

    @staticmethod
    def compute_magnitudes(smoothed_data):
        acc_magnitudes = [math.sqrt(x**2 + y**2 + z**2) for x, y, z in zip(*smoothed_data[:3])]
        mag_magnitudes = [math.sqrt(mx**2 + my**2 + mz**2) for mx, my, mz in zip(*smoothed_data[3:])]
        return acc_magnitudes, mag_magnitudes
