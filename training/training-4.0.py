import os
import numpy as np
import csv
import time
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, BatchNormalization, Concatenate, Bidirectional, LSTM, Dense, Dropout, Flatten

def read_motion_data(file_path):
    return np.genfromtxt(file_path, delimiter=' ', dtype=float, usecols=[0, 1, 2, 3, 10, 11, 12, 13])[::5]

def read_label_file(file_path):
    return np.genfromtxt(file_path, delimiter=' ', dtype=int, usecols=[1])

def compute_magnitude(d):
    return np.sqrt(d[0]**2 + d[1]**2 + d[2]**2)

def compute_jerk(data, timestamps):
    jerk_data = []
    for i in range(len(data) - 1):
        delta_t = timestamps[i + 1] - timestamps[i]
        jerk = (data[i + 1] - data[i]) / delta_t
        jerk_data.append(jerk)
    return jerk_data

def create_model(input_shape):
    # Input
    inputs = Input(shape=input_shape)

    # CNN Layers
    conv1 = Conv1D(32, 15, activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)

    conv2 = Conv1D(64, 10, activation='relu', padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)

    conv3 = Conv1D(64, 10, activation='relu', padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)

    conv4 = Conv1D(128, 5, activation='relu', padding='same')(conv3)
    conv4 = BatchNormalization()(conv4)

    conv5 = Conv1D(128, 5, activation='relu', padding='same')(conv4)
    conv5 = BatchNormalization()(conv5)

    concatenated = Flatten()(conv5)

    # BiLSTM Layer
    reshaped = tf.keras.layers.Reshape((1, 384))(concatenated)
    bi_lstm = Bidirectional(LSTM(128, return_sequences=True))(reshaped)
    bi_lstm = Flatten()(bi_lstm)

    # Fully Connected Layers
    fc1 = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(bi_lstm)
    fc2 = Dense(9, activation='softmax')(fc1)  # 9 classes for transportation modes

    model = Model(inputs, fc2)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    return model


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

# users = ["User1", "User2", "User3"]
users = ["User0"]
all_data = []
start_time = time.time()
motion_files = ["Bag_Motion.txt", "Hips_Motion.txt", "Hand_Motion.txt", "Torso_Motion"]
# motion_files = ["Bag_Motion.txt"]

for user in users:
    data_loading_start = time.time()
    user_folder = os.path.join("files", user)
    dates_folders = [folder for folder in os.listdir(user_folder) if not folder.startswith('.')]

    for date_folder in dates_folders:
        label_file_path = os.path.join(user_folder, date_folder, "Label.txt")

        try:
            labels = read_label_file(label_file_path)
        except FileNotFoundError:
            print(f"Warning: {label_file_path} not found. Skipping...")
            continue

        for motion_file in motion_files:
            motion_file_path = os.path.join(user_folder, date_folder, motion_file)

            if os.path.exists(motion_file_path):
                data = read_motion_data(motion_file_path)

                timestamps = [row[0] for row in data]
                acc_data = [row[1:4] for row in data]
                mag_data = [row[4:7] for row in data]

                acc_magnitudes = [compute_magnitude(d) for d in acc_data]
                mag_magnitudes = [compute_magnitude(d) for d in mag_data]
                
                acc_jerks = compute_jerk(acc_magnitudes, timestamps)
                mag_jerks = compute_jerk(mag_magnitudes, timestamps)

                # for idx, row in enumerate(data[:-1]):
                #     if np.isnan(row).any():
                #         continue
                    
                #     # Check for NaN values in the computed jerks
                #     if np.isnan(acc_jerks[idx]).any() or np.isnan(mag_jerks[idx]).any():
                #         print("Skipping current row due to NaN values in jerks.")
                #         continue

                #     mode = labels[idx]
                #     if mode != 0:
                #         all_data.append(list(row) + [acc_jerks[idx], mag_jerks[idx], mode])

                for idx, timestamp in enumerate(timestamps[:-1]):
                    # Ensure jerk calculations are not NaN
                    if not (np.isnan(acc_jerks[idx]) or np.isnan(mag_jerks[idx])):
                        mode = labels[idx]
                        if mode != 0:
                            all_data.append([timestamp, acc_jerks[idx], mag_jerks[idx], mode])

    print(f"Time taken for data loading for {user}: {time.time() - data_loading_start:.2f} seconds")

print(f"Total time taken for all users: {time.time() - start_time:.2f} seconds")



# Process the all_data list to separate features and labels
all_data_np = np.array(all_data)


X = all_data_np[:, :-1]
y = all_data_np[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=9)  # Convert labels to one-hot encoding

# Reshape the data
X = X.reshape(X.shape[0], X.shape[1], 1)

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the CNN-BiLSTM model
input_shape = (X_train.shape[1], 1)
model = create_model(input_shape)

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Save the model if needed
model.save('cnn_bilstm_model.h5')
