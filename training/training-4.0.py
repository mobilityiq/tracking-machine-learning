import os
import numpy as np
import time
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate, Flatten
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers import Reshape
from keras.layers import Input, Conv1D, MaxPooling1D, BatchNormalization, Bidirectional, LSTM, Dense
from keras.utils import to_categorical


def apply_gaussian_filter(data, sigma=1):
    return gaussian_filter(data, sigma=sigma)

def spline_interpolate(data, original_sample_rate=100, desired_sample_rate=20):
    x = np.arange(len(data))
    s = UnivariateSpline(x, data, s=0)
    new_length = int(len(data) * desired_sample_rate / original_sample_rate)
    new_x = np.linspace(0, len(data) - 1, new_length)
    return s(new_x)


def read_motion_data(file_path):
    return np.genfromtxt(file_path, delimiter=' ', dtype=float, usecols=[0, 1, 2, 3, 7, 8, 9])

def read_label_file(file_path):
    return np.genfromtxt(file_path, delimiter=' ', dtype=int, usecols=[1])

def compute_magnitude(d):
    return np.sqrt(d[0]**2 + d[1]**2 + d[2]**2)

def compute_jerk(data):
    return [data[i+1] - data[i] for i in range(len(data)-1)] + [0]


def create_multichannel_model(input_shapes):
    channel_inputs = []
    conv_outputs = []

    for shape in input_shapes:
        channel_input = Input(shape=shape)
        channel_inputs.append(channel_input)

        # 1st Conv layer
        conv1 = Conv1D(32, 3, activation='relu')(channel_input)
        conv1 = BatchNormalization()(conv1)

        # 2nd Conv layer
        conv2 = Conv1D(64, 3, activation='relu')(conv1)
        conv2 = BatchNormalization()(conv2)

        # 3rd Conv layer
        conv3 = Conv1D(64, 3, activation='relu')(conv2)
        conv3 = BatchNormalization()(conv3)

        # 4th Conv layer
        conv4 = Conv1D(128, 3, activation='relu')(conv3)
        conv4 = BatchNormalization()(conv4)

        # 5th Conv layer
        conv5 = Conv1D(128, 3, activation='relu')(conv4)
        conv5 = BatchNormalization()(conv5)

        conv_outputs.append(Flatten()(conv5))

    concatenated = Concatenate()(conv_outputs)
    
    # BiLSTM Layer
    bi_lstm = Bidirectional(LSTM(128, return_sequences=True))(Reshape((-1, 1))(concatenated))
    bi_lstm = Flatten()(bi_lstm)

    # Fully Connected Layers
    fc1 = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(bi_lstm)
    fc2 = Dense(9, activation='softmax')(fc1)

    model = Model(channel_inputs, fc2)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model




early_stop = EarlyStopping(monitor='val_loss', patience=5)


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
users = ["User1"]
all_data = []
start_time = time.time()
# motion_files = ["Bag_Motion.txt", "Hips_Motion.txt", "Hand_Motion.txt", "Torso_Motion"]
motion_files = ["Bag_Motion.txt"]

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

        # Downsample the labels from 100Hz to 20Hz
        labels = labels[::5]  # Take every 5th label

        for motion_file in motion_files:
            motion_file_path = os.path.join(user_folder, date_folder, motion_file)

            if os.path.exists(motion_file_path):
                data = read_motion_data(motion_file_path)

                data = data[::5]

                print("Shape of data after reading:", data.shape)


                # Apply Spline Interpolation on each feature column
                # And downgrade the smaple from 100Hz to 20Hz
                # interpolated_data = np.zeros((int(data.shape[0] * 20 / 100), data.shape[1]))

                # for col in range(data.shape[1]):
                #     interpolated_data[:, col] = spline_interpolate(data[:, col])

                # print(f"Shape of interpolated data for column {col}:", interpolated_data[:, col].shape)


                # data = interpolated_data

                # # Compute the FFT of each feature column
                # fft_data = np.fft.fft(data, axis=0)

                # # Get frequency components for the data
                # frequencies = np.fft.fftfreq(data.shape[0])

                # # For visualization, you can plot the magnitude spectrum
                # import matplotlib.pyplot as plt
                # plt.plot(frequencies, np.abs(fft_data))
                # plt.xlabel('Frequency (Hz)')
                # plt.ylabel('Magnitude')
                # plt.title('Frequency Spectrum')
                # plt.show()

                # inverse_data = np.fft.ifft(fft_data, axis=0)

                # data = inverse_data

                timestamps = [row[0] for row in data]

                acc_data = [row[1:4] for row in data]
                mag_data = [row[4:7] for row in data]   

                # Compute jerk outside of the loop
                acc_jerks_x, acc_jerks_y, acc_jerks_z = compute_jerk([row[0] for row in acc_data]), compute_jerk([row[1] for row in acc_data]), compute_jerk([row[2] for row in acc_data])
                mag_jerks_x, mag_jerks_y, mag_jerks_z = compute_jerk([row[0] for row in mag_data]), compute_jerk([row[1] for row in mag_data]), compute_jerk([row[2] for row in mag_data])

                acc_magnitudes = [compute_magnitude(acc) for acc in acc_data]
                mag_magnitudes = [compute_magnitude(mag) for mag in mag_data]

                for idx, (timestamp, acc_values, mag_values) in enumerate(zip(timestamps, acc_data, mag_data)):
                    mode = labels[idx]
                    if mode != 0:
                        # Append the data in the desired order
                        all_data.append([timestamp] + 
                                        list(acc_values) + 
                                        [acc_jerks_x[idx], acc_jerks_y[idx], acc_jerks_z[idx]] + 
                                        [acc_magnitudes[idx]] + 
                                        [mag_jerks_x[idx], mag_jerks_y[idx], mag_jerks_z[idx]] + 
                                        [mag_magnitudes[idx]] + 
                                        list(mag_values) + 
                                        [mode])

                print(f"Data length after processing {motion_file}: {len(all_data)}")

    print(f"Time taken for data loading for {user}: {time.time() - data_loading_start:.2f} seconds")

print(f"Total time taken for all users: {time.time() - start_time:.2f} seconds")


print(all_data[:5])

# Process the all_data list to separate features and labels
all_data_np = np.array(all_data)

X = all_data_np[:, :-1]  # All columns except the last one
y = all_data_np[:, -1].astype(int)  # Last column

y = tf.keras.utils.to_categorical(y, num_classes=9)  # Convert labels to one-hot encoding

# Reshape the data
X = X.reshape(X.shape[0], X.shape[1], 1)

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the CNN-BiLSTM model
num_channels = 6  # 6 channels as described
input_shapes = [(X_train.shape[1], 1) for _ in range(num_channels)]  # Each channel has the same input shape

model = create_multichannel_model(input_shapes=input_shapes)

# For training, you need to adapt your data so that you provide each channel's input separately.
# Here I'm using the same data for each channel for demonstration purposes.
X_train_channels = [X_train for _ in range(num_channels)]
X_test_channels = [X_test for _ in range(num_channels)]

y_train_encoded = to_categorical(y_train, num_classes=9)
y_test_encoded = to_categorical(y_test, num_classes=9)

history = model.fit(
    x=X_train_channels,
    y=y_train_encoded,
    validation_data=(X_test_channels, y_test_encoded),
    epochs=20,
    batch_size=32,
    callbacks=[early_stop]
)


# Save the model if needed
model.save('cnn_bilstm_model.h5')
