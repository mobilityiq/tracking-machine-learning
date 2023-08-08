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
        conv1 = Conv1D(32, 3, activation='relu', padding='same')(channel_input)
        conv1 = BatchNormalization()(conv1)

        # 2nd Conv layer
        conv2 = Conv1D(64, 3, activation='relu', padding='same')(conv1)
        conv2 = BatchNormalization()(conv2)

        # 3rd Conv layer
        conv3 = Conv1D(64, 3, activation='relu', padding='same')(conv2)
        conv3 = BatchNormalization()(conv3)

        # 4th Conv layer
        conv4 = Conv1D(128, 3, activation='relu', padding='same')(conv3)
        conv4 = BatchNormalization()(conv4)

        # 5th Conv layer
        conv5 = Conv1D(128, 3, activation='relu', padding='same')(conv4)
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
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

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

                print(f"Data length after processing {motion_file}: {len(all_data)}")

    print(f"Time taken for data loading for {user}: {time.time() - data_loading_start:.2f} seconds")

print(f"Total time taken for all users: {time.time() - start_time:.2f} seconds")


print(all_data[:5])

# Process the all_data list to separate features and labels
all_data_np = np.array(all_data)

X = all_data_np[:, :-1]  # All columns except the last one
y = all_data_np[:, -1].astype(int)  # Last column

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Extract the respective data for each channel from training data
X_train_acc = X_train[:, :3]  # Acceleration x, y, z
X_train_jerk = X_train[:, 3:6]  # Jerk x, y, z
X_train_acc_mag = X_train[:, 6:7]  # Acceleration magnitude
X_train_mag_jerk = X_train[:, 7:10]  # Magnetic jerk x, y, z
X_train_mag_mag = X_train[:, 10:11]  # Magnetic magnitude
X_train_magnetic = X_train[:, 11:14]  # Magnetic field x, y, z

X_train_acc = X_train_acc[..., np.newaxis]
X_train_jerk = X_train_jerk[..., np.newaxis]
X_train_acc_mag = X_train_acc_mag[..., np.newaxis]
X_train_mag_jerk = X_train_mag_jerk[..., np.newaxis]
X_train_mag_mag = X_train_mag_mag[..., np.newaxis]
X_train_magnetic = X_train_magnetic[..., np.newaxis]

# Now that the X_train_... variables are defined, you can create the input_shapes list:
input_shapes = [
    X_train_acc.shape[1:], 
    X_train_jerk.shape[1:], 
    X_train_acc_mag.shape[1:], 
    X_train_mag_jerk.shape[1:], 
    X_train_mag_mag.shape[1:], 
    X_train_magnetic.shape[1:]
]

model = create_multichannel_model(input_shapes=input_shapes)

X_train_channels = [X_train_acc, X_train_jerk, X_train_acc_mag, X_train_mag_jerk, X_train_mag_mag, X_train_magnetic]

# Extract the respective data for each channel from test data
X_test_acc = X_test[:, :3]
X_test_jerk = X_test[:, 3:6]
X_test_acc_mag = X_test[:, 6:7]
X_test_mag_jerk = X_test[:, 7:10]
X_test_mag_mag = X_test[:, 10:11]
X_test_magnetic = X_test[:, 11:14]

X_test_channels = [X_test_acc, X_test_jerk, X_test_acc_mag, X_test_mag_jerk, X_test_mag_mag, X_test_magnetic]

# Convert labels to one-hot encoding
y_train_encoded = to_categorical(y_train, num_classes=9)
y_test_encoded = to_categorical(y_test, num_classes=9)

X_test_acc = X_test_acc[..., np.newaxis]
X_test_jerk = X_test_jerk[..., np.newaxis]
X_test_acc_mag = X_test_acc_mag[..., np.newaxis]
X_test_mag_jerk = X_test_mag_jerk[..., np.newaxis]
X_test_mag_mag = X_test_mag_mag[..., np.newaxis]
X_test_magnetic = X_test_magnetic[..., np.newaxis]


# Training the model
history = model.fit(
    x=X_train_channels,
    y=y_train_encoded,
    validation_data=(X_test_channels, y_test_encoded),
    epochs=10,
    batch_size=128,
    callbacks=[early_stop]
)

# Save the model
model.save('cnn_bilstm_model.h5')