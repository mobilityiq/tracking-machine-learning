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
from keras.layers import Dropout

import matplotlib.pyplot as plt
from preprocessing import Preprocessing
from models import Models

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


early_stop = EarlyStopping(monitor='val_loss', patience=5)

# users = ["User1", "User2", "User3"]
users = ["UserTest"]
motion_files = ["Bag_Motion.txt", "Hips_Motion.txt", "Hand_Motion.txt", "Torso_Motion.txt"]
# motion_files = ["Hand_Motion.txt", "Hips_Motion.txt"]

# Import the data
data = Preprocessing.data_for_cnn_bilstm(users=users, motion_files=motion_files)

# Process the all_data list to separate features and labels
all_data_np = np.array(data)

X = all_data_np[:, :-1]  # All columns except the last one
y = all_data_np[:, -1].astype(int)  # Last column

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

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

# Create the model
model = Models.create_multichannel_model(input_shapes=input_shapes)

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
model.save('../model/cnn-bi-lstm/cnn_bilstm_model.h5')