import os
import numpy as np
import time
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Concatenate, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter
from keras.layers import Reshape
from keras.layers import Input, Conv1D, MaxPooling1D, BatchNormalization, Bidirectional, LSTM, Dense
from keras.utils import to_categorical
from keras.layers import Dropout

import matplotlib.pyplot as plt
from preprocessing import Preprocessing
from models import Models
from transportation_mode import TransportationMode

users = ["User1", "User2", "User3"]
# users = ["UserTest"]
motion_files = ["Bag_Motion.txt", "Hips_Motion.txt", "Hand_Motion.txt", "Torso_Motion.txt"]
# motion_files = ["Hand_Motion.txt", "Hips_Motion.txt"]

# Load data from the text file
data = Preprocessing.data_for_classification_model(users=users,motion_files=motion_files)
data = np.array(data)

# Extract relevant information from the loaded data
modes = data[:, -1]  # transportation modes
timestamps = data[:, 0].astype(float)  # timestamps
x = data[:, 1].astype(float)  # x accel value
y = data[:, 2].astype(float)  # y accel
z = data[:, 3].astype(float)  # z accel
mx = data[:, 4].astype(float)  # mx magnetometer value
my = data[:, 5].astype(float)  # my magnetometer
mz = data[:, 6].astype(float)  # mz magnetometer


# Perform any necessary preprocessing steps
# For example, you can normalize the sensor values

# Normalize timestamp, speed, x, y, and z values
def normalize(array):
    mean = np.mean(array)
    std = np.std(array)
    normalized = (array - mean) / std
    return normalized, mean, std

# Perform normalization on the sensor values
normalized_timestamp, mean_timestamp, std_timestamp = normalize(timestamps)
normalized_x, mean_x, std_x = normalize(x)
normalized_y, mean_y, std_y = normalize(y)
normalized_z, mean_z, std_z = normalize(z)
normalized_mx, mean_mx, std_mx = normalize(mx)
normalized_my, mean_my, std_my = normalize(my)
normalized_mz, mean_mz, std_mz = normalize(mz)

# Encode transportation modes as numerical labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(modes)
num_classes = len(TransportationMode)

# Get the list of transportation mode labels
labels = label_encoder.classes_.tolist()

# Combine normalized sensor values into features
features = np.column_stack((normalized_timestamp, normalized_x, normalized_y, normalized_z, normalized_mx, normalized_my, normalized_mz))

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, encoded_labels, test_size=0.2)

# Define learning rate schedule function
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 10:
        lr *= 1e-1
    elif epoch > 20:
        lr *= 1e-2
    print('Learning rate: ', lr)
    return lr

input_dim = features.shape[1]

model = Models.create_classification_model(num_clases=num_classes, input_dim=input_dim, num_classes=num_classes)

# Define callbacks
early_stopping = EarlyStopping(patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint('../model/classification/trained_classification_model.h5', save_best_only=True)
lr_scheduler = LearningRateScheduler(lr_schedule)

# Compile and fit the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# One-hot encode the labels
train_labels = to_categorical(train_labels, num_classes=num_classes)
test_labels = to_categorical(test_labels, num_classes=num_classes)

# Save the label encoder
np.save('../model/classification/label_encoder.npy', label_encoder.classes_)

# Save the mean and standard deviation
np.save('../model/classification/mean.npy', [mean_timestamp, mean_x, mean_y, mean_z, mean_mx, mean_my, mean_mz])
np.save('../model/classification/std.npy', [std_timestamp, std_x, std_y, std_z, std_mx, std_my, std_mz])

history = model.fit(train_features, train_labels, epochs=20, batch_size=64, 
                    validation_data=(test_features, test_labels),
                    callbacks=[early_stopping, checkpoint, lr_scheduler])

# Save the trained model as a TensorFlow h5 file
model.save('../model/classification/trained_classification_model.h5')

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()

