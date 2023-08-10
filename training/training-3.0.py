import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from enum import Enum
import coremltools as ct
import json
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical


# Define the transportation mode Enum
class TransportationMode(Enum):
    DRIVING = 'driving'
    CYCLING = 'cycling'
    TRAIN = 'train'
    BUS = 'bus'
    SUBWAY = 'metro'
    TRAM = 'tram'
    # ESCOOTER = 'e-scooter'

# Check if the data file is provided as a command-line argument
if len(sys.argv) < 2:
    print("Please provide the path to the data file as a command-line argument.")
    sys.exit(1)

# Get the data file path from the command-line argument
data_file = sys.argv[1]

# Load data from the text file
data = np.genfromtxt(data_file, delimiter=',', dtype=str)


# Extract relevant information from the loaded data
modes = data[:, -1]  # transportation modes
timestamps = data[:, 0].astype(float)  # timestamps
speed = data[:, 1].astype(float)  # speed
course = data[:, 2].astype(float) #course
x = data[:, 3].astype(float)  # x accel value
y = data[:, 4].astype(float)  # y accel
z = data[:, 5].astype(float)  # z accel
qx = data[:, 6].astype(float)  # qx quaternion value
qy = data[:, 7].astype(float)  # qy quaternion
qz = data[:, 8].astype(float)  # qz quaternion
qw = data[:, 9].astype(float)  # qw quaternion 


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
normalized_speed, mean_speed, std_speed = normalize(speed)
normalized_course, mean_course, std_course = normalize(course)
normalized_x, mean_x, std_x = normalize(x)
normalized_y, mean_y, std_y = normalize(y)
normalized_z, mean_z, std_z = normalize(z)
normalized_qx, mean_qx, std_qx = normalize(qx)
normalized_qy, mean_qy, std_qy = normalize(qy)
normalized_qz, mean_qz, std_qz = normalize(qz)
normalized_qw, mean_qw, std_qw = normalize(qw)

# Encode transportation modes as numerical labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(modes)
num_classes = len(TransportationMode)

# Get the list of transportation mode labels
labels = label_encoder.classes_.tolist()

# Combine normalized sensor values into features
features = np.column_stack((normalized_timestamp, normalized_speed, normalized_course, normalized_x, normalized_y, normalized_z, normalized_qx, normalized_qy, normalized_qz, normalized_qw))

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, encoded_labels, test_size=0.2)

def normalize(array):
    mean = np.mean(array)
    std = np.std(array)
    normalized = (array - mean) / std
    return normalized, mean, std

normalized_timestamp, mean_timestamp, std_timestamp = normalize(timestamps)
normalized_speed, mean_speed, std_speed = normalize(speed)
normalized_course, mean_course, std_course = normalize(course)
normalized_x, mean_x, std_x = normalize(x)
normalized_y, mean_y, std_y = normalize(y)
normalized_z, mean_z, std_z = normalize(z)
normalized_qx, mean_qx, std_qx = normalize(qx)
normalized_qy, mean_qy, std_qy = normalize(qy)
normalized_qz, mean_qz, std_qz = normalize(qz)
normalized_qw, mean_qw, std_qw = normalize(qw)

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

# Define model with L2 regularization
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(64, input_dim=input_dim, kernel_regularizer=regularizers.l2(0.01)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dense(num_classes))
model.add(tf.keras.layers.Activation('softmax'))


# Define callbacks
early_stopping = EarlyStopping(patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint('model.h5', save_best_only=True)
lr_scheduler = LearningRateScheduler(lr_schedule)

# Compile and fit the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# One-hot encode the labels
train_labels = to_categorical(train_labels, num_classes=num_classes)
test_labels = to_categorical(test_labels, num_classes=num_classes)

# Save the label encoder
np.save('../model/3.0/label_encoder.npy', label_encoder.classes_)

# Save the mean and standard deviation
np.save('../model/3.0/mean.npy', [mean_timestamp, mean_speed, mean_course, mean_x, mean_y, mean_z, mean_qx, mean_qy, mean_qz, mean_qw])
np.save('../model/3.0/std.npy', [std_timestamp, std_speed, std_course, std_x, std_y, std_z, std_qx, std_qy, std_qz, std_qw])

history = model.fit(train_features, train_labels, epochs=20, batch_size=64, 
                    validation_data=(test_features, test_labels),
                    callbacks=[early_stopping, checkpoint, lr_scheduler])

# Save the trained model as a TensorFlow h5 file
model.save('../model/3.0/trained_model-3.0.h5')

# Create a dictionary to hold the metadata
metadata = {
    'mean': [mean_timestamp, mean_speed, mean_course, mean_x, mean_y, mean_z, mean_qx, mean_qy, mean_qz, mean_qw],
    'std': [std_timestamp, std_speed, std_course, std_x, std_y, std_z, std_qx, std_qy, std_qz, std_qw],
    'labels': ['driving','cycling','train','bus','metro', 'tram']
}

# Convert the metadata dictionary to JSON string
metadata_json = json.dumps(metadata)

# Save the metadata as a JSON file
with open('../model/3.0/metadata.json', 'w') as f:
    f.write(metadata_json)

# Convert the model to Core ML format with a single input
# input_shape = (1, features.shape[1])
# input_feature = ct.TensorType(shape=input_shape)

# coreml_model = ct.convert(model, inputs=[input_feature], source='tensorflow')
coreml_model = ct.convert(model)

# Add the metadata to the model as user-defined metadata
coreml_model.user_defined_metadata['preprocessing_metadata'] = metadata_json

# Set the prediction_type to "probability"
coreml_model.user_defined_metadata['prediction_type'] = 'probability'

# Save the Core ML model
coreml_model.save('../model/3.0/TransitModePredictor.mlmodel')

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