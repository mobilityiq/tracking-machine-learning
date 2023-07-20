import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from enum import Enum
import coremltools as ct
import json

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

# Move the first column to the end of each line
data = np.roll(data, -1, axis=1)

# Extract relevant information from the loaded data
modes = data[:, -1]  # transportation modes
timestamps = data[:, 0].astype(float)  # timestamps
speed = data[:, 1].astype(float)  # speed
x = data[:, 2].astype(float)  # x coordinate
y = data[:, 3].astype(float)  # y coordinate
z = data[:, 4].astype(float)  # z coordinate

# Perform any necessary preprocessing steps
# For example, you can normalize the sensor values

# Normalize timestamp, speed, x, y, and z values
mean_timestamp = np.mean(timestamps)
std_timestamp = np.std(timestamps)
normalized_timestamp = (timestamps - mean_timestamp) / std_timestamp

mean_speed = np.mean(speed)
std_speed = np.std(speed)
normalized_speed = (speed - mean_speed) / std_speed

mean_x = np.mean(x)
std_x = np.std(x)
normalized_x = (x - mean_x) / std_x

mean_y = np.mean(y)
std_y = np.std(y)
normalized_y = (y - mean_y) / std_y

mean_z = np.mean(z)
std_z = np.std(z)
normalized_z = (z - mean_z) / std_z

# Encode transportation modes as numerical labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(modes)
num_classes = len(TransportationMode)

# Get the list of transportation mode labels
labels = label_encoder.classes_.tolist()

# Combine normalized sensor values into features
features = np.column_stack((normalized_timestamp, normalized_speed, normalized_x, normalized_y, normalized_z))

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, encoded_labels, test_size=0.2)

# Define the model architecture
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(64, activation='relu', input_shape=(features.shape[1],)),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(num_classes, activation='softmax')
# ])

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(features.shape[1], 1)),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Train the model
model.fit(train_features, train_labels, epochs=10, batch_size=32, validation_data=(test_features, test_labels))

# Save the trained model as a TensorFlow h5 file
model.save('../model/trained_model-2.0.h5')

# Save the label encoder
np.save('../model/label_encoder.npy', label_encoder.classes_)

# Save the mean and standard deviation
np.save('../model/mean.npy', [mean_timestamp, mean_speed, mean_x, mean_y, mean_z])
np.save('../model/std.npy', [std_timestamp, std_speed, std_x, std_y, std_z])

# Create a dictionary to hold the metadata
metadata = {
    'mean': [mean_timestamp, mean_speed, mean_x, mean_y, mean_z],
    'std': [std_timestamp, std_speed, std_x, std_y, std_z],
    'labels': ['driving','cycling','train','bus','metro','tram','e-scooter']
}

# Convert the metadata dictionary to JSON string
metadata_json = json.dumps(metadata)

# Save the metadata as a JSON file
with open('../model/metadata.json', 'w') as f:
    f.write(metadata_json)

# Convert the model to Core ML format with a single input
# input_shape =
