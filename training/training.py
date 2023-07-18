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
    WALKING = 'Walking'
    DRIVING = 'Driving'
    CYCLING = 'Cycling'
    TRAIN = 'Train'
    BUS = 'Bus'
    SUBWAY = 'Metro'

# Check if the data file is provided as a command-line argument
if len(sys.argv) < 2:
    print("Please provide the path to the data file as a command-line argument.")
    sys.exit(1)

# Get the data file path from the command-line argument
data_file = sys.argv[1]

# Load data from the text file
data = np.genfromtxt(data_file, delimiter=',', dtype=str)

# Extract relevant information from the loaded data
user_ids = data[:, 0].astype(int)
timestamps = data[:, 1]
transportation_modes = data[:, 2]
accelerometer_values = data[:, -3:].astype(float)  # Last three columns

# Perform any necessary preprocessing steps
# For example, you can normalize the accelerometer values

# Normalize accelerometer values
mean = np.mean(accelerometer_values, axis=0)
std = np.std(accelerometer_values, axis=0)
normalized_accelerometer_values = (accelerometer_values - mean) / std

# Encode transportation modes as numerical labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(transportation_modes)
num_classes = len(TransportationMode)

# Split the data into features and labels
features = normalized_accelerometer_values
labels = encoded_labels

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2)

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(features.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Train the model
model.fit(train_features, train_labels, epochs=10, batch_size=32, validation_data=(test_features, test_labels))

# Save the trained model in the native Keras format
model.save('../model/trained_model.keras')

# Save the label encoder
np.save('../model/label_encoder.npy', label_encoder.classes_)

# Save the mean and standard deviation
np.save('../model/mean.npy', mean)
np.save('../model/std.npy', std)

# Create a dictionary to hold the metadata
metadata = {
    'mean': mean.tolist(),
    'std': std.tolist()
}

# Convert the metadata dictionary to JSON string
metadata_json = json.dumps(metadata)

# Convert the model to Core ML format
coreml_model = ct.convert(model)

# Add the metadata to the model as user-defined metadata
coreml_model.user_defined_metadata['preprocessing_metadata'] = metadata_json

# Set the prediction_type to "probability"
coreml_model.user_defined_metadata['prediction_type'] = 'probability'

# Save the Core ML model
coreml_model.save('../model/TransitModePredictor.mlmodel')


