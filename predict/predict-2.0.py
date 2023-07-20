import sys
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Check if the filename is provided as a command-line argument
if len(sys.argv) < 2:
    print("Please provide the filename as a command-line argument.")
    sys.exit(1)

# Get the filename from the command-line argument
filename = sys.argv[1]

# Load the new data for prediction
new_data = np.genfromtxt(filename, delimiter=',', dtype=float)

# Extract the relevant columns from the new data
timestamps = new_data[:, 0]
speed = new_data[:, 1]
x = new_data[:, 2]
y = new_data[:, 3]
z = new_data[:, 4]

# Load the trained model
loaded_model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), '..', 'model', 'trained_model-2.0.h5'))

# Load the label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('../model/label_encoder.npy')

# Perform preprocessing on the new data
data = np.column_stack((timestamps, speed, x, y, z))

mean = np.load('../model/mean.npy')
std = np.load('../model/std.npy')

normalized_data = (data - mean) / std

# Expand dimensions to match the LSTM input shape
normalized_data = np.expand_dims(normalized_data, axis=2)
normalized_data = np.reshape(normalized_data, (-1, 5, 1))


# Print the intermediate values for debugging
# print("Normalized New Data:")
# print(normalized_new_data)

# Make predictions on the new data
predictions = loaded_model.predict(normalized_data)

# Get the predicted labels
predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))

# Get the probabilities for each predicted label
probabilities = np.max(predictions, axis=1)

# Create a dictionary to store the mode probabilities
mode_probabilities = {}

# Iterate over the predicted labels and probabilities
for label, probability in zip(predicted_labels, probabilities):
    label = label.upper()
    if label not in mode_probabilities:
        mode_probabilities[label] = []
    mode_probabilities[label].append(probability)

# Calculate the average probability for each mode
average_probabilities = {
    mode: np.mean(probabilities)
    for mode, probabilities in mode_probabilities.items()
}

# Sort the mode probabilities in descending order
sorted_probabilities = sorted(average_probabilities.items(), key=lambda x: x[1], reverse=True)

print("Mode Probabilities:")
for mode, probability in sorted_probabilities:
    print(f"{mode}: {probability}")
