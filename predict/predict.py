import sys
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
new_data = np.genfromtxt(filename, delimiter=',', dtype=str)
new_accelerometer_values = new_data[:, 3:].astype(float)

# Load the trained model
loaded_model = tf.keras.models.load_model('../model/trained_model.h5')

# Load the label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('../model/label_encoder.npy')

# Perform preprocessing on the new data
mean = np.load('../model/mean.npy')
std = np.load('../model/std.npy')
normalized_new_data = (new_accelerometer_values - mean) / std

# Make predictions on the new data
predictions = loaded_model.predict(normalized_new_data)

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
