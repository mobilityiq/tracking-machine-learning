import sys
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

if len(sys.argv) < 2:
    print("Please provide the filename as a command-line argument.")
    sys.exit(1)

filename = sys.argv[1]

new_data = np.genfromtxt(filename, delimiter=',', dtype=float)

# Extract only the first 9 columns of the new data
timestamps = new_data[:, 0]
speed = new_data[:, 1]
course = new_data[:, 2]
x = new_data[:, 3]
y = new_data[:, 4]
z = new_data[:, 5]
qx = new_data[:, 6]
qy = new_data[:, 7]
qz = new_data[:, 8]
qw = new_data[:, 9]

loaded_model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), '..', 'model', 'trained_model-3.0.h5'))

label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('../model/label_encoder.npy')

mean = np.load('../model/mean.npy')
std = np.load('../model/std.npy')

normalized_timestamp = (timestamps - mean[0]) / std[0]
normalized_speed = (speed - mean[1]) / std[1]
normalized_course = (course - mean[2]) / std[2]
normalized_x = (x - mean[3]) / std[3]
normalized_y = (y - mean[4]) / std[4]
normalized_z = (z - mean[5]) / std[5]
normalized_qx = (qx - mean[6]) / std[6]
normalized_qy = (qy - mean[7]) / std[7]
normalized_qz = (qz - mean[8]) / std[8]
normalized_qw = (qw - mean[9]) / std[9]

# Include normalized features in data
data = np.column_stack((normalized_timestamp, normalized_speed, normalized_course, normalized_x, normalized_y, normalized_z, normalized_qx, normalized_qy, normalized_qz, normalized_qw))

predictions = loaded_model.predict(data)

predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))

probabilities = np.max(predictions, axis=1)

mode_probabilities = {}

for label, probability in zip(predicted_labels, probabilities):
    label = label.upper()
    if label not in mode_probabilities:
        mode_probabilities[label] = []
    mode_probabilities[label].append(probability)

average_probabilities = {mode: np.mean(probabilities) for mode, probabilities in mode_probabilities.items()}

sorted_probabilities = sorted(average_probabilities.items(), key=lambda x: x[1], reverse=True)

print("Mode Probabilities:")
for mode, probability in sorted_probabilities:
    print(f"{mode}: {probability}")
