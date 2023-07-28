import os
import tensorflow as tf
import numpy as np
from io import StringIO
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from flask import Flask, request

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the path to the second script
second_script_path = '../predict/predict-2.0.py'

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file is uploaded
    if 'file' not in request.files:
        return 'No file uploaded.'

    file = request.files['file']

    # Check if the file has an allowed extension
    # if not filename.lower().endswith(('.csv')):
    #     return 'Authentication failed. Your connection has been recorded and will be reported'

    # Load the new data for prediction
    new_data = np.genfromtxt(StringIO(file.read().decode('utf-8')), delimiter=',', dtype=float)

    # Extract the relevant columns from the data
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

    probability = sorted_probabilities[0]

    return probability[0]


@app.route('/upload', methods=['POST'])
def upload():
    # Check if a file is uploaded
    if 'file' not in request.files:
        return 'Unauthorised acess. Your ip has been tracked and will be reported', 400

    file = request.files['file']

    # Check if the file has an allowed extension
    if not file.filename.lower().endswith(('.csv')):
        return 'Unauthorised acess. Your ip has been tracked and will be reported', 400

    # Create a unique filename using current datetime
    current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"training_uploaded_file_{current_datetime}.csv"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Create the new file and write the contents of the uploaded file into it
    with open(file_path, 'w') as new_file:
        new_file.write(file.read().decode('utf-8'))

    return 'File uploaded successfully.', 200


if __name__ == "__main__":
    app.run(host='51.68.196.15', port=8000, debug=True)
