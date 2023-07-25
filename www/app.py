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

    probability = sorted_probabilities[0]

    return probability[0]


@app.route('/upload', methods=['POST'])
def upload():
    # Check if a file is uploaded
    if 'file' not in request.files:
        return 'No file uploaded.', 400

    file = request.files['file']

    # Check if the file has an allowed extension
    if not file.filename.lower().endswith(('.csv')):
        return 'Invalid file type. Only .csv files are allowed.', 400

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
