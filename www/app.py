import os
import numpy as np
import tensorflow as tf
from flask import Flask, request
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('../model/trained_model-2.0.h5')

# Load the label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('../model/label_encoder.npy')

UPLOAD_FOLDER = 'uploads'  # Define the folder to store uploaded files
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file is uploaded
    if 'file' not in request.files:
        return 'No file uploaded.'

    file = request.files['file']

    # Check if the file has an allowed extension
    if not file.filename.lower().endswith(('.txt')):
        return 'Invalid file extension. Only .txt files are allowed.'

    # Save the uploaded file to the uploads folder
    # file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    # file.save(file_path)

    # Read the file content
    file_content = file.read().decode('utf-8')

    # Split the content into lines
    lines = file_content.split('\n')

    # Initialize an empty array to store the accelerometer values
    data = []
    expected_num_values = 5
    # Iterate over the lines and extract the accelerometer values
    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespace
        if line:  # Skip empty lines
            values = line.split(',')
            if len(values) != expected_num_values:  # Replace `expected_num_values` with the actual number of values expected
                print(f"Inconsistent values in line: {line}")
                return 'Invalid file format. Inconsistent number of values in the lines.'
            if values:  # Check if `values` is not empty
                try:
                    values = [float(value) for value in values]
                    data.append(values)
                except ValueError:
                    return 'Invalid file format. Only numeric values are allowed.'
            else:
                return 'Invalid file format. Empty values found in the lines.'

    # Convert the data to a NumPy array
    data = np.array(data)

    # Preprocess the accelerometer values
    mean = np.load('../model/mean.npy')
    std = np.load('../model/std.npy')

    normalized_data = (data - mean) / std

    # Expand dimensions to match the LSTM input shape
    normalized_data = np.expand_dims(normalized_data, axis=2)
    normalized_data = np.reshape(normalized_data, (-1, 5, 1))


    # Make predictions on the preprocessed data
    predictions = model.predict(normalized_data)

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

    # Print the average probabilities
    for mode, probability in sorted_probabilities:
        print(f"Mode: {mode}, Average Probability: {probability}")

    # Return the predicted mode as a string
    predicted_mode = predicted_labels[0]  # Assuming only one prediction is made
    return predicted_mode


if __name__ == "__main__":
    app.run(host='192.168.18.200', port=8000, debug=True)
