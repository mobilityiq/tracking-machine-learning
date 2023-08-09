import os
import tensorflow as tf
import numpy as np
from io import StringIO
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from flask import Flask, request
from training.preprocesing import preprocess_data_for_prediction

app = Flask(__name__)

# Load trained model into memory
model = tf.keras.models.load_model('model/cnn_bilstm_model.h5')

encoder = LabelEncoder()  
encoder.classes_ = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])  # assuming these are the correct labels 

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file is uploaded
    if 'file' not in request.files:
        return 'No file uploaded.'

    file = request.files['file']

    # Load the new data for prediction
    data = np.genfromtxt(StringIO(file.read().decode('utf-8')), delimiter=',', dtype=float)

    # Transform data to multiple channels
    X_channels = preprocess_data_for_prediction(data)

    # Predict with the reshaped data
    predictions = model.predict(X_channels)

    # Decode the predictions to get the labels
    predicted_labels = encoder.inverse_transform(np.argmax(predictions, axis=1))
    
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


if __name__ == "__main__":
    # app.run(host='51.68.196.15', port=8001, debug=True)
    app.run(host='192.168.18.200', port=8001, debug=True)

