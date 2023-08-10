import os
import tensorflow as tf
import numpy as np
from io import StringIO
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from flask import Flask, request
from training.preprocessing import preprocess_data_for_prediction

app = Flask(__name__)

# Load trained model into memory
model = tf.keras.models.load_model('model/cnn_bilstm_model.h5')

encoder = LabelEncoder()  
encoder.classes_ = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])  # these are the numeric labels 

LABEL_MAP = {
    0: "unknown",
    1: "stationary",
    2: "walking",
    3: "running",
    4: "cycling",
    5: "driving",
    6: "bus",
    7: "train",
    8: "metro"
}

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

    # Decode the predictions to get the numeric labels
    numeric_labels = np.argmax(predictions, axis=1)

    # Convert numeric labels to string labels
    predicted_labels = [LABEL_MAP[label] for label in numeric_labels]
    
    probabilities = np.max(predictions, axis=1)

    mode_probabilities = {}

    for label_str, probability in zip(predicted_labels, probabilities):
        if label_str not in mode_probabilities:
            mode_probabilities[label_str] = []
        mode_probabilities[label_str].append(probability)

    average_probabilities = {mode: np.mean(probabilities) for mode, probabilities in mode_probabilities.items()}

    sorted_probabilities = sorted(average_probabilities.items(), key=lambda x: x[1], reverse=True)

    print("Mode Probabilities:")
    for mode, probability in sorted_probabilities:
        print(f"{mode}: {probability}")

    # Assuming you want to return the label with the highest probability
    return sorted_probabilities[0][0]

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

if __name__ == "__main__":
    app.run(host='192.168.18.200', port=8001, debug=True)
