import os
import numpy as np
from tensorflow import keras
from io import StringIO
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from flask import Flask, request
from flask import jsonify
from training.preprocessing import Preprocessing
from keras.models import load_model

app = Flask(__name__)

UPLOAD_FOLDER = 'www/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

encoder = LabelEncoder()  
encoder.classes_ = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])  # these are the numeric labels 

# Load models
loaded_model = keras.models.load_model(os.path.join(os.path.dirname(__file__), 'model', '3.0', 'trained_model-3.0'))
# loaded_cnn_bilstm_model = keras.models.load_model(os.path.join(os.path.dirname(__file__), 'model', 'cnn-bi-lstm', 'cnn_bilstm_model.h5'))
loaded_lstm_model = keras.models.load_model(os.path.join(os.path.dirname(__file__), 'model', 'lstm', 'trained_lstm_model.h5'))
loaded_bi_lstm_model = keras.models.load_model(os.path.join(os.path.dirname(__file__), 'model', 'bi-lstm', 'trained_bi-lstm_model.h5'))
loaded_conv1d_lstm_model = keras.models.load_model(os.path.join(os.path.dirname(__file__), 'model', 'conv1d-lstm', 'trained_conv1d-lstm_model.h5'))

@app.route('/predict-bi-lstm', methods=['POST'])
def predict_lbi_stm():
    # Check if a file is uploaded
    if 'file' not in request.files:
        return 'Unauthorised acess. Your ip has been tracked and will be reported', 400

    file = request.files['file']

    # Check if the file has an allowed extension
    # if not filename.lower().endswith(('.csv')):
    #     return 'Authentication failed. Your connection has been recorded and will be reported'

    # Load the new data for prediction
    new_data = np.genfromtxt(StringIO(file.read().decode('utf-8')), delimiter=',', dtype=float)

    # Extract the relevant columns from the data
    timestamps = new_data[:, 0]
    x = new_data[:, 1]
    y = new_data[:, 2]
    z = new_data[:, 3]
    mx = new_data[:, 4]
    my = new_data[:, 5]
    mz = new_data[:, 6]

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(os.path.join(os.path.dirname(__file__), 'model', 'bi-lstm', 'label_encoder.npy'))

    mean = np.load(os.path.join(os.path.dirname(__file__), 'model', 'bi-lstm', 'mean.npy'))
    std = np.load(os.path.join(os.path.dirname(__file__), 'model', 'bi-lstm', 'std.npy'))

    normalized_timestamp = (timestamps - mean[0]) / std[0]
    normalized_x = (x - mean[1]) / std[1]
    normalized_y = (y - mean[2]) / std[2]
    normalized_z = (z - mean[3]) / std[3]
    normalized_mx = (mx - mean[4]) / std[4]
    normalized_my = (my - mean[5]) / std[5]
    normalized_mz = (mz - mean[6]) / std[6]

    # Include normalized features in data
    data = np.column_stack((normalized_timestamp, normalized_x, normalized_y, normalized_z, normalized_mx, normalized_my, normalized_mz))

    print("Inference Data (First 5 Rows):")
    print(data[:5])  # Print the first 5 rows of the normalized features

    # Reshape the data
    data = data.reshape(data.shape[0], 1, data.shape[1])

    predictions = loaded_bi_lstm_model.predict(data)

    predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
    print(predicted_labels)
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
    # return jsonify({"mode": probability[0], "probability": float(probability[1])})

    return probability[0]

@app.route('/predict-lstm', methods=['POST'])
def predict_lstm():
    # Check if a file is uploaded
    if 'file' not in request.files:
        return 'Unauthorised acess. Your ip has been tracked and will be reported', 400

    file = request.files['file']

    # Check if the file has an allowed extension
    # if not filename.lower().endswith(('.csv')):
    #     return 'Authentication failed. Your connection has been recorded and will be reported'

    # Load the new data for prediction
    new_data = np.genfromtxt(StringIO(file.read().decode('utf-8')), delimiter=',', dtype=float)

    # Extract the relevant columns from the data
    timestamps = new_data[:, 0]
    x = new_data[:, 1]
    y = new_data[:, 2]
    z = new_data[:, 3]
    mx = new_data[:, 4]
    my = new_data[:, 5]
    mz = new_data[:, 6]

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(os.path.join(os.path.dirname(__file__), 'model', 'lstm', 'label_encoder.npy'))

    mean = np.load(os.path.join(os.path.dirname(__file__), 'model', 'lstm', 'mean.npy'))
    std = np.load(os.path.join(os.path.dirname(__file__), 'model', 'lstm', 'std.npy'))

    normalized_timestamp = (timestamps - mean[0]) / std[0]
    normalized_x = (x - mean[1]) / std[1]
    normalized_y = (y - mean[2]) / std[2]
    normalized_z = (z - mean[3]) / std[3]
    normalized_mx = (mx - mean[4]) / std[4]
    normalized_my = (my - mean[5]) / std[5]
    normalized_mz = (mz - mean[6]) / std[6]

    # Include normalized features in data
    data = np.column_stack((normalized_timestamp, normalized_x, normalized_y, normalized_z, normalized_mx, normalized_my, normalized_mz))

    # Reshape the data
    data = data.reshape(data.shape[0], 1, data.shape[1])

    predictions = loaded_lstm_model.predict(data)

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
    # return jsonify({"mode": probability[0], "probability": float(probability[1])})

    return probability[0]

@app.route('/predict-conv1d-lstm', methods=['POST'])
def predict_conv1d_lstm():
    # Check if a file is uploaded
    if 'file' not in request.files:
        return 'Unauthorised acess. Your ip has been tracked and will be reported', 400

    file = request.files['file']

    # Check if the file has an allowed extension
    # if not filename.lower().endswith(('.csv')):
    #     return 'Authentication failed. Your connection has been recorded and will be reported'

    # Load the new data for prediction
    new_data = np.genfromtxt(StringIO(file.read().decode('utf-8')), delimiter=',', dtype=float)

    # Extract the relevant columns from the data
    timestamps = new_data[:, 0]
    x = new_data[:, 1]
    y = new_data[:, 2]
    z = new_data[:, 3]
    mx = new_data[:, 4]
    my = new_data[:, 5]
    mz = new_data[:, 6]

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(os.path.join(os.path.dirname(__file__), 'model', 'conv1d-lstm', 'label_encoder.npy'))

    mean = np.load(os.path.join(os.path.dirname(__file__), 'model', 'conv1d-lstm', 'mean.npy'))
    std = np.load(os.path.join(os.path.dirname(__file__), 'model', 'conv1d-lstm', 'std.npy'))

    normalized_timestamp = (timestamps - mean[0]) / std[0]
    normalized_x = (x - mean[1]) / std[1]
    normalized_y = (y - mean[2]) / std[2]
    normalized_z = (z - mean[3]) / std[3]
    normalized_mx = (mx - mean[4]) / std[4]
    normalized_my = (my - mean[5]) / std[5]
    normalized_mz = (mz - mean[6]) / std[6]

    # Include normalized features in data
    data = np.column_stack((normalized_timestamp, normalized_x, normalized_y, normalized_z, normalized_mx, normalized_my, normalized_mz))

    # Reshape the data
    data = data.reshape(data.shape[0], 1, data.shape[1])

    predictions = loaded_conv1d_lstm_model.predict(data)

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
    # return jsonify({"mode": probability[0], "probability": float(probability[1])})

    return probability[0]

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file is uploaded
    if 'file' not in request.files:
        return 'Unauthorised acess. Your ip has been tracked and will be reported', 400

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

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load('model/3.0/label_encoder.npy')

    mean = np.load('model/3.0/mean.npy')
    std = np.load('model/3.0/std.npy')

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

# @app.route('/predict-cnn-bilstm', methods=['POST'])
# def predict_cnn_bilstm():
# 
    # LABEL_MAP = {
    #     1: "stationary",
    #     2: "walking",
    #     3: "running",
    #     4: "cycling",
    #     5: "driving",
    #     6: "bus",
    #     7: "train",
    #     8: "metro"
    # }
    # 
#     # Check if a file is uploaded
#     if 'file' not in request.files:
#         return 'No file uploaded.'

#     file = request.files['file']

#     # Load the new data for prediction
#     data = np.genfromtxt(StringIO(file.read().decode('utf-8')), delimiter=',', dtype=float)

#     # Transform data to multiple channels
#     X_channels = Preprocessing.preprocess_data_for_prediction(data)


#     # Predict with the reshaped data
#     predictions = loaded_cnn_bilstm_model.predict(X_channels)

#     # Decode the predictions to get the numeric labels
#     numeric_labels = np.argmax(predictions, axis=1)

#     # Convert numeric labels to string labels
#     predicted_labels = [LABEL_MAP[label] for label in numeric_labels]
    
#     probabilities = np.max(predictions, axis=1)

#     mode_probabilities = {}

#     for label_str, probability in zip(predicted_labels, probabilities):
#         if label_str not in mode_probabilities:
#             mode_probabilities[label_str] = []
#         mode_probabilities[label_str].append(probability)

#     average_probabilities = {mode: np.mean(probabilities) for mode, probabilities in mode_probabilities.items()}

#     sorted_probabilities = sorted(average_probabilities.items(), key=lambda x: x[1], reverse=True)

#     print("Mode Probabilities:")
#     for mode, probability in sorted_probabilities:
#         print(f"{mode}: {probability}")

#     # Assuming you want to return the label with the highest probability
#     return sorted_probabilities[0][0]

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

    return "Ok", 200

if __name__ == "__main__":
    # app.run(host='51.68.196.15', port=8000, debug=True)
    app.run(host='192.168.18.200', port=8000, debug=True)
