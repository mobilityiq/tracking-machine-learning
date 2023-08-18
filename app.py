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

# encoder = LabelEncoder()  
# encoder.classes_ = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])  # these are the numeric labels 

f1_metric = Preprocessing.f1_metric
custom_objects = {'f1_metric': f1_metric}

# Load models
# 3.0
loaded_model = keras.models.load_model('model/3.0/trained_model-3.0/')
# LSTM
loaded_lstm_model = keras.models.load_model('model/lstm/trained_lstm_model/', custom_objects=custom_objects)
# BiLSTM
loaded_bi_lstm_model = keras.models.load_model('model/bi-lstm/trained_bi-lstm_model/', custom_objects=custom_objects)
# CONV1D-LSTM
loaded_conv1d_lstm_model = keras.models.load_model('model/conv1d-lstm/trained_conv1d-lstm_model/', custom_objects=custom_objects)
# CNN-BiLSTM
loaded_cnn_bi_lstm_model = keras.models.load_model('model/cnn-bi-lstm/cnn_bilstm_model/', custom_objects=custom_objects)

@app.route('/predict-bi-lstm', methods=['POST'])
def predict_bi_lstm():
    data = request.get_json()
    
    # Check if data is available and is a list
    if not data or not isinstance(data, list):
        return 'Invalid data provided', 400

    data_list = [[entry['timestamp'], entry['x'], entry['y'], entry['z'], entry['mx'], entry['my'], entry['mz']] for entry in data]
    all_data = np.array(data_list, dtype=float)

    print("Data shape:",all_data.shape)

    # Extract relevant columns
    timestamps = all_data[:, 0]
    x = all_data[:, 1]
    y = all_data[:, 2]
    z = all_data[:, 3]
    mx = all_data[:, 4]
    my = all_data[:, 5]
    mz = all_data[:, 6]

    # Apply the filter to data
    x = Preprocessing.apply_savitzky_golay(x)
    y = Preprocessing.apply_savitzky_golay(y)
    z = Preprocessing.apply_savitzky_golay(z)
    mx = Preprocessing.apply_savitzky_golay(mx)
    my = Preprocessing.apply_savitzky_golay(my)
    mz = Preprocessing.apply_savitzky_golay(mz)

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

    # print("Inference Data (First 5 Rows):")
    # print(data[:5])  # Print the first 5 rows of the normalized features

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
    data = request.get_json()
    
    # Check if data is available and is a list
    if not data or not isinstance(data, list):
        return 'Invalid data provided', 400

    data_list = [[entry['timestamp'], entry['x'], entry['y'], entry['z'], entry['mx'], entry['my'], entry['mz']] for entry in data]
    all_data = np.array(data_list, dtype=float)

    print("Data shape:",all_data.shape)

    # Extract relevant columns
    timestamps = all_data[:, 0]
    x = all_data[:, 1]
    y = all_data[:, 2]
    z = all_data[:, 3]
    mx = all_data[:, 4]
    my = all_data[:, 5]
    mz = all_data[:, 6]


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

    # print(predictions[:10])  # Print the first 10 predictions to get an idea of the scores


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
    data = request.get_json()
    
    # Check if data is available and is a list
    if not data or not isinstance(data, list):
        return 'Invalid data provided', 400

    data_list = [[entry['timestamp'], entry['x'], entry['y'], entry['z'], entry['mx'], entry['my'], entry['mz']] for entry in data]
    all_data = np.array(data_list, dtype=float)

    print("Data shape:",all_data.shape)

    # Extract relevant columns
    timestamps = all_data[:, 0]
    x = all_data[:, 1]
    y = all_data[:, 2]
    z = all_data[:, 3]
    mx = all_data[:, 4]
    my = all_data[:, 5]
    mz = all_data[:, 6]

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


@app.route('/predict-cnn-bilstm', methods=['POST'])
def predict_cnn_bilstm():
    
    # Extract JSON data from the request
    data = request.get_json()
    
    # Check if data is available and is a list
    if not data or not isinstance(data, list):
        return 'Invalid data provided', 400

    # Convert data to the expected format
    data_list = [[entry['x'], entry['y'], entry['z'], entry['mx'], entry['my'], entry['mz']] for entry in data]
    all_data = np.array(data_list, dtype=float)
    print(all_data.shape)  # Expected: (1200, 6)

    # If you're using a label encoder
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load('model/cnn-bi-lstm/label_encoder.npy')

    # Since you're moving from files to JSON, we no longer use genfromtxt. 
    # Instead, you directly have the all_data array from the JSON data.

    # Transform data to multiple channels
    # Assuming Preprocessing is a module/class you've defined elsewhere
    X_channels = Preprocessing.preprocess_data_for_cnn_bilstm_prediction(all_data)

    # Predict with the reshaped data
    predictions = loaded_cnn_bi_lstm_model.predict(X_channels)

    # Convert numeric labels to string labels
    predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
    
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
    # return jsonify(mode=sorted_probabilities[0][0], probability=sorted_probabilities[0][1])
    return sorted_probabilities[0]



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
