import os
import sys
import pandas as pd
from io import StringIO
import numpy as np
from tensorflow import keras
from io import StringIO
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from flask import Flask, request, jsonify
from training.preprocessing import Preprocessing
from keras.models import load_model
import joblib
from enum import Enum
from sklearn.metrics import accuracy_score
import pickle

app = Flask(__name__)

UPLOAD_FOLDER = 'www/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained Random Forest model and imputer
rf_classifier = joblib.load('model/3.0/rf_trained_model-3.0.joblib')
imputer = joblib.load('model/3.0/imputer.joblib')
# Load the labels
loaded_classes = np.load('model/3.0/label_encoder.npy')
label_encoder = LabelEncoder()
label_encoder.classes_ = loaded_classes
# Load the statistics
with open('model/3.0/statistics.pkl', 'rb') as f:
    statistics = pickle.load(f)

class FeaturesNames(Enum):
    SPEED = 'speed'
    BEARING = 'bearing'
    ACC_X = 'acc_x'
    ACC_Y = 'acc_y'
    ACC_Z = 'acc_z'
    JERK_X = 'jerk_x'
    JERK_Y = 'jerk_y'
    JERK_Z = 'jerk_z'
    ACC_MAGNITUDE = 'acc_mag'
    MAG_X = 'mag_x'
    MAG_Y = 'mag_y'
    MAG_Z = 'mag_z'
    JERK_MAG_X = 'jerk_mx'
    JERK_MAG_Y = 'jerk_my'
    JERK_MAG_Z = 'jerk_mz'
    MAG_MAGNITUDE = 'mag_mag'

def compute_magnitude(arrays):
    squares = [np.square(array) for array in arrays]
    sum_of_squares = np.sum(squares, axis=0)
    return np.sqrt(sum_of_squares)

def compute_jerk(data):
    return [data[i+1] - data[i] for i in range(len(data)-1)] + [0]

def load_and_extract_features(data_string):
    df = pd.read_csv(StringIO(data_string))

    # Assuming the CSV doesn't include headers
    data = df.to_numpy()

    speed = data[:, 1].astype(float)
    course = data[:, 2].astype(float)
    x = data[:, 3].astype(float)
    y = data[:, 4].astype(float)
    z = data[:, 5].astype(float)
    mx = data[:, 6].astype(float)
    my = data[:, 7].astype(float)
    mz = data[:, 8].astype(float)
    
    return speed, course, x, y, z, mx, my, mz


def compute_statistics(data):
    return np.mean(data), np.std(data)


def normalize_data(data, mean, std):
    return (data - mean) / std

def check_data_types(*arrays):
    for arr in arrays:
        if not np.issubdtype(arr.dtype, np.number):
            print(f"Found non-numeric data: {arr[arr != arr.astype(float).astype(str)]}")


def preprocess_data(speed, course, x, y, z, jerk_ax, jerk_ay, jerk_az, acc_magnitude, mx, my, mz, jerk_mx, jerk_my, jerk_mz, mag_magnitude, 
                    mean_acc_magnitude=None, std_acc_magnitude=None, mean_mag_magnitude=None, std_mag_magnitude=None, 
                    mean_speed=None, std_speed=None, mean_course=None, std_course=None, 
                    mean_x=None, std_x=None, mean_y=None, std_y=None, mean_z=None, std_z=None, 
                    mean_mx=None, std_mx=None, mean_my=None, std_my=None, mean_mz=None, std_mz=None,
                    mean_jerk_ax=None, std_jerk_ax=None, mean_jerk_ay=None, std_jerk_ay=None, mean_jerk_az=None, std_jerk_az=None,
                    mean_jerk_mx=None, std_jerk_mx=None, mean_jerk_my=None, std_jerk_my=None, mean_jerk_mz=None, std_jerk_mz=None):    
    
    # This part was repeated, so removing the redundant calculations
    normalized_speed = normalize_data(speed, statistics["mean_speed"], statistics["std_speed"])
    normalized_course = normalize_data(course, statistics["mean_course"], statistics["std_course"])
    normalized_x = normalize_data(x, statistics["mean_x"], statistics["std_x"])
    normalized_y = normalize_data(y, statistics["mean_y"], statistics["std_y"])
    normalized_z = normalize_data(z, statistics["mean_z"], statistics["std_z"])
    normalized_mx = normalize_data(mx, statistics["mean_mx"], statistics["std_mx"])
    normalized_my = normalize_data(my, statistics["mean_my"], statistics["std_my"])
    normalized_mz = normalize_data(mz, statistics["mean_mz"], statistics["std_mz"])
    normalized_acc_magnitude = normalize_data(acc_magnitude, statistics["mean_acc_magnitude"], statistics["std_acc_magnitude"])
    normalized_mag_magnitude = normalize_data(mag_magnitude, statistics["mean_mag_magnitude"], statistics["std_mag_magnitude"])
    normalized_jerk_ax = normalize_data(jerk_ax, statistics["mean_jerk_ax"], statistics["std_jerk_ax"])
    normalized_jerk_ay = normalize_data(jerk_ay, statistics["mean_jerk_ay"], statistics["std_jerk_ay"])
    normalized_jerk_az = normalize_data(jerk_az, statistics["mean_jerk_az"], statistics["std_jerk_az"])
    normalized_jerk_mx = normalize_data(jerk_mx, statistics["mean_jerk_mx"], statistics["std_jerk_mx"])
    normalized_jerk_my = normalize_data(jerk_my, statistics["mean_jerk_my"], statistics["std_jerk_my"])
    normalized_jerk_mz = normalize_data(jerk_mz, statistics["mean_jerk_mz"], statistics["std_jerk_mz"])


    features = np.column_stack((normalized_speed, normalized_course, normalized_x, normalized_y, normalized_z, normalized_jerk_ax, normalized_jerk_ay, normalized_jerk_az, normalized_acc_magnitude, 
                                normalized_mx, normalized_my, normalized_mz, normalized_jerk_mx, normalized_jerk_my, normalized_jerk_mz, normalized_mag_magnitude))    
        

    return features


def predict_with_rf(features, true_labels):
    predicted_labels = rf_classifier.predict(features)
    accuracy = accuracy_score(true_labels, predicted_labels)
    return accuracy



@app.route('/predict', methods=['POST'])
def predict():
    try:

        # Receive the uploaded file
        uploaded_file = request.files['file']

        if uploaded_file.filename != '':
            # Directly read the uploaded file without saving it to disk
            data_file = uploaded_file.read().decode('utf-8')
            speed, course, x, y, z, mx, my, mz = load_and_extract_features(data_file)


        acc_magnitudes = compute_magnitude([x, y, z])
        mag_magnitudes = compute_magnitude([mx, my, mz])

        smoothed_acc_x = Preprocessing.apply_savitzky_golay(x)
        smoothed_acc_y = Preprocessing.apply_savitzky_golay(y)
        smoothed_acc_z = Preprocessing.apply_savitzky_golay(z)
        smoothed_mag_x = Preprocessing.apply_savitzky_golay(mx)
        smoothed_mag_y = Preprocessing.apply_savitzky_golay(my)
        smoothed_mag_z = Preprocessing.apply_savitzky_golay(mz)

        # Assuming you have your accelerometer data as:
        jerk_ax = compute_jerk(smoothed_acc_x)
        jerk_ay = compute_jerk(smoothed_acc_y)
        jerk_az = compute_jerk(smoothed_acc_z)

        # Assuming you have your magnetometer data as:
        jerk_mx = compute_jerk(smoothed_mag_x)
        jerk_my = compute_jerk(smoothed_mag_y)
        jerk_mz = compute_jerk(smoothed_mag_z)

        features = preprocess_data(speed, course, x, y, z, jerk_ax, jerk_ay, jerk_az, acc_magnitudes, mx, my, mz, jerk_mx, jerk_my, jerk_mz, mag_magnitudes)
        
        # Fit and transform the features
        features_imputed = imputer.fit_transform(features)

        # 1. Get prediction probabilities
        predicted_probabilities = rf_classifier.predict_proba(features_imputed)

        # 2. Average the probabilities
        avg_probabilities = predicted_probabilities.mean(axis=0)

        # 3. Map the numeric labels to modes
        modes = label_encoder.classes_

        # Create a dictionary of numeric labels and their probabilities
        numeric_label_probs = {str(mode): round(probability * 100, 2) for mode, probability in zip(modes, avg_probabilities)}

        print(f"numeric_label_probs: {numeric_label_probs}")

        # Replace numeric labels with actual labels
        # label_probs = {label_encoder.inverse_transform([float(key)])[0]: value for key, value in numeric_label_probs.items()}

        return jsonify(numeric_label_probs)
    
    except Exception as e:
        print(str(e))
        return jsonify({'error': str(e)})



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
    app.run(host='51.68.196.15', port=8000, debug=True)
    # app.run(host='192.168.18.200', port=8000, debug=True)
