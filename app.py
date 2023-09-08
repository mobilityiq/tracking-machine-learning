import os
import sys
import numpy as np
from tensorflow import keras
from io import StringIO
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from flask import Flask, request, jsonify
from training.preprocessing import Preprocessing
from keras.models import load_model
from sklearn.externals import joblib
from enum import Enum
from sklearn.metrics import accuracy_score


app = Flask(__name__)

UPLOAD_FOLDER = 'www/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained Random Forest model and imputer
rf_classifier = joblib.load('../model/3.0/rf_trained_model-3.0.joblib')
imputer = joblib.load('../model/3.0/imputer.joblib')


# encoder = LabelEncoder()  
# encoder.classes_ = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])  # these are the numeric labels 

# Define the transportation mode Enum
class TransportationMode(Enum):
    DRIVING = 'driving'
    CYCLING = 'cycling'
    TRAIN = 'train'
    BUS = 'bus'
    SUBWAY = 'metro'
    TRAM = 'tram'
    # ESCOOTER = 'e-scooter'

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

def load_and_extract_features(file_path):
    data = np.genfromtxt(file_path, delimiter=',', dtype=str)
    timestamp = data[:, 0].astype(float)  # This is the new timestamp column
    speed = data[:, 1].astype(float)
    course = data[:, 2].astype(float)
    x = data[:, 3].astype(float)
    y = data[:, 4].astype(float)
    z = data[:, 5].astype(float)
    mx = data[:, 6].astype(float)
    my = data[:, 7].astype(float)
    mz = data[:, 8].astype(float)
    modes = data[:, -1]
    
    return timestamp, speed, course, x, y, z, mx, my, mz, modes

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
    
    if mean_speed is None or std_speed is None:
        mean_speed, std_speed = compute_statistics(speed)
    if mean_course is None or std_course is None:
        mean_course, std_course = compute_statistics(course)
    if mean_x is None or std_x is None:
        mean_x, std_x = compute_statistics(x)
    if mean_y is None or std_y is None:
        mean_y, std_y = compute_statistics(y)
    if mean_z is None or std_z is None:
        mean_z, std_z = compute_statistics(z)
    if mean_mx is None or std_mx is None:
        mean_mx, std_mx = compute_statistics(mx)
    if mean_my is None or std_my is None:
        mean_my, std_my = compute_statistics(my)
    if mean_mz is None or std_mz is None:
        mean_mz, std_mz = compute_statistics(mz)
    if mean_acc_magnitude is None or std_acc_magnitude is None:
        mean_acc_magnitude, std_acc_magnitude = compute_statistics(acc_magnitude)
    if mean_mag_magnitude is None or std_mag_magnitude is None:
        mean_mag_magnitude, std_mag_magnitude = compute_statistics(mag_magnitude)
         # Add the new jerk statistics
    if mean_jerk_ax is None or std_jerk_ax is None:
        mean_jerk_ax, std_jerk_ax = compute_statistics(jerk_ax)
    if mean_jerk_ay is None or std_jerk_ay is None:
        mean_jerk_ay, std_jerk_ay = compute_statistics(jerk_ay)
    if mean_jerk_az is None or std_jerk_az is None:
        mean_jerk_az, std_jerk_az = compute_statistics(jerk_az)
    if mean_jerk_mx is None or std_jerk_mx is None:
        mean_jerk_mx, std_jerk_mx = compute_statistics(jerk_mx)
    if mean_jerk_my is None or std_jerk_my is None:
        mean_jerk_my, std_jerk_my = compute_statistics(jerk_my)
    if mean_jerk_mz is None or std_jerk_mz is None:
        mean_jerk_mz, std_jerk_mz = compute_statistics(jerk_mz)
        
    
    # This part was repeated, so removing the redundant calculations
    normalized_speed = normalize_data(speed, mean_speed, std_speed)
    normalized_course = normalize_data(course, mean_course, std_course)
    normalized_x = normalize_data(x, mean_x, std_x)
    normalized_y = normalize_data(y, mean_y, std_y)
    normalized_z = normalize_data(z, mean_z, std_z)
    normalized_mx = normalize_data(mx, mean_mx, std_mx)
    normalized_my = normalize_data(my, mean_my, std_my)
    normalized_mz = normalize_data(mz, mean_mz, std_mz)
    normalized_acc_magnitude = normalize_data(acc_magnitude, mean_acc_magnitude, std_acc_magnitude)
    normalized_mag_magnitude = normalize_data(mag_magnitude, mean_mag_magnitude, std_mag_magnitude)
    normalized_jerk_ax = normalize_data(jerk_ax, mean_jerk_ax, std_jerk_ax)
    normalized_jerk_ay = normalize_data(jerk_ay, mean_jerk_ay, std_jerk_ay)
    normalized_jerk_az = normalize_data(jerk_az, mean_jerk_az, std_jerk_az)
    normalized_jerk_mx = normalize_data(jerk_mx, mean_jerk_mx, std_jerk_mx)
    normalized_jerk_my = normalize_data(jerk_my, mean_jerk_my, std_jerk_my)
    normalized_jerk_mz = normalize_data(jerk_mz, mean_jerk_mz, std_jerk_mz)

    features = np.column_stack((normalized_speed, normalized_course, normalized_x, normalized_y, normalized_z, normalized_jerk_ax, normalized_jerk_ay, normalized_jerk_az, normalized_acc_magnitude, 
                                normalized_mx, normalized_my, normalized_mz, normalized_jerk_mx, normalized_jerk_my, normalized_jerk_mz, normalized_mag_magnitude))    
        
    statistics = {
        'mean_speed': mean_speed, 'std_speed': std_speed,
        'mean_course': mean_course, 'std_course': std_course,
        'mean_x': mean_x, 'std_x': std_x,
        'mean_y': mean_y, 'std_y': std_y,
        'mean_z': mean_z, 'std_z': std_z,
        'mean_mx': mean_mx, 'std_mx': std_mx,
        'mean_my': mean_my, 'std_my': std_my,
        'mean_mz': mean_mz, 'std_mz': std_mz,
        'mean_acc_magnitude': mean_acc_magnitude, 
        'std_acc_magnitude': std_acc_magnitude,
        'mean_mag_magnitude': mean_mag_magnitude, 
        'std_mag_magnitude': std_mag_magnitude,
        'mean_jerk_ax': mean_jerk_ax, 'std_jerk_ax': std_jerk_ax,
        'mean_jerk_ay': mean_jerk_ay, 'std_jerk_ay': std_jerk_ay,
        'mean_jerk_az': mean_jerk_az, 'std_jerk_az': std_jerk_az,
        'mean_jerk_mx': mean_jerk_mx, 'std_jerk_mx': std_jerk_mx,
        'mean_jerk_my': mean_jerk_my, 'std_jerk_my': std_jerk_my,
        'mean_jerk_mz': mean_jerk_mz, 'std_jerk_mz': std_jerk_mz
    }
    
    return features, statistics


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
            # Save the uploaded file temporarily
            file_path = 'temp_data.csv'
            uploaded_file.save(file_path)

        # Load and preprocess the data from the uploaded file
        timestamp, speed, course, x, y, z, mx, my, mz, modes = load_and_extract_features(file_path)

        acc_magnitude = compute_magnitude([x, y, z])
        mag_magnitude = compute_magnitude([mx, my, mz])

        # Apply the filters
        smoothed_acc_x = Preprocessing.apply_savitzky_golay(x)
        smoothed_acc_y = Preprocessing.apply_savitzky_golay(y)
        smoothed_acc_z = Preprocessing.apply_savitzky_golay(z)
        smoothed_mag_x = Preprocessing.apply_savitzky_golay(mx)
        smoothed_mag_y = Preprocessing.apply_savitzky_golay(my)
        smoothed_mag_z = Preprocessing.apply_savitzky_golay(mz)

        # Calculating jerks
        jerk_ax = compute_jerk(smoothed_acc_x)
        jerk_ay = compute_jerk(smoothed_acc_y)
        jerk_az = compute_jerk(smoothed_acc_z)


        jerk_mx = compute_jerk(smoothed_mag_x)
        jerk_my = compute_jerk(smoothed_mag_y)
        jerk_mz = compute_jerk(smoothed_mag_z)

        # Encode transportation modes as numerical labels
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(modes)
        num_classes = len(TransportationMode)

        # Preprocess the training data
        train_features, train_statistics = preprocess_data(speed, course, x, y, z, jerk_ax, jerk_ay, jerk_az, acc_magnitude, mx, my, mz, jerk_mx, jerk_my, jerk_mz, mag_magnitude)

        train_labels = label_encoder.transform(modes)

        # Load testing data
        data_test_file = 'testing-3.0.csv'
        test_timestamp, test_speed, test_course, test_x, test_y, test_z, test_mx, test_my, test_mz, test_modes = load_and_extract_features(data_test_file)

        acc_test_magnitudes = compute_magnitude([test_x, test_y, test_z])
        mag_test_magnitudes = compute_magnitude([test_mx, test_my, test_mz])

        smoothed_acc_test_x = Preprocessing.apply_savitzky_golay(test_x)
        smoothed_acc_test_y = Preprocessing.apply_savitzky_golay(test_y)
        smoothed_acc_test_z = Preprocessing.apply_savitzky_golay(test_z)
        smoothed_mag_test_x = Preprocessing.apply_savitzky_golay(test_mx)
        smoothed_mag_test_y = Preprocessing.apply_savitzky_golay(test_my)
        smoothed_mag_test_z = Preprocessing.apply_savitzky_golay(test_mz)

        # Assuming you have your accelerometer data as:
        jerk_test_ax = compute_jerk(smoothed_acc_test_x)
        jerk_test_ay = compute_jerk(smoothed_acc_test_y)
        jerk_test_az = compute_jerk(smoothed_acc_test_z)

        # Assuming you have your magnetometer data as:
        jerk_test_mx = compute_jerk(smoothed_mag_test_x)
        jerk_test_my = compute_jerk(smoothed_mag_test_y)
        jerk_test_mz = compute_jerk(smoothed_mag_test_z)

        test_features = preprocess_data(
        test_speed, test_course, test_x, test_y, test_z, jerk_test_ax, jerk_test_ay, jerk_test_az, acc_test_magnitudes, test_mx, test_my, test_mz, jerk_test_mx, jerk_test_my, jerk_test_mz, mag_test_magnitudes,  **train_statistics
    )[0]

        # Make predictions using the trained Random Forest model
        predicted_labels = rf_classifier.predict(test_features)

        # Convert numeric labels back to transportation modes
        predicted_modes = label_encoder.inverse_transform(predicted_labels)

        # Return the predicted modes as JSON
        result = {'predicted_modes': predicted_modes.tolist()}
        
        return jsonify(result)
    
    except Exception as e:
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
    # app.run(host='51.68.196.15', port=8000, debug=True)
    app.run(host='192.168.18.200', port=8000, debug=True)
