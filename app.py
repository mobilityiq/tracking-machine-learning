import os
import pickle
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from flask import Flask, request, jsonify
from training.preprocessing import Preprocessing
from keras.models import load_model
import joblib
from enum import Enum
from sklearn.metrics import accuracy_score
from io import BytesIO
import pandas as pd
from sklearn.impute import SimpleImputer

app = Flask(__name__)

UPLOAD_FOLDER = 'www/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Define the transportation mode Enum
class TransportationMode(Enum):
    CYCLING = 'cycling'
    DRIVING = 'driving'
    TRAIN = 'train'
    BUS = 'bus'
    SUBWAY = 'metro'
    TRAM = 'tram'
    ESCOOTER = 'e-scooter'

# Load the saved statistics
with open('model/3.0/statistics.pkl', 'rb') as f:
    loaded_statistics = pickle.load(f)

# Load the trained Random Forest model and imputer
rf_classifier = joblib.load('model/3.0/rf_trained_model-3.0.joblib')
imputer = joblib.load('model/3.0/imputer.joblib')
labels = np.load('model/3.0/label_encoder.npy')

def compute_magnitude(arrays):
    squares = [np.square(array) for array in arrays]
    sum_of_squares = np.sum(squares, axis=0)
    return np.sqrt(sum_of_squares)

def compute_jerk(data):
    return [data[i+1] - data[i] for i in range(len(data)-1)] + [0]


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

    mean_speed = loaded_statistics['mean_speed']
    std_speed = loaded_statistics['std_speed']
    mean_course = loaded_statistics['mean_course']
    std_course = loaded_statistics['std_course']
    mean_x = loaded_statistics['mean_x']
    std_x = loaded_statistics['std_x']
    mean_y = loaded_statistics['mean_y']
    std_y = loaded_statistics['std_y']
    mean_z = loaded_statistics['mean_z']
    std_z = loaded_statistics['std_z']
    mean_mx = loaded_statistics['mean_mx']
    std_mx = loaded_statistics['std_mx']
    mean_my = loaded_statistics['mean_my']
    std_my = loaded_statistics['std_my']
    mean_mz = loaded_statistics['mean_mz']
    std_mz = loaded_statistics['std_mz']
    mean_acc_magnitude = loaded_statistics['mean_acc_magnitude']
    std_acc_magnitude = loaded_statistics['std_acc_magnitude']
    mean_mag_magnitude = loaded_statistics['mean_mag_magnitude']
    std_mag_magnitude = loaded_statistics['std_mag_magnitude']
    mean_jerk_ax = loaded_statistics['mean_jerk_ax']
    std_jerk_ax = loaded_statistics['std_jerk_ax']
    mean_jerk_ay = loaded_statistics['mean_jerk_ay']
    std_jerk_ay = loaded_statistics['std_jerk_ay']
    mean_jerk_az = loaded_statistics['mean_jerk_az']
    std_jerk_az = loaded_statistics['std_jerk_az']
    mean_jerk_mx = loaded_statistics['mean_jerk_mx']
    std_jerk_mx = loaded_statistics['std_jerk_mx']
    mean_jerk_my = loaded_statistics['mean_jerk_my']
    std_jerk_my = loaded_statistics['std_jerk_my']
    mean_jerk_mz = loaded_statistics['mean_jerk_mz']
    std_jerk_mz = loaded_statistics['std_jerk_mz']

    # Then normalize the data based on the loaded statistics
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

    
    features = np.column_stack((
        normalized_speed,
        normalized_course,
        normalized_x,
        normalized_y,
        normalized_z,
        normalized_mx,
        normalized_my,
        normalized_mz,
        normalized_acc_magnitude,
        normalized_mag_magnitude,
        normalized_jerk_ax,
        normalized_jerk_ay,
        normalized_jerk_az,
        normalized_jerk_mx,
        normalized_jerk_my,
        normalized_jerk_mz
    ))

    return features, loaded_statistics

def load_and_extract_features(df):
    data = df.values
    speed = data[:, 1].astype(float)
    course = data[:, 2].astype(float)
    x = data[:, 3].astype(float)
    y = data[:, 4].astype(float)
    z = data[:, 5].astype(float)
    mx = data[:, 6].astype(float)
    my = data[:, 7].astype(float)
    mz = data[:, 8].astype(float)
    
    return speed, course, x, y, z, mx, my, mz


@app.route('/predict', methods=['POST'])
def predict():
    try:

        # Receive the uploaded file
        uploaded_file = request.files['file']
        
        if not uploaded_file:
            return jsonify({'error': 'Unauthorised access already reported'})


        # Load testing data
        data_loaded = pd.read_csv(BytesIO(uploaded_file.read()))
        print(f"Data loaded shape: {data_loaded.shape}")


        speed, course, x, y, z, mx, my, mz = load_and_extract_features(data_loaded)
        print(f"Extracted arrays shapes - speed: {speed.shape}, course: {course.shape}, x: {x.shape}, "
      f"y: {y.shape}, z: {z.shape}, mx: {mx.shape}, my: {my.shape}, mz: {mz.shape}")


        print(f"Before computing magnitude - acc: {len(x), len(y), len(z)}")
        acc_magnitudes = compute_magnitude([x, y, z])
        print(f"After computing magnitude - acc_magnitudes: {acc_magnitudes.shape}")

        print(f"Before computing magnitude - mat: {len(mx), len(my), len(mz)}")
        mag_magnitudes = compute_magnitude([mx, my, mz])
        print(f"After computing magnitude - mag: {mag_magnitudes.shape}")

        smoothed_acc_x = Preprocessing.apply_savitzky_golay(x)
        smoothed_acc_y = Preprocessing.apply_savitzky_golay(y)
        smoothed_acc_z = Preprocessing.apply_savitzky_golay(z)
        smoothed_mag_x = Preprocessing.apply_savitzky_golay(mx)
        smoothed_mag_y = Preprocessing.apply_savitzky_golay(my)
        smoothed_mag_z = Preprocessing.apply_savitzky_golay(mz)

        # Assuming you have your accelerometer data as:
        print(f"Before computing jerk - acc: {len(smoothed_acc_x), len(smoothed_acc_y), len(smoothed_acc_z)}")
        jerk_ax = compute_jerk(smoothed_acc_x)
        jerk_ay = compute_jerk(smoothed_acc_y)
        jerk_az = compute_jerk(smoothed_acc_z)
        print(f"After computing jerk - jerk_ax: {len(jerk_ax)}")


        # Assuming you have your magnetometer data as:
        jerk_mx = compute_jerk(smoothed_mag_x)
        jerk_my = compute_jerk(smoothed_mag_y)
        jerk_mz = compute_jerk(smoothed_mag_z)

        # Preprocess the testing data using statistics from the training data
        # print(f"Before preprocessing data lengths - speed: {len(speed)}, course: {len(course)}, ...")
        # features = preprocess_data(
        #     speed, course, x, y, z, jerk_ax, jerk_ay, jerk_az, acc_magnitudes, mx, my, mz, jerk_mx, jerk_my, jerk_mz, mag_magnitudes
        # )[0]

        features, _ = preprocess_data(
            speed, course, x, y, z, jerk_ax, jerk_ay, jerk_az, acc_magnitudes, 
            mx, my, mz, jerk_mx, jerk_my, jerk_mz, mag_magnitudes,
            mean_speed=loaded_statistics['mean_speed'], std_speed=loaded_statistics['std_speed'],
            mean_course=loaded_statistics['mean_course'], std_course=loaded_statistics['std_course'],
            mean_x=loaded_statistics['mean_x'], std_x=loaded_statistics['std_x'],
            mean_y=loaded_statistics['mean_y'], std_y=loaded_statistics['std_y'],
            mean_z=loaded_statistics['mean_z'], std_z=loaded_statistics['std_z'],
            mean_jerk_ax=loaded_statistics['mean_jerk_ax'], std_jerk_ax=loaded_statistics['std_jerk_ax'],
            mean_jerk_ay=loaded_statistics['mean_jerk_ay'], std_jerk_ay=loaded_statistics['std_jerk_ay'],
            mean_jerk_az=loaded_statistics['mean_jerk_az'], std_jerk_az=loaded_statistics['std_jerk_az'],
            mean_acc_magnitude=loaded_statistics['mean_acc_magnitude'], std_acc_magnitude=loaded_statistics['std_acc_magnitude'],
            mean_mx=loaded_statistics['mean_mx'], std_mx=loaded_statistics['std_mx'],
            mean_my=loaded_statistics['mean_my'], std_my=loaded_statistics['std_my'],
            mean_mz=loaded_statistics['mean_mz'], std_mz=loaded_statistics['std_mz'],
            mean_jerk_mx=loaded_statistics['mean_jerk_mx'], std_jerk_mx=loaded_statistics['std_jerk_mx'],
            mean_jerk_my=loaded_statistics['mean_jerk_my'], std_jerk_my=loaded_statistics['std_jerk_my'],
            mean_jerk_mz=loaded_statistics['mean_jerk_mz'], std_jerk_mz=loaded_statistics['std_jerk_mz'],
            mean_mag_magnitude=loaded_statistics['mean_mag_magnitude'], std_mag_magnitude=loaded_statistics['std_mag_magnitude']
        )

        print(f"Test features shape: {features.shape}")

        print(f"Before imputation - features: {features.shape}")
        data_imputed = imputer.transform(features)
        print(f"After imputation - data_imputed: {data_imputed.shape}")

         # Make predictions using the trained Random Forest model
        predicted_probabilities = rf_classifier.predict_proba(data_imputed)


        # for mode in TransportationMode:
        #     print(f"Name: {mode.name}")   # This will print the name of the enum member (a string)
        #     print(f"Value: {mode.value}")  # This will print the value of the enum member (whatever datatype it might be)

        # Define your transportation modes in the order they appear in predicted_probabilities
        modes_list = [TransportationMode.DRIVING, TransportationMode.CYCLING, TransportationMode.TRAIN]

        mode_probabilities = {}
        for mode in modes_list:
            index = modes_list.index(mode)  # Get the index of the mode in the modes_list
            mode_probabilities[mode.name] = predicted_probabilities[:, index].mean() * 100
        print(mode_probabilities)
        # Sort by probability
        sorted_modes = sorted(mode_probabilities.items(), key=lambda x: x[1], reverse=True)
        print(sorted_modes)
        return sorted_modes[0][0]

                
    except Exception as e:
        print(e)
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
