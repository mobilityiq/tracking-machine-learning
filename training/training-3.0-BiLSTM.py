import sys
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from joblib import dump
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from enum import Enum
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Dropout
from keras.optimizers import Adam
from scipy.signal import savgol_filter
from keras import backend as K
import coremltools as ct
import json

# Define the transportation mode Enum
class TransportationMode(Enum):
    BUS = 'bus'
    CYCLING = 'cycling'
    DRIVING = 'driving'
    STATIONARY = 'stationary'
    TRAIN = 'train'
    WALKING = 'walking'
    # SUBWAY = 'metro'
    # TRAM = 'tram'
    # ESCOOTER = 'e-scooter'

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



def apply_savitzky_golay(data, window_length=5, polynomial_order=2):
    """
    Apply Savitzky-Golay filter to data.
    
    Parameters:
    - data: The input data (e.g., a list or numpy array)
    - window_length: The length of the filter window (should be an odd integer). Default is 5.
    - polynomial_order: The order of the polynomial used to fit the samples. Default is 2.
    
    Returns:
    - Smoothed data
    """
    return savgol_filter(data, window_length, polynomial_order)


# Check if the data file is provided as a command-line argument
if len(sys.argv) == 2:
    # Get the data file path from the command-line argument
    data_file = sys.argv[1]
else:
    data_file = "training-3.0.csv"

# Load training data
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(f"{current_time} - Loading data")
timestamp, speed, course, x, y, z, mx, my, mz, modes = load_and_extract_features(data_file)
 
# Compute magnitudes
acc_magnitude = compute_magnitude([x, y, z])
mag_magnitude = compute_magnitude([mx, my, mz])

# Apply the filters
# smoothed_acc_x = apply_savitzky_golay(x)
# smoothed_acc_y = apply_savitzky_golay(y)
# smoothed_acc_z = apply_savitzky_golay(z)
# smoothed_mag_x = apply_savitzky_golay(mx)
# smoothed_mag_y = apply_savitzky_golay(my)
# smoothed_mag_z = apply_savitzky_golay(mz)

# Calculating jerks
# jerk_ax = compute_jerk(smoothed_acc_x)
# jerk_ay = compute_jerk(smoothed_acc_y)
# jerk_az = compute_jerk(smoothed_acc_z)
# jerk_mx = compute_jerk(smoothed_mag_x)
# jerk_my = compute_jerk(smoothed_mag_y)
# jerk_mz = compute_jerk(smoothed_mag_z)

jerk_ax = compute_jerk(x)
jerk_ay = compute_jerk(y)
jerk_az = compute_jerk(z)
jerk_mx = compute_jerk(mx)
jerk_my = compute_jerk(my)
jerk_mz = compute_jerk(mz)

# Encode transportation modes as numerical labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(modes)
num_classes = len(TransportationMode)

# Preprocess the training data
train_features, train_statistics = preprocess_data(speed, course, x, y, z, jerk_ax, jerk_ay, jerk_az, acc_magnitude, mx, my, mz, jerk_mx, jerk_my, jerk_mz, mag_magnitude)

train_labels = label_encoder.transform(modes)

# Apply SMOTE to the training data
print(f"Using SMOTE to balance the classes: {train_features.shape}")
smote = SMOTE()
train_features, train_labels = smote.fit_resample(train_features, train_labels)
# Now, the training data should have roughly equal number of instances for each class
print(f"Finished balancing the classes: {train_features.shape}")

# Get the list of transportation mode labels
labels = label_encoder.classes_.tolist()

# Convert labels to one-hot encoding
train_labels_one_hot = to_categorical(train_labels, num_classes=num_classes)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels_one_hot, test_size=0.2, random_state=42)

# Initialize the BiLSTM model
model = Sequential()
model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=(X_train.shape[1], 1)))
model.add(Bidirectional(LSTM(50)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


# Define learning rate schedule function
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 10:
        lr *= 1e-1
    elif epoch > 20:
        lr *= 5e-2  # Consider making this a less aggressive decay
    print('Learning rate: ', lr)
    return lr

def f1_metric(y_true, y_pred):
    """
    Compute the F1 score, also known as balanced F-score or F-measure.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

# Define callbacks
early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('../model/bi-lstm/trained_bi-lstm_model_{epoch:02d}_{val_loss:.4f}', save_best_only=True)
lr_scheduler = LearningRateScheduler(lr_schedule)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', f1_metric])

print(X_train.shape)

print("Before Reshape: ", X_train.shape)  # Print shape before reshaping
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
print("After Reshape: ", X_train.shape)  # Print shape after reshaping

model.summary()


# Fit the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=40, batch_size=1024, verbose=1)

# Save statistics
with open('../model/bi-lstm/statistics.pkl', 'wb') as f:
    pickle.dump(train_statistics, f)

# Save the label encoder
np.save('../model/bi-lstm/label_encoder.npy', label_encoder.classes_)

# Save the model
model.save('../model/bi-lstm/trained_bi-lstm_model')

# Create a dictionary to hold the metadata
metadata = {
    'statistics': train_statistics,
    'labels': ['bus','cycling','driving','stationary','train','walking']
}

# Convert the metadata dictionary to JSON string
metadata_json = json.dumps(metadata)

# Save the metadata as a JSON file
with open('../model/bi-lstm/metadata.json', 'w') as f:
    f.write(metadata_json)

statistics_json = json.dumps(train_statistics)

with open('../model/bi-lstm/statistics.json', 'w') as f:
    f.write(statistics_json)

# coreml_model = ct.convert(model, inputs=[input_feature], source='tensorflow')
coreml_model = ct.convert(model)

# Add the metadata to the model as user-defined metadata
coreml_model.user_defined_metadata['preprocessing_metadata'] = metadata_json

# Set the prediction_type to "probability"
coreml_model.user_defined_metadata['prediction_type'] = 'probability'

# Save the Core ML model
coreml_model.save('../model/bi-lstm/bi-lstm.mlmodel')

# ** Evaluate on the test set **
print("*** Evaluating model ***")
# Load the labels
loaded_classes = np.load('../model/bi-lstm/label_encoder.npy')
label_encoder = LabelEncoder()
label_encoder.classes_ = loaded_classes
# Load the statistics
with open('../model/bi-lstm/statistics.pkl', 'rb') as f:
    statistics = pickle.load(f)

def preprocess_test_data(speed, course, x, y, z, jerk_ax, jerk_ay, jerk_az, acc_magnitude, mx, my, mz, jerk_mx, jerk_my, jerk_mz, mag_magnitude):    

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

# Load testing data
data_test_file = 'testing-3.0.csv'
test_timestamp, test_speed, test_course, test_x, test_y, test_z, test_mx, test_my, test_mz, test_modes = load_and_extract_features(data_test_file)

# Compute magnitudes
acc_test_magnitudes = compute_magnitude([test_x, test_y, test_z])
mag_test_magnitudes = compute_magnitude([test_mx, test_my, test_mz])

# Apply the filters
smoothed_acc_test_x = apply_savitzky_golay(test_x)
smoothed_acc_test_y = apply_savitzky_golay(test_y)
smoothed_acc_test_z = apply_savitzky_golay(test_z)
smoothed_mag_test_x = apply_savitzky_golay(test_mx)
smoothed_mag_test_y = apply_savitzky_golay(test_my)
smoothed_mag_test_z = apply_savitzky_golay(test_mz)

# Compute jerks
jerk_test_ax = compute_jerk(smoothed_acc_test_x)
jerk_test_ay = compute_jerk(smoothed_acc_test_y)
jerk_test_az = compute_jerk(smoothed_acc_test_z)
jerk_test_mx = compute_jerk(smoothed_mag_test_x)
jerk_test_my = compute_jerk(smoothed_mag_test_y)
jerk_test_mz = compute_jerk(smoothed_mag_test_z)

# Preprocess the testing data using statistics from the training data
X_test = preprocess_test_data(test_speed, test_course, test_x, test_y, test_z, jerk_test_ax, jerk_test_ay, jerk_test_az, acc_test_magnitudes, test_mx, test_my, test_mz, jerk_test_mx, jerk_test_my, jerk_test_mz, mag_test_magnitudes)

X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1)) # Assuming that the LSTM expects a 3D input. Adjust if otherwise.
test_labels_one_hot = to_categorical(label_encoder.transform(test_modes), num_classes=num_classes)
loss, accuracy, f1_score = model.evaluate(X_test, test_labels_one_hot, verbose=0)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"F1 Score: {f1_score * 100:.2f}%")



