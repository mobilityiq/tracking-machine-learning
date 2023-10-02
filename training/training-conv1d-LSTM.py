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
from keras.layers import Dense, LSTM
from scipy.signal import savgol_filter
from keras import backend as K
import coremltools as ct
import json
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Conv1D, BatchNormalization, LSTM, Dense
from models import Models
from preprocessing import Preprocessing

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



# Check if the data file is provided as a command-line argument
if len(sys.argv) == 2:
    # Get the data file path from the command-line argument
    data_file = sys.argv[1]
else:
    data_file = "training-3.0.csv"

# Load training data
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(f"{current_time} - Loading data")
timestamp, speed, course, x, y, z, mx, my, mz, modes = Preprocessing.load_and_extract_features(data_file)

# Compute magnitudes
acc_magnitude = Preprocessing.compute_magnitude([x, y, z])
mag_magnitude = Preprocessing.compute_magnitude([mx, my, mz])

# Calculating jerks
jerk_ax = Preprocessing.compute_jerk(x)
jerk_ay = Preprocessing.compute_jerk(y)
jerk_az = Preprocessing.compute_jerk(z)
jerk_mx = Preprocessing.compute_jerk(mx)
jerk_my = Preprocessing.compute_jerk(my)
jerk_mz = Preprocessing.compute_jerk(mz)

# Encode transportation modes as numerical labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(modes)
num_classes = len(TransportationMode)

# Preprocess the training data
train_features, train_statistics =Preprocessing. preprocess_data(speed, course, x, y, z, jerk_ax, jerk_ay, jerk_az, acc_magnitude, mx, my, mz, jerk_mx, jerk_my, jerk_mz, mag_magnitude)

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


model = Models.create_conv1d_lstm_model(num_classes=num_classes, input_shape=(X_train.shape[1], 1))


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
checkpoint = ModelCheckpoint('../model/conv1d-lstm/trained_conv1d-lstm_model_{epoch:02d}_{val_loss:.4f}', save_best_only=True)
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
with open('../model/conv1d-lstm/statistics.pkl', 'wb') as f:
    pickle.dump(train_statistics, f)

# Save the label encoder
np.save('../model/conv1d-lstm/label_encoder.npy', label_encoder.classes_)

# Save the model to path
model.save('../model/conv1d-lstm/trained_conv1d-lstm_model')

# Create a dictionary to hold the metadata
metadata = {
    'statistics': train_statistics,
    'labels': ['bus','cycling','driving','stationary','train','walking']
}

# Convert the metadata dictionary to JSON string
metadata_json = json.dumps(metadata)

# Save the metadata as a JSON file
with open('../model/conv1d-lstm/metadata.json', 'w') as f:
    f.write(metadata_json)

statistics_json = json.dumps(train_statistics)

with open('../model/conv1d-lstm/statistics.json', 'w') as f:
    f.write(statistics_json)

# coreml_model = ct.convert(model, inputs=[input_feature], source='tensorflow')
coreml_model = ct.convert(model)

# Add the metadata to the model as user-defined metadata
coreml_model.user_defined_metadata['preprocessing_metadata'] = metadata_json

# Set the prediction_type to "probability"
coreml_model.user_defined_metadata['prediction_type'] = 'probability'

# Save the Core ML model
coreml_model.save('../model/conv1d-lstm/conv1d-lstm.mlmodel')

# ** Evaluate on the test set **
print("*** Evaluating model ***")
# Load the labels
loaded_classes = np.load('../model/conv1d-lstm/label_encoder.npy')
label_encoder = LabelEncoder()
label_encoder.classes_ = loaded_classes

# Load testing data
data_test_file = 'testing-3.0.csv'
test_timestamp, test_speed, test_course, test_x, test_y, test_z, test_mx, test_my, test_mz, test_modes = Preprocessing.load_and_extract_features(data_test_file)


# Compute magnitudes
acc_test_magnitudes = Preprocessing.compute_magnitude([test_x, test_y, test_z])
mag_test_magnitudes = Preprocessing.compute_magnitude([test_mx, test_my, test_mz])

# Compute jerks
jerk_test_ax = Preprocessing.compute_jerk(test_x)
jerk_test_ay = Preprocessing.compute_jerk(test_y)
jerk_test_az = Preprocessing.compute_jerk(test_z)
jerk_test_mx = Preprocessing.compute_jerk(test_mx)
jerk_test_my = Preprocessing.compute_jerk(test_my)
jerk_test_mz = Preprocessing.compute_jerk(test_mz)


# Preprocess the testing data using statistics from the training data
X_test = Preprocessing.preprocess_test_data(test_speed, test_course, test_x, test_y, test_z, jerk_test_ax, jerk_test_ay, jerk_test_az, acc_test_magnitudes, test_mx, test_my, test_mz, jerk_test_mx, jerk_test_my, jerk_test_mz, mag_test_magnitudes, train_statistics)

X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1)) # Assuming that the LSTM expects a 3D input. Adjust if otherwise.
test_labels_one_hot = to_categorical(label_encoder.transform(test_modes), num_classes=num_classes)
loss, accuracy, f1_score = model.evaluate(X_test, test_labels_one_hot, verbose=0)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"F1 Score: {f1_score * 100:.2f}%")


converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Depending on your model, you might need to set these options
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, 
    tf.lite.OpsSet.SELECT_TF_OPS
]

tflite_model = converter.convert()

with open('../model/conv1d-lstm/conv1d-lstm-lite.tflite', 'wb') as f:
    f.write(tflite_model)


  # Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="../model/conv1d-lstm/conv1d-lstm-lite.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print the input shape
print("Input Shape: ", input_details[0]['shape'])