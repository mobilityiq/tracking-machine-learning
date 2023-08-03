import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Conv1D, Flatten
import tensorflow as tf
from scipy import stats

# Function to preprocess data
def preprocess_data_2d(data, sampling_rate, target_rate):
    # Calculate the downsampling factor
    factor = int(sampling_rate / target_rate)

    # Reshape the data to a 2D array where each row is a different sensor
    # and each column is a different timestep
    data = data.reshape((-1, data.shape[1] // factor, factor))

    # Downsample the data by taking the mean in each window
    downsampled = data.mean(axis=2)

    # Calculate the magnitude of the downsampled data
    magnitude = np.sqrt(np.sum(downsampled**2, axis=1))

    return magnitude

def extract_features_2d(data):
    # Calculate features for each sensor
    means = data.mean()
    vars = data.var()
    mins = data.min()
    maxs = data.max()

    return np.array([means, vars, mins, maxs])

# Load the data
acc_x = np.loadtxt('files/Acc_x.txt').reshape(-1, 6000)
acc_y = np.loadtxt('files/Acc_y.txt').reshape(-1, 6000)
acc_z = np.loadtxt('files/Acc_z.txt').reshape(-1, 6000)
mag_x = np.loadtxt('files/Mag_x.txt').reshape(-1, 6000)
mag_y = np.loadtxt('files/Mag_y.txt').reshape(-1, 6000)
mag_z = np.loadtxt('files/Mag_z.txt').reshape(-1, 6000)

# Here we're just loading labels as integers, assuming they start from 0
labels = np.loadtxt('files/Label.txt').reshape(-1, 6000).mean(axis=1).astype(int)
order = np.loadtxt('files/train_order.txt').astype(int) - 1

# Combine only the accelerometer and magnetometer data into a single 3D array
data = np.stack((acc_x, acc_y, acc_z, mag_x, mag_y, mag_z), axis=1)

# Preprocess and extract features from the data
data = np.array([
    extract_features_2d(preprocess_data_2d(series, sampling_rate=6000, target_rate=50))
    for series in data
])

# Reorder the data based on the order specified in train_order.txt
data = data[order]
labels = labels[order]

# Create stratified shuffle split object
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)

# Get train and validation indices
train_index, val_index = next(sss.split(data, labels))

# Split data into train and validation sets
X_train, X_val = data[train_index], data[val_index]
y_train, y_val = labels[train_index], labels[val_index]


# Create a Sequential model
def create_model():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(651, activation='softmax'))  # Adjust the number of units to match the number of classes in your problem
    return model

model = create_model()

opt = tf.keras.optimizers.Adam(learning_rate=0.001)  # Here is the optimizer with defined learning rate
early_stopping = EarlyStopping(monitor='val_loss', patience=3)  # Early stopping callback

model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, batch_size=100, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Evaluate model on the train and validation data
train_accuracy = model.evaluate(X_train, y_train)[1]
val_accuracy = model.evaluate(X_val, y_val)[1]

print(f"Train Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")
