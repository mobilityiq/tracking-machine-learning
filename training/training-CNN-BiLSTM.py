import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from keras.optimizers import legacy
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from preprocessing import Preprocessing
from models import Models
# from imblearn.over_sampling import SMOTE
from keras.optimizers import Adam


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

early_stop = EarlyStopping(monitor='val_loss', patience=5)

# Phone was located in different parts to collect the data
locations = ["Bag","Hand","Hips","Torso"]

# Import the data
data = Preprocessing.data_from_phone_locations_for_cnn_bilstm(locations=locations)
data = np.array(data)

# Load validation data for test
test_data = Preprocessing.data_from_phone_locations_for_cnn_bilstm(locations=locations,is_validation=True)
test_data = np.array(test_data)

X_train = data[:, :-1]  # All columns except the last one
y_train = data[:, -1].astype(int)  # Last column

# Extract the respective data from test_data
X_test = test_data[:, :-1]  # All columns except the last one
y_test = test_data[:, -1].astype(int)  # Last column

# Calculate mean and standard deviation for each channel
mean_acc = np.mean(X_train[:, :3], axis=0)
std_acc = np.std(X_train[:, :3], axis=0)

mean_jerk = np.mean(X_train[:, 3:6], axis=0)
std_jerk = np.std(X_train[:, 3:6], axis=0)

mean_acc_mag = np.mean(X_train[:, 6:7], axis=0)
std_acc_mag = np.std(X_train[:, 6:7], axis=0)

mean_mag_jerk = np.mean(X_train[:, 7:10], axis=0)
std_mag_jerk = np.std(X_train[:, 7:10], axis=0)

mean_mag_mag = np.mean(X_train[:, 10:11], axis=0)
std_mag_mag = np.std(X_train[:, 10:11], axis=0)

mean_magnetic = np.mean(X_train[:, 11:14], axis=0)
std_magnetic = np.std(X_train[:, 11:14], axis=0)


# Normalize using means and stds from training data
X_test_acc = (X_test[:, :3] - mean_acc) / std_acc
X_test_jerk = (X_test[:, 3:6] - mean_jerk) / std_jerk
X_test_acc_mag = (X_test[:, 6:7] - mean_acc_mag) / std_acc_mag
X_test_mag_jerk = (X_test[:, 7:10] - mean_mag_jerk) / std_mag_jerk
X_test_mag_mag = (X_test[:, 10:11] - mean_mag_mag) / std_mag_mag
X_test_magnetic = (X_test[:, 11:14] - mean_magnetic) / std_magnetic

# Reshape the data for model input
X_test_acc = X_test_acc[..., np.newaxis]
X_test_jerk = X_test_jerk[..., np.newaxis]
X_test_acc_mag = X_test_acc_mag[..., np.newaxis]
X_test_mag_jerk = X_test_mag_jerk[..., np.newaxis]
X_test_mag_mag = X_test_mag_mag[..., np.newaxis]
X_test_magnetic = X_test_magnetic[..., np.newaxis]

X_test_channels = [X_test_acc, X_test_jerk, X_test_acc_mag, X_test_mag_jerk, X_test_mag_mag, X_test_magnetic]

# Convert labels to one-hot encoding
y_test_encoded = to_categorical(y_test, num_classes=9)


# Extract the respective data for each channel from training data
X_train_acc = X_train[:, :3]  # Acceleration x, y, z
X_train_jerk = X_train[:, 3:6]  # Jerk x, y, z
X_train_acc_mag = X_train[:, 6:7]  # Acceleration magnitude
X_train_mag_jerk = X_train[:, 7:10]  # Magnetic jerk x, y, z
X_train_mag_mag = X_train[:, 10:11]  # Magnetic magnitude
X_train_magnetic = X_train[:, 11:14]  # Magnetic field x, y, z

# Normalize training data
X_train_acc = (X_train_acc - mean_acc) / std_acc
X_train_jerk = (X_train_jerk - mean_jerk) / std_jerk
X_train_acc_mag = (X_train_acc_mag - mean_acc_mag) / std_acc_mag
X_train_mag_jerk = (X_train_mag_jerk - mean_mag_jerk) / std_mag_jerk
X_train_mag_mag = (X_train_mag_mag - mean_mag_mag) / std_mag_mag
X_train_magnetic = (X_train_magnetic - mean_magnetic) / std_magnetic


X_train_acc = X_train_acc[..., np.newaxis]
X_train_jerk = X_train_jerk[..., np.newaxis]
X_train_acc_mag = X_train_acc_mag[..., np.newaxis]
X_train_mag_jerk = X_train_mag_jerk[..., np.newaxis]
X_train_mag_mag = X_train_mag_mag[..., np.newaxis]
X_train_magnetic = X_train_magnetic[..., np.newaxis]

# Ensure all arrays have the same number of dimensions before stacking
X_combined = np.hstack([
    X_train_acc.reshape(X_train_acc.shape[0], -1),
    X_train_jerk.reshape(X_train_jerk.shape[0], -1),
    X_train_acc_mag.reshape(X_train_acc_mag.shape[0], -1),
    X_train_mag_jerk.reshape(X_train_mag_jerk.shape[0], -1),
    X_train_mag_mag.reshape(X_train_mag_mag.shape[0], -1),
    X_train_magnetic.reshape(X_train_magnetic.shape[0], -1)
])


# Instantiate SMOTE
# smote = SMOTE(random_state=42)
print("Using SMOTE to balance the classes")
# # Apply SMOTE
# X_sm, y_sm = smote.fit_resample(X_combined, y_train)

# Split the SMOTE'd data back into their channels
# X_train_acc = X_sm[:, :3][..., np.newaxis]
# X_train_jerk = X_sm[:, 3:6][..., np.newaxis]
# X_train_acc_mag = X_sm[:, 6:7][..., np.newaxis]
# X_train_mag_jerk = X_sm[:, 7:10][..., np.newaxis]
# X_train_mag_mag = X_sm[:, 10:11][..., np.newaxis]
# X_train_magnetic = X_sm[:, 11:14][..., np.newaxis]
# y_train = y_sm


# Now that the X_train_... variables are defined, you can create the input_shapes list:
input_shapes = [
    X_train_acc.shape[1:], 
    X_train_jerk.shape[1:], 
    X_train_acc_mag.shape[1:], 
    X_train_mag_jerk.shape[1:], 
    X_train_mag_mag.shape[1:], 
    X_train_magnetic.shape[1:]
]

# Create the model
model = Models.create_multichannel_model(input_shapes=input_shapes)

# Define learning rate 
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)


# Define callbacks
early_stopping = EarlyStopping(patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint('../model/cnn-bi-lstm/cnn_bilstm_model', save_best_only=True)


# optimizer = legacy.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)



# Compile and fit the model
f1_metric = Preprocessing.f1_metric
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', f1_metric])

X_train_channels = [X_train_acc, X_train_jerk, X_train_acc_mag, X_train_mag_jerk, X_train_mag_mag, X_train_magnetic]

# Extract the respective data for each channel from test data
X_test_acc = X_test[:, :3]
X_test_jerk = X_test[:, 3:6]
X_test_acc_mag = X_test[:, 6:7]
X_test_mag_jerk = X_test[:, 7:10]
X_test_mag_mag = X_test[:, 10:11]
X_test_magnetic = X_test[:, 11:14]

# Normalize test data
X_test_acc = (X_test_acc - mean_acc) / std_acc
X_test_jerk = (X_test_jerk - mean_jerk) / std_jerk
X_test_acc_mag = (X_test_acc_mag - mean_acc_mag) / std_acc_mag
X_test_mag_jerk = (X_test_mag_jerk - mean_mag_jerk) / std_mag_jerk
X_test_mag_mag = (X_test_mag_mag - mean_mag_mag) / std_mag_mag
X_test_magnetic = (X_test_magnetic - mean_magnetic) / std_magnetic


X_test_channels = [X_test_acc, X_test_jerk, X_test_acc_mag, X_test_mag_jerk, X_test_mag_mag, X_test_magnetic]


# Convert labels to one-hot encoding
y_train_encoded = to_categorical(y_train, num_classes=9)
y_test_encoded = to_categorical(y_test, num_classes=9)

print(np.unique(y_train))


X_test_acc = X_test_acc[..., np.newaxis]
X_test_jerk = X_test_jerk[..., np.newaxis]
X_test_acc_mag = X_test_acc_mag[..., np.newaxis]
X_test_mag_jerk = X_test_mag_jerk[..., np.newaxis]
X_test_mag_mag = X_test_mag_mag[..., np.newaxis]
X_test_magnetic = X_test_magnetic[..., np.newaxis]


# Ensure every array is 1D
mean_acc = mean_acc.ravel()
mean_jerk = mean_jerk.ravel()
mean_mag_jerk = mean_mag_jerk.ravel()
mean_magnetic = mean_magnetic.ravel()

# Concatenate all 1D arrays
means = np.concatenate([
    mean_acc.ravel(),
    mean_jerk.ravel(),
    mean_acc_mag.ravel(),
    mean_mag_jerk.ravel(),
    mean_mag_mag.ravel(),
    mean_magnetic.ravel()
])

# Ensure every array is 1D
std_acc_acc = std_acc.ravel()
std_jerk = std_jerk.ravel()
std_mag_jerk = std_mag_jerk.ravel()
std_magnetic = std_magnetic.ravel()

# Concatenate all 1D arrays

stds = np.concatenate([
    std_acc.ravel(),
    std_jerk.ravel(),
    std_acc_mag.ravel(),
    std_mag_jerk.ravel(),
    std_mag_mag.ravel(),
    std_magnetic.ravel()
])

# Initialize label encoder
label_encoder = LabelEncoder()

# Fit label encoder and transform the labels 
y = label_encoder.fit_transform(y_train)

# Save the label encoder
np.save('../model/cnn-bi-lstm/label_encoder.npy', label_encoder.classes_)
np.save('../model/cnn-bi-lstm/training_means.npy', means)
np.save('../model/cnn-bi-lstm/std.npy', stds)

print(y_train_encoded.shape)

# Training the model
history = model.fit(
    x=X_train_channels,
    y=y_train_encoded,
    validation_data=(X_test_channels, y_test_encoded),
    epochs=40,
    batch_size=1024,
    callbacks=[early_stop, checkpoint, lr_reducer]
)

# Save the model
model.save('../model/cnn-bi-lstm/cnn_bilstm_model')

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()