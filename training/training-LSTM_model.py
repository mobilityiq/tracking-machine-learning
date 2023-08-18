import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from enum import Enum
import matplotlib.pyplot as plt
from preprocessing import Preprocessing
from models import Models
from imblearn.over_sampling import SMOTE

# Phone was located in different parts to collect the data
locations = ["Bag","Hand","Hips","Torso"]

# Load data from the text file
data = Preprocessing.data_from_phone_locations(locations=locations)
data = np.array(data)

# Load validation data for test
test_data = Preprocessing.data_from_phone_locations(locations=locations,is_validation=True)
test_data = np.array(test_data)


# Extract relevant information from the loaded data
modes = data[:, -1]  # transportation modes
timestamps = data[:, 0].astype(float)  # timestamps
x = data[:, 1].astype(float)  # x accel value
y = data[:, 2].astype(float)  # y accel
z = data[:, 3].astype(float)  # z accel
mx = data[:, 4].astype(float)  # mx magnetometer value
my = data[:, 5].astype(float)  # my magnetometer
mz = data[:, 6].astype(float)  # mz magnetometer

# Extract relevant information from the loaded data
test_modes = test_data[:, -1]  # transportation modes
test_timestamps = test_data[:, 0].astype(float)  # timestamps
test_x = test_data[:, 1].astype(float)  # x accel value
test_y = test_data[:, 2].astype(float)  # y accel
test_z = test_data[:, 3].astype(float)  # z accel
test_mx = test_data[:, 4].astype(float)  # mx magnetometer value
test_my = test_data[:, 5].astype(float)  # my magnetometer
test_mz = test_data[:, 6].astype(float)  # mz magnetometer


# Perform any necessary preprocessing steps
# For example, you can normalize the sensor values

# Normalize timestamp, speed, x, y, and z values
def normalize(array):
    mean = np.mean(array)
    std = np.std(array)
    normalized = (array - mean) / std
    return normalized, mean, std

# Perform normalization on the sensor values
normalized_timestamp, mean_timestamp, std_timestamp = normalize(timestamps)
normalized_x, mean_x, std_x = normalize(x)
normalized_y, mean_y, std_y = normalize(y)
normalized_z, mean_z, std_z = normalize(z)
normalized_mx, mean_mx, std_mx = normalize(mx)
normalized_my, mean_my, std_my = normalize(my)
normalized_mz, mean_mz, std_mz = normalize(mz)

# Perform normalization on the sensor values
normalized_test_timestamp, mean_test_timestamp, std_test_timestamp = normalize(test_timestamps)
normalized_test_x, mean_test_x, std_test_x = normalize(test_x)
normalized_test_y, mean_test_y, std_test_y = normalize(test_y)
normalized_test_z, mean_test_z, std_test_z = normalize(test_z)
normalized_test_mx, mean_test_mx, std_test_mx = normalize(test_mx)
normalized_test_my, mean_test_my, std_test_my = normalize(test_my)
normalized_test_mz, mean_test_mz, std_test_mz = normalize(test_mz)

# Before label encoding
unique_modes = np.unique(modes)
print(f"Unique labels before encoding: {unique_modes}")

# Create and fit the label encoder on the training data
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(modes)

# Use the same encoder to transform labels for the test data
test_encoded_labels = label_encoder.transform(test_modes)


num_classes = len(Preprocessing.LABEL_MAP)

# After label encoding
print("A few samples of encoded labels:")
for original, encoded in zip(modes[:10], encoded_labels[:10]):
    print(f"{original} -> {encoded}")

# Debugging
unique, counts = np.unique(encoded_labels, return_counts=True)
print(dict(zip(label_encoder.inverse_transform(unique), counts)))

# Get the list of transportation mode labels
labels = label_encoder.classes_.tolist()
test_labels = label_encoder.classes_.tolist()


# Combine normalized sensor values into features
features = np.column_stack((normalized_timestamp, normalized_x, normalized_y, normalized_z, normalized_mx, normalized_my, normalized_mz))
test_features = np.column_stack((normalized_test_timestamp, normalized_test_x, normalized_test_y, normalized_test_z, normalized_test_mx, normalized_test_my, normalized_test_mz))

# Split the data into training and testing sets
# train_features, test_features, train_labels, test_labels = train_test_split(features, encoded_labels, test_size=0.2)

train_features = features
train_labels = encoded_labels

# # Apply SMOTE to the training data
# print("Using SMOTE to balance the classes")
# smote = SMOTE()
# train_features, train_labels = smote.fit_resample(train_features, train_labels)
# # Now, the training data should have roughly equal number of instances for each class
# print("Finished balancing the classes")


# Load the data into training and testing sets
train_features = train_features[:, np.newaxis, :]
test_features = test_features[:, np.newaxis, :]


# Define learning rate schedule function
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 10:
        lr *= 1e-1
    elif epoch > 20:
        lr *= 5e-2  # Consider making this a less aggressive decay
    print('Learning rate: ', lr)
    return lr


model = Models.create_lstm_model(num_classes=num_classes)

# Define callbacks
early_stopping = EarlyStopping(patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint('../model/lstm/trained_lstm_model_{epoch:02d}_{val_loss:.4f}', save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

# Compile and fit the model
f1_metric = Preprocessing.f1_metric
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', f1_metric])


# One-hot encode the labels
train_labels = to_categorical(train_labels, num_classes=num_classes)
test_labels = to_categorical(test_encoded_labels, num_classes=num_classes)

# Save the label encoder
np.save('../model/lstm/label_encoder.npy', label_encoder.classes_)

# Save the mean and standard deviation
np.save('../model/lstm/mean.npy', [mean_timestamp, mean_x, mean_y, mean_z, mean_mx, mean_my, mean_mz])
np.save('../model/lstm/std.npy', [std_timestamp, std_x, std_y, std_z, std_mx, std_my, std_mz])

history = model.fit(train_features, train_labels, epochs=40, batch_size=1024, 
                    validation_data=(test_features, test_labels),
                    callbacks=[early_stopping, checkpoint, lr_scheduler])


# Save the trained model 
model.save('../model/lstm/trained_lstm_model')

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
