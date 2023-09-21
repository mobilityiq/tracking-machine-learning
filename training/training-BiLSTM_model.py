import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from enum import Enum
from preprocessing import Preprocessing
from models import Models
from transportation_mode import TransportationMode
from datetime import datetime


# Load the preprocessed data
locations = ["Hand"]  # Replace with your actual list of locations
X_train, y_train = Preprocessing.load_and_process_data(locations, is_validation=True)
X_test, y_test = Preprocessing.load_and_process_data(locations, is_validation=True)

current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(f"{current_time} - Data loaded")

assert set(np.unique(y_test)).issubset(set(np.unique(y_train))), "Validation set contains unseen labels!"

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

 # Normalize data
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(f"{current_time} - Normalize data")
X_train, X_test, means, stds = Preprocessing.normalize_data(X_train, X_test) 

# X_train = pd.DataFrame(X_train)
# X_test = pd.DataFrame(X_test)


# Before label encoding
unique_modes = np.unique(y_train)
num_classes = len(unique_modes)

sequence_length = X_train.shape[0]  # Assuming X_train is of shape (num_samples, sequence_length)
num_features = X_train.shape[1]

current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(f"{current_time} - Creating model")
model = Models.create_bidirectional_lstm_model(num_classes, sequence_length, num_features)

# Define learning rate schedule function
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 10:
        lr *= 1e-1
    elif epoch > 20:
        lr *= 5e-2  # Consider making this a less aggressive decay
    print('Learning rate: ', lr)
    return lr

# Define callbacks
early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('../model/bi-lstm/trained_bi-lstm_model_{epoch:02d}_{val_loss:.4f}', save_best_only=True)
lr_scheduler = LearningRateScheduler(lr_schedule)

# Compile and fit the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(f"{current_time} - Encoding labels")
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

train_labels = to_categorical(y_train_encoded, num_classes=num_classes)
test_labels = to_categorical(y_test_encoded, num_classes=num_classes)

# Save the label encoder
np.save('../model/bi-lstm/label_encoder.npy', le.classes_)

# Save the mean and standard deviation
np.save('../model/bi-lstm/mean.npy', means)
np.save('../model/bi-lstm/std.npy', stds)

current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(f"{current_time} - Training model")

history = model.fit(X_train, train_labels, epochs=20, batch_size=1024, 
                    validation_data=(X_test, test_labels),
                    callbacks=[early_stopping, checkpoint, lr_scheduler])


# Save the trained model
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(f"{current_time} - Saving model")
model.save('../model/bi-lstm/trained_bi-lstm_model')

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
