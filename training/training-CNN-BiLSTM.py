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


def normalize_data(X_train, X_test):
    means = np.mean(X_train, axis=0)
    stds = np.std(X_train, axis=0)
    X_train_norm = (X_train - means) / stds
    X_test_norm = (X_test - means) / stds
    return X_train_norm, X_test_norm

def split_channels(X):
    return [X[:, i:i+3] if i < 11 else X[:, i:i+1] for i in range(0, X.shape[1], 3)]

def prep_data_for_model(X):
    return [channel[..., np.newaxis] for channel in split_channels(X)]

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def main():
    # Load data
    X_train, y_train = Preprocessing.load_and_process_data(locations=["Hand"])
    X_test, y_test = Preprocessing.load_and_process_data(locations=["Hand"], is_validation=True)

    # Normalize data
    X_train, X_test = normalize_data(X_train, X_test)

    # Prepare data for model input
    X_train_channels = prep_data_for_model(X_train)
    X_test_channels = prep_data_for_model(X_test)

    # One-hot encode labels
    y_train_encoded = to_categorical(y_train, num_classes=9)
    y_test_encoded = to_categorical(y_test, num_classes=9)

    assert set(np.unique(y_test)).issubset(set(np.unique(y_train))), "Validation set contains unseen labels!"


    print("Example of one-hot encoded training label:", y_train_encoded[0])
    print("Example of one-hot encoded validation label:", y_test_encoded[0])

    print("Training data shape:", X_train.shape)
    print("Validation data shape:", X_test.shape)
    print("Training labels shape:", y_train_encoded.shape)
    print("Validation labels shape:", y_test_encoded.shape)

    print("Training data type:", X_train.dtype)
    print("Validation data type:", X_test.dtype)
    print("Training labels data type:", y_train_encoded.dtype)
    print("Validation labels data type:", y_test_encoded.dtype)


    # Create model
    input_shapes = [channel.shape[1:] for channel in X_train_channels]
    # model = Models.create_multichannel_model(input_shapes=input_shapes)
    model = Models.create_simplified_multichannel_model(input_shapes=input_shapes)


    # Callbacks
    early_stopping = EarlyStopping(patience=3, restore_best_weights=True)
    checkpoint = ModelCheckpoint('../model/cnn-bi-lstm/cnn_bilstm_model', save_best_only=True)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

    # Compile model
    optimizer = legacy.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
    f1_metric = Preprocessing.f1_metric
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', f1_metric])

    # Train model
    history = model.fit(
        x=X_train_channels,
        y=y_train_encoded,
        validation_data=(X_test_channels, y_test_encoded),
        epochs=40,
        batch_size=1024,
        callbacks=[early_stopping, checkpoint, lr_reducer]
    )

    # Save model
    model.save('../model/cnn-bi-lstm/cnn_bilstm_model')

    # Plot training history
    plot_training_history(history)

if __name__ == '__main__':
    main()
