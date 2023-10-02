import sys
import pickle
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
from enum import Enum
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras import backend as K
import coremltools as ct
from models import Models
from preprocessing import Preprocessing
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix, classification_report


class TransportationMode(Enum):
    BUS = 'bus'
    CYCLING = 'cycling'
    DRIVING = 'driving'
    STATIONARY = 'stationary'
    TRAIN = 'train'
    WALKING = 'walking'


class TransportModeClassifier:
    def __init__(self, data_file, is_training=True):
        self.data_file = data_file
        self.label_encoder = LabelEncoder()
        self.num_classes = len(TransportationMode)
        self.model = None
        self.train_features = None
        self.train_labels = None
        self.train_statistics = None
        self.is_training = is_training  # Added this line

    def load_data(self):
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Loading data")
        data = Preprocessing.load_and_extract_features(self.data_file)
        return data

    def preprocess_and_encode_data(self, data):
        timestamp, speed, course, x, y, z, mx, my, mz, modes = data
        theData = [x, y, z, mx, my, mz]  # Replace with actual series
        smoothed_data = Preprocessing.apply_savitzky_golay_to_all(theData)
        
        # Compute magnitudes
        acc_magnitude = np.sqrt(np.square(smoothed_data[:3]).sum(axis=0))  # Assuming smoothed_data[:3] gives [x, y, z]
        mag_magnitude = np.sqrt(np.square(smoothed_data[3:]).sum(axis=0))  # Assuming smoothed_data[3:] gives [mx, my, mz]

        # If this method is used for training data, fit the encoder, else just transform
        if self.is_training:
            encoded_labels = self.label_encoder.fit_transform(modes)
        else:
            encoded_labels = self.label_encoder.transform(modes)
        
        processed_data, self.train_statistics = Preprocessing.preprocess_data_no_jerks(
            timestamp, speed, course, *smoothed_data, acc_magnitude, mag_magnitude
        )
        return processed_data, encoded_labels


    def balance_classes(self, features, labels):
        print(f"Using SMOTE to balance the classes: {features.shape}")
        smote = SMOTE()
        balanced_features, balanced_labels = smote.fit_resample(features, labels)
        print(f"Finished balancing the classes: {balanced_features.shape}")
        return balanced_features, balanced_labels

    def split_data(self):
        X_train, X_val, y_train, y_val = train_test_split(
            self.train_features, self.train_labels, test_size=0.2, random_state=42
        )
        return X_train, X_val, y_train, y_val

    def compile_and_train_model(self, X_train, y_train, X_val, y_val):
        input_shapes = {
            'shape1': (3, 1),  # Corresponding to 'timestamp', 'speed', 'course'
            'shape2': (4, 1),  # Corresponding to 'timestamp', 'ax', 'ay', 'az'
            'shape3': (4, 1),  # Corresponding to 'timestamp', 'mx', 'my', 'mz'
            'shape4': (3, 1)   # Corresponding to 'timestamp', 'acc_magnitude', 'mag_magnitude
        }

        self.model = Models.create_multi_input_conv1d_model(
            num_classes=self.num_classes, 
            input_shape1=input_shapes['shape1'], 
            input_shape2=input_shapes['shape2'], 
            input_shape3=input_shapes['shape3'],
            input_shape4=input_shapes['shape4']
        )

        self.model.summary()

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy', self.f1_metric]
        )

        early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
        checkpoint = ModelCheckpoint(
            '../model/multi-conv1d/trained_conv1d_model_{epoch:02d}_{val_loss:.4f}',
            save_best_only=True
        )

        lr_scheduler = LearningRateScheduler(self.lr_schedule)

        reshaped_X_train = self.reshape_and_scale_data(X_train)

        y_train = to_categorical(y_train, num_classes=6)
        y_val = to_categorical(y_val, num_classes=6)

        # Apply the same transformations to validation data
        reshaped_X_val = self.reshape_and_scale_data(X_val)

        history = self.model.fit(
            reshaped_X_train, 
            y_train, 
            epochs=20, 
            batch_size=1024, 
            verbose=1,
            callbacks=[early_stopping, checkpoint, lr_scheduler],
            validation_data=(reshaped_X_val, y_val) 
        )

        return history
    
    def evaluate_model(self, test_file):
        # Load and preprocess the test data similar to training data
        test_data = Preprocessing.load_and_extract_features(test_file)
        X_test, y_test = self.preprocess_and_encode_data(test_data)

        # Transform labels back to original encoding
        y_true = self.label_encoder.inverse_transform(y_test)
        
        # Get predictions
        reshaped_X_test = self.reshape_and_scale_data(X_test)
        predictions = self.model.predict(reshaped_X_test)
        y_pred = np.argmax(predictions, axis=1)
        y_pred = self.label_encoder.inverse_transform(y_pred)
        
        # Converting unique labels in y_true to lowercase
        available_labels = set(y_true)
        lowercase_available_labels = {label.lower() for label in available_labels}

        labels_for_cm = [label.name.lower() for label in TransportationMode if label.name.lower() in lowercase_available_labels]
        
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels_for_cm)
        plt.figure(figsize=(10,7))
        sn.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_for_cm, yticklabels=labels_for_cm)
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.show()

        print(classification_report(y_true=y_true, y_pred=y_pred, labels=labels_for_cm, zero_division=1))


    @staticmethod
    def f1_metric(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())

        f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
        return f1_val

    @staticmethod
    def lr_schedule(epoch):
        lr = 1e-3
        if epoch > 10:
            lr *= 1e-1
        elif epoch > 20:
            lr *= 5e-2
        print('Learning rate: ', lr)
        return lr

    def reshape_and_scale_data(self, X):
        timestamp_speed_course_indices = [0, 1, 2]  # Example indices corresponding to 'timestamp', 'speed', 'course'
        timestamp_ax_ay_az_indices = [0, 3, 4, 5]  # Example indices corresponding to 'timestamp', 'ax', 'ay', 'az'
        timestamp_mx_my_mz_indices = [0, 6, 7, 8]  # Example indices corresponding to 'timestamp', 'mx', 'my', 'mz'
        timestamp_macc_mmag = [0, 9, 10] # Example indices corresponding to 'timestamp', 'acc_mag', 'mag_mag'

        X1 = MinMaxScaler().fit_transform(X[:, timestamp_speed_course_indices])
        X2 = MinMaxScaler().fit_transform(X[:, timestamp_ax_ay_az_indices])
        X3 = MinMaxScaler().fit_transform(X[:, timestamp_mx_my_mz_indices])
        X4 = MinMaxScaler().fit_transform(X[:, timestamp_macc_mmag])
        
        # Reshape the data after scaling.
        X1 = X1.reshape((X1.shape[0], X1.shape[1], 1))
        X2 = X2.reshape((X2.shape[0], X2.shape[1], 1))
        X3 = X3.reshape((X3.shape[0], X3.shape[1], 1))
        X4 = X4.reshape((X4.shape[0], X4.shape[1], 1))

        return [X1, X2, X3, X4]

    def save_model_and_metadata(self):
        self.model.save('../model/multi-conv1d/trained_conv1d_model')
        with open('../model/multi-conv1d/statistics.pkl', 'wb') as f:
            pickle.dump(self.train_statistics, f)

        np.save('../model/multi-conv1d/label_encoder.npy', self.label_encoder.classes_)

        metadata = {
            'statistics': self.train_statistics,
            'labels': [mode.value for mode in TransportationMode]
        }

        with open('../model/multi-conv1d/statistics.json', 'w') as f:
            json.dump(self.train_statistics, f)

        coreml_model = ct.convert(self.model)
        coreml_model.user_defined_metadata['preprocessing_metadata'] = json.dumps(metadata)
        coreml_model.user_defined_metadata['prediction_type'] = 'probability'
        coreml_model.save('../model/multi-conv1d/multi-conv1d.mlmodel')

        
    def export_to_lite(self):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()

        with open('../model/multi-conv1d/multi-conv1d-lite.tflite', 'wb') as f:
            f.write(tflite_model)

    def run(self):
        data = self.load_data()
        self.train_features, self.train_labels = self.preprocess_and_encode_data(data)
        # self.train_features, self.train_labels = self.balance_classes(self.train_features, self.train_labels)
        X_train, X_val, y_train, y_val = self.split_data()
        self.compile_and_train_model(X_train, y_train, X_val, y_val)
        self.save_model_and_metadata()


if __name__ == "__main__":
    data_file_path = sys.argv[1] if len(sys.argv) == 2 else "training-3.0.csv"
    classifier = TransportModeClassifier(data_file_path, is_training=True)  # Here pass True if it is training phase
    classifier.run()

    # Evaluate model after training
    test_file_path = "testing-3.0.csv"
    classifier.evaluate_model(test_file_path)

    classifier.export_to_lite()
