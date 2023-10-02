import sys
import os
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
        self.is_training = is_training
        
        if self.is_training:  # If it is training phase, load and preprocess data
            data = self.load_data()
            self.train_features, self.train_labels = self.preprocess_and_encode_data(data)

    def run(self):
        if self.is_training:  # If it is training phase, proceed with training, saving and exporting model
            X_train, X_val, y_train, y_val = self.split_data()
            self.compile_and_train_model(X_train, y_train, X_val, y_val)
            self.save_model_and_metadata()
            self.export_to_lite()
        else:  # If it is not training phase, instantiate the model from saved file
            self.load_model_and_metadata()

    def load_model_and_metadata(self):
        # Load the saved model and metadata when it is not training phase
        model_path = '../model/sentiance-replic/trained_sentiance-replic_model'
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path, compile=False)
            self.model.compile(
                loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy', self.f1_metric]
            )

            # Load label encoder
            le_path = '../model/sentiance-replic/label_encoder.npy'
            if os.path.exists(le_path):
                classes = np.load(le_path)
                self.label_encoder.classes_ = classes

            # Load training statistics
            stats_path = '../model/sentiance-replic/statistics.pkl'
            if os.path.exists(stats_path):
                with open(stats_path, 'rb') as f:
                    self.train_statistics = pickle.load(f)

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
        input_shape_acc = (128, 3)  # Example, modify as per your actual input shape for accelerometer data
        input_shape_gps = (3,)  # Example, modify as per your actual input shape for GPS data
        input_shape_state = (10,)  # Example, modify as per your actual input shape for state data
        
        self.model = Models.create_sentiance_replic_model(
            input_shape_acc=input_shape_acc,
            input_shape_gps=input_shape_gps,
            input_shape_state=input_shape_state,
            num_classes=self.num_classes
        )

        self.model.summary()

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy', self.f1_metric]
        )

        early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
        checkpoint = ModelCheckpoint(
            '../model/sentiance-replic/trained_sentiance-replic_model_{epoch:02d}_{val_loss:.4f}',
            save_best_only=True
        )

        lr_scheduler = LearningRateScheduler(self.lr_schedule)

        y_train = to_categorical(y_train, num_classes=6)
        y_val = to_categorical(y_val, num_classes=6)

        # Initialize State Tensor
        state_train = np.full((len(X_train), 500), -1.0)
        state_val = np.full((len(X_val), 500), -1.0)

        reshaped_and_scaled_X_train = self.reshape_and_scale_data(X_train)
        reshaped_and_scaled_X_val = self.reshape_and_scale_data(X_val)

        history = self.model.fit(
            x=reshaped_and_scaled_X_train,  # [X_acc, X_gps, X_state]
            y=y_train, 
            epochs=20, 
            batch_size=1024, 
            verbose=1,
            callbacks=[early_stopping, checkpoint, lr_scheduler],
            validation_data=(reshaped_and_scaled_X_val, y_val)
        )

        return history
    
    def evaluate_model(self, test_file):
        # Load and preprocess the test data similar to training data
        test_data = Preprocessing.load_and_extract_features(test_file)
        X_test, y_test = self.preprocess_and_encode_data(test_data)

        # Here adjust the data shaping and scaling as per the new model inputs
        reshaped_X_test = self.reshape_and_scale_data(test_data)
        
        predictions = self.model.predict(reshaped_X_test)

        # Transform labels back to original encoding
        y_true = self.label_encoder.inverse_transform(y_test)

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
        acc_indices = [3, 4, 5]
        gps_indices = [1, 2]
            
        # Use MinMaxScaler().fit_transform to scale the data and then store scaled data.
        X_acc = MinMaxScaler().fit_transform(X[:, acc_indices])
        X_gps = MinMaxScaler().fit_transform(X[:, gps_indices])
            
        # Adjusting X_acc shape to match the model’s expected input shape
        X_acc = X_acc.reshape((X_acc.shape[0], 128, 3))  # Adjust the second dimension as per your sequence length.
        
        print(X_acc.shape)  # To understand the initial shape.


        # Initialize state vector
        X_state = np.full((X.shape[0], 500), -1.0)
            
        return [X_acc, X_gps, X_state]  # Return List of Reshaped, Scaled Data and State



    def save_model_and_metadata(self):
        self.model.save('../model/sentiance-replic/trained_sentiance-replic_model')
        with open('../model/sentiance-replic/statistics.pkl', 'wb') as f:
            pickle.dump(self.train_statistics, f)

        np.save('../model/sentiance-replic/label_encoder.npy', self.label_encoder.classes_)

        metadata = {
            'statistics': self.train_statistics,
            'labels': [mode.value for mode in TransportationMode]
        }

        with open('../model/sentiance-replic/statistics.json', 'w') as f:
            json.dump(self.train_statistics, f)

        coreml_model = ct.convert(self.model)
        coreml_model.user_defined_metadata['preprocessing_metadata'] = json.dumps(metadata)
        coreml_model.user_defined_metadata['prediction_type'] = 'probability'
        coreml_model.save('../model/sentiance-replic/multi-sentiance-replic.mlmodel')

        
    def export_to_lite(self):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()

        with open('../model/sentiance-replic/multi-sentiance-replic-lite.tflite', 'wb') as f:
            f.write(tflite_model)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        data_file_path = sys.argv[1]
        is_training = True
    elif len(sys.argv) == 3 and sys.argv[2] == 'test':
        data_file_path = sys.argv[1]
        is_training = False
    else:
        data_file_path = "training-3.0.csv"
        is_training = True

    classifier = TransportModeClassifier(data_file_path, is_training=is_training)
    classifier.run()

    if is_training:
        # Evaluate model after training
        test_file_path = "testing-3.0.csv"
        classifier.evaluate_model(test_file_path)
