import sys
import os
import tensorflow as tf
import pandas as pd
from tensorflow.lite.python.interpreter import Interpreter
import numpy as np
from enum import Enum

class TransportationMode(Enum):
    IDLE = 0
    WALKING = 1
    RUNNING = 2
    BIKING = 3
    TRAIN = 4
    METRO = 5
    CAR = 6
    BUS = 7
    MOTORCYCLE = 8

class TestModeClassifier:
    # Initialize this dictionary where you are calling this predict function multiple times
    mode_counts = {mode: 0 for mode in TransportationMode}
    total_predictions = 0

    def __init__(self, data_file):
        self.data_file = data_file
        self.interpreter = None
        self.mode_counts = {mode.name: 0 for mode in TransportationMode}


    def run(self):
        self.load_model()
        data = self.load_data()

        state_data = np.full((500,), -1, dtype=np.float32)  # Initialize with -1's as no state is provided initially.

        for chunk in self.data_generator(data):
            # Process each chunk of data here
            # You can call the modified 'predict' function with each chunk and pass the state_data
            state_data = self.predict(chunk, state_data)

        most_common_mode = self.get_most_common_mode()
        print(f"Most Common Mode: {most_common_mode}")

    def data_generator(self, data, chunk_size=3600):
        num_rows = data.shape[0]
        for i in range(0, num_rows, chunk_size):
            yield data.iloc[i:i + chunk_size]

    def predict(self, data, state_data):
        # Extract pad_acc_data
        pad_acc_data = data[['timestamp', 'a_x', 'a_y', 'a_z']].to_numpy(dtype=np.float32)
        pad_acc_data = self.pad_acc_data_to_size(pad_acc_data, 3600)

        # Extract other data according to the model's requirement
        gps_fix_data = data[['timestamp', 'latitude', 'longitude', 'speed', 'accuracy']].iloc[-1].to_numpy(dtype=np.float32)

        # Get input details and output details
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        # Set tensors
        self.interpreter.set_tensor(input_details[0]['index'], pad_acc_data)
        self.interpreter.set_tensor(input_details[1]['index'], gps_fix_data)
        self.interpreter.set_tensor(input_details[2]['index'], state_data)

        # Run the model
        self.interpreter.invoke()

        # Get outputs
        trip_level_prob = self.interpreter.get_tensor(output_details[1]['index'])
        new_state = self.interpreter.get_tensor(output_details[3]['index'])

         # After getting trip_level_prob
        if trip_level_prob is not None:
            mode_probabilities = trip_level_prob.flatten()
            mode_names = [mode.name for mode in TransportationMode]
            print("Mode Percentages:")
            for mode_name, prob in zip(mode_names, mode_probabilities):
                print(f"{mode_name} = {prob * 100:.5f}%")
                
            # Find the mode with the highest probability
            predicted_mode_index = np.argmax(mode_probabilities)
            predicted_mode_name = TransportationMode(predicted_mode_index).name

            # Update the mode counts
            self.mode_counts[predicted_mode_name] += 1

        return new_state
    
    def get_most_common_mode(self):
        most_common_mode = max(self.mode_counts, key=self.mode_counts.get)
        return most_common_mode

    def load_model(self):
        model_path = '../model/sentiance-original/TransportClassifier.default.1_0.tflite'
        if os.path.exists(model_path):
            self.interpreter = Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
        else:
            raise FileNotFoundError(f"Model file {model_path} does not exist.")

    def load_data(self):
        # Assuming the CSV file has no header, if it has, change header to 'infer'
        columns = ['timestamp', 'speed', 'latitude', 'longitude', 'accuracy', 'a_x', 'a_y', 'a_z']
        data = pd.read_csv(self.data_file, header=None, names=columns)
        return data
    
    def pad_acc_data_to_size(self, data, target_size=3600):
        current_size = len(data)
        if current_size < target_size:
            padding_element = data[0:1]  # Take the first element for padding
            padding_size = target_size - current_size  # Calculate how many elements are needed to pad
            padding = np.repeat(padding_element, padding_size, axis=0)  # Create a padding array
            data = np.concatenate((data, padding), axis=0)  # Add padding to the original array
        return data

if __name__ == "__main__":

    if len(sys.argv) < 2:  # Check if the filename is provided as command-line argument
        print("Usage: python script_name.py <test_file_path>")
        sys.exit(1)  # Exit the script if filename is not provided

    test_file_path = sys.argv[1]  # Get the first command-line argument

    classifier = TestModeClassifier(data_file=test_file_path)
    classifier.run()
