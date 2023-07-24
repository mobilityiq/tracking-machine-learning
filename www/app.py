import os
import tensorflow as tf
from datetime import datetime
from flask import Flask, request

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the path to the second script
second_script_path = '../predict/predict-2.0.py'

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file is uploaded
    if 'file' not in request.files:
        return 'No file uploaded.'

    file = request.files['file']

    # Check if the file has an allowed extension
    if not file.filename.lower().endswith(('.csv')):
        return 'Authentication failed. Your connection has been recorded and will be reported'

    # Save the uploaded file to the uploads folder
    current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"predict_uploaded_file_{current_datetime}.csv"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Call the second script for prediction
    command = f"python {second_script_path} {file_path}"
    prediction = os.popen(command).read()
    print(prediction)
    
    # Extract the mode probabilities from the prediction output
    mode_probabilities_index = prediction.find("Mode Probabilities:")
    if mode_probabilities_index == -1:
        return "No mode probabilities found."

    start_index = mode_probabilities_index + len("Mode Probabilities:") + 1
    end_index = prediction.find("\n", start_index)
    mode_probabilities_line = prediction[start_index:end_index]

    # Get the first word after "Mode Probabilities:" and remove the colon (:) if present
    mode = mode_probabilities_line.split()[0].rstrip(":")

    # Delete the uploaded file after making the prediction
    os.remove(file_path)

    return mode

@app.route('/upload', methods=['POST'])
def upload():
    # Check if a file is uploaded
    if 'file' not in request.files:
        return 'No file uploaded.'

    file = request.files['file']

    # Check if the file has an allowed extension
    if not file.filename.lower().endswith(('.csv')):
        return 'Authentication failed. Your connection has been recorded and will be reported'

    # Save the uploaded file to the uploads folder with the generated filename
    current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"train_uploaded_file_{current_datetime}.csv"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    return 'File uploaded successfully.'

if __name__ == "__main__":
    app.run(host='51.68.196.15', port=8000, debug=True)
