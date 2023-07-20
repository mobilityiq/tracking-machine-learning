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
        return 'Invalid file extension. Only .txt files are allowed.'

    # Save the uploaded file to the uploads folder
    current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"uploaded_file_{current_datetime}.csv"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Call the second script for prediction
    command = f"python {second_script_path} {file_path}"
    prediction = os.popen(command).read()

    return prediction

@app.route('/upload', methods=['POST'])
def upload():
    # Check if a file is uploaded
    if 'file' not in request.files:
        return 'No file uploaded.'

    file = request.files['file']

    # Check if the file has an allowed extension
    if not file.filename.lower().endswith(('.txt')):
        return 'Invalid file extension. Only .txt files are allowed.'

    # Save the uploaded file to the uploads folder with the generated filename
    current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"uploaded_file_{current_datetime}.txt"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    return 'File uploaded successfully.'

if __name__ == "__main__":
    app.run(host='192.168.18.200', port=8000, debug=True)
