# Mode of Transportation Prediction Project

This project utilizes machine learning models to predict modes of transportation and provides a web service to make predictions on new data.

## Requirements

- Python 3
- pip
- Flask
- TensorFlow
- scikit-learn
- coremltools

## Installation

1. Install Python 3 and pip:

    ```bash
    sudo apt update
    sudo apt install python3
    sudo apt install python3-pip
    ```

2. (Optional) Set up a Python virtual environment:

    ```bash
    python3 -m venv venv      
    source venv/bin/activate  
    ```

3. Install Flask and other dependencies:

    ```bash
    pip install Flask tensorflow scikit-learn coremltools
    ```

## Usage

1. Clone the project repository or copy the necessary files to the Debian machine.

2. Navigate to the project directory:

    ```bash
    cd /path/to/your/project
    ```

3. Run the Flask app:

    ```bash
    python app.py
    ```

4. Access the Flask app by opening a web browser and navigating to `http://<your_debian_ip>:8000`.

5. (Optional) To keep the Flask app running even after closing the terminal:

    ```bash
    nohup python app.py > flask_log.txt 2>&1 &
    ```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Credits

- TensorFlow: https://www.tensorflow.org/
- scikit-learn: https://scikit-learn.org/
- coremltools: https://github.com/apple/coremltools

