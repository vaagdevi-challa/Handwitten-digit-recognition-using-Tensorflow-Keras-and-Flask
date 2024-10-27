
#  Handwritten Digit Recognition

It is a simple web application that recognizes handwritten digits (0-9) using a neural network built with Keras and TensorFlow. The model is deployed using Flask, allowing users to draw digits in a web browser, which are then classified by the model.



## Features

- Recognizes handwritten digits from 0 to 9 with high accuracy.
- Interactive web interface for drawing and testing digits.
- Real-time predictions rendered directly in the browser.

## Project Structure

```plaintext
Handwritten-digit-recognition-using-Tensorflow-Keras-and-Flask/  
├── requirements.txt  # To install all libraries needed
├── index.html   # Main HTML file
├── digit_recognition_model_cnn.h5     # Trained Keras model file
├── flaskInterface.py                   # Flask app to handle requests
├── Handwritten_digit_recognition.py           # Script to train the digit recognition model
└── README.md

```

## Getting Started

### Dataset
Download the dataset : https://www.kaggle.com/datasets/olafkrastovski/handwritten-digits-0-9
### Prerequisites

Ensure you have Python 3.x installed. Install required libraries by running:

```bash
pip install -r requirements.txt
```

### Requirements

- **Flask** - Web framework to serve the app.
- **TensorFlow/Keras** - Machine learning framework for model training and inference.
- **NumPy** - Numerical operations.
- **OpenCV** - For image processing (if used for pre-processing).

## Usage

1. **Train the Model (optional)**: If you’d like to train the model yourself, run `Handwritten_digit_recognition.py`. This will save `digit_recognition_model_cnn.h5` 
   ```bash
   python Handwritten_digit_recognition.py
   ```

2. **Run the Flask App**:
   ```bash
   python flaskInterface.py
   ```

3. **Open the App**:
   Open a web browser and go to `http://127.0.0.1:5000` to access the digit recognition interface.

## Model Training

The model is trained on the dataset of handwritten digits, which consists of 21600 training images. The neural network uses a simple CNN architecture to achieve high accuracy.
We got 95% accuracy while training the model

### Architecture involves
```plaintext
- Conv2D layer
- MaxPooling layer
- Flatten layer
- Dense layers
- Softmax output
```

## Technologies Used

- **Keras** & **TensorFlow** for neural network training.
- **Flask** to create a lightweight web server for the app.
- **HTML/CSS** for the front-end.
- **JavaScript** for handling canvas drawing and sending requests to the back-end.

## output
(![Screenshot 2024-10-24 055343](https://github.com/user-attachments/assets/200c22f2-320f-434f-9e24-503c52a3bafe)
)
