from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io
import base64
import re

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('digit_recognition_model_cnn.h5')

# Function to preprocess the image before making a prediction
def preprocess_image(image):
    # Resize to 28x28 pixels and convert to grayscale
    image = image.resize((28, 28)).convert('L')
    image = np.array(image)
    
    # Invert colors (white background, black digit)
    image = 255 - image
    
    # Normalize pixel values
    image = image / 255.0
    
    # Reshape to match model input shape
    image = np.reshape(image, (1, 28, 28, 1))
    
    return image

# Route for the home page
@app.route('/')
def index():
    return render_template('D:\project1\templates')

# Route for processing the image and predicting the digit
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Get the base64-encoded image and decode it
    image_data = re.search(r'base64,(.*)', data['image']).group(1)
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Predict the digit
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)

    # Return the predicted digit as JSON
    return jsonify({'digit': int(predicted_digit)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
