from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load_model('skin_cancer_model.h5')

# Define a function to preprocess the input image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        # Save the file to disk
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        # Preprocess the image
        img = preprocess_image(file_path)

        # Make a prediction
        prediction = model.predict(img)
        os.remove(file_path)  # Remove the file after prediction

        # Map prediction to class label
        class_labels = ['melanoma', 'basal cell carcinoma', 'squamous cell carcinoma', 'benign']
        predicted_class = class_labels[np.argmax(prediction)]

        return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
