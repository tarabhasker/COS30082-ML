from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import glob

app = Flask(__name__)

# Load the trained model
model = load_model('path_to_your_model.h5')

# Path to your dataset directory
dataset_path = 'path_to_your_dataset/'

# Function to find similar images
def find_similar_images(predicted_label):
    similar_images = []
    for img_path in glob.glob(f"{dataset_path}/{predicted_label}/*.jpg"):  # Modify for your dataset structure
        similar_images.append(img_path)
    return similar_images[:5]  # Return the top 5 similar images

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Load and preprocess the image
    img = load_img(file, target_size=(224, 224))  # Adjust target size based on your model
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index]

    # Map the predicted index to the crop-disease label
    labels = ['Tomato - Bacterial Spot', 'Tomato - Early Blight', ...]  # Replace with your labels
    predicted_label = labels[predicted_class_index]

    # Find similar images
    similar_images = find_similar_images(predicted_label)

    # Prepare response
    response = {
        'class': predicted_label,
        'confidence': f"{confidence * 100:.2f}%",
        'matchedImages': [os.path.relpath(img, dataset_path) for img in similar_images]
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
