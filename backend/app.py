from flask import Flask, request, jsonify, render_template
from flask_cors import CORS 
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import clip
import torch
from torchvision import transforms
import uuid  # 用于生成唯一的文件名
import pandas as pd
from flask import send_from_directory
from torch.nn.functional import normalize
from torchvision import models
import torch.nn.functional as F
from torch import nn

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model 1
clip_model, preprocess = clip.load("ViT-B/32", device=device)
crop_model = tf.keras.models.load_model("model/VisualLanguageClipCrop.h5")
disease_model = tf.keras.models.load_model("model/VisualLanguageClipDisease.h5")

# Load Model 2
class ImprovedDualBranchModel(torch.nn.Module):
    def __init__(self, num_crops, num_diseases, embedding_dim=300, pretrained_embeddings=None):
        super(ImprovedDualBranchModel, self).__init__()
        self.backbone = models.resnet18(weights="IMAGENET1K_V1")
        self.backbone.fc = torch.nn.Identity()
        self.feature_projector = torch.nn.Linear(512, embedding_dim)
        self.crop_classifier = torch.nn.Linear(embedding_dim, num_crops)
        self.disease_classifier = torch.nn.Linear(embedding_dim, num_diseases)

    def forward(self, x):
        features = self.backbone(x)
        projected_features = self.feature_projector(features)
        normalized_features = normalize(projected_features, p=2, dim=1)
        crop_logits = self.crop_classifier(normalized_features)
        disease_logits = self.disease_classifier(normalized_features)
        return crop_logits, disease_logits, normalized_features


# Load pretrained Model 2
model_2 = torch.load("model/model_test_without_domain.h5", map_location=torch.device("cpu"))
model_2.eval()

# Model 2's transformations
transform_2 = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

# Load Model 3
class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(DomainDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Lab and field
        )

    def forward(self, x):
        return self.model(x)

class ImprovedDualBranchModelWithDomainDiscriminator(nn.Module):
    def __init__(self, num_crops, num_diseases, embedding_dim, pretrained_embeddings):
        super(ImprovedDualBranchModelWithDomainDiscriminator, self).__init__()
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        self.backbone.fc = nn.Identity()
        self.feature_projector = nn.Linear(512, embedding_dim)
        self.crop_classifier = nn.Linear(embedding_dim, num_crops)
        self.disease_classifier = nn.Linear(embedding_dim, num_diseases)
        self.domain_discriminator = DomainDiscriminator(input_dim=embedding_dim)

    def forward(self, x, alpha=1.0):
        features = self.backbone(x)
        projected_features = self.feature_projector(features)
        normalized_features = F.normalize(projected_features, p=2, dim=1)
        crop_logits = self.crop_classifier(normalized_features)
        disease_logits = self.disease_classifier(normalized_features)
        domain_logits = self.domain_discriminator(normalized_features)
        return crop_logits, disease_logits, domain_logits


model_3 = ImprovedDualBranchModelWithDomainDiscriminator(
    num_crops=14,
    num_diseases=21,
    embedding_dim=384,
    pretrained_embeddings={"crop": torch.zeros(14, 384), "disease": torch.zeros(21, 384)}
)
# Adjust to ignore unexpected keys
state_dict_3 = torch.load("model/best_model_with_domain.pth", map_location=torch.device("cpu"))
model_3.load_state_dict(state_dict_3, strict=False)
model_3.eval()

transform_3 = transform_2

# Labels
crop_labels = {
    0: "Apple Leaf",
    1: "Blueberry Leaf",
    2: "Cherry Leaf",
    3: "Corn Leaf",
    4: "Grape Leaf",
    5: "Orange Leaf",
    6: "Peach Leaf",
    7: "Pepper Leaf",
    8: "Potato Leaf",
    9: "Raspberry Leaf",
    10: "Soybean Leaf",
    11: "Squash Leaf",
    12: "Strawberry Leaf",
    13: "Tomato Leaf",
}

disease_labels = {
    0: "Healthy Leaf",
    1: "Apple Leaf with Scab",
    2: "Black Rot",
    3: "Cedar apple rust",
    4: "Powdery Mildew",
    5: "Cercospora leaf spot (Gray leaf spot)",
    6: "Common rust",
    7: "Northern Leaf Blight",
    8: "Esca (Black Measles)",
    9: "Leaf blight (Isariopsis Leaf Spot)",
    10: "Huanglongbing (Citrus Greening)",
    11: "Bacterial spot",
    12: "Early blight",
    13: "Late blight",
    14: "Leaf scorch",
    15: "Leaf Mold",
    16: "Septoria leaf spot",
    17: "Spider mites (Two-spotted spider mite)",
    18: "Target Spot",
    19: "Tomato Leaf with Yellow Leaf Curl Virus",
    20: "Tomato Leaf with Mosaic Virus",
}

domain_labels = {0: "Lab", 1: "Field"}

# Data augmentation and preprocessing
augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Resize((224, 224)),  # Resize to match CLIP's input size
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                         std=(0.26862954, 0.26130258, 0.27577711))  # CLIP normalization
])

# Helper function for Model 2
def predict_with_model_2(image_path):
    # Preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = transform_2(image).unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        # Get the logits from the model
        crop_logits, disease_logits, _ = model_2(image)

        # Predict the class with the highest score
        crop_pred = torch.argmax(crop_logits, dim=1).item()
        disease_pred = torch.argmax(disease_logits, dim=1).item()

        # Compute confidence scores using softmax
        crop_probs = F.softmax(crop_logits, dim=1)
        disease_probs = F.softmax(disease_logits, dim=1)

        # Extract the confidence of the predicted class
        crop_confidence = crop_probs[0, crop_pred].item()
        disease_confidence = disease_probs[0, disease_pred].item()

    # Map predictions to descriptions
    crop_name = crop_labels[crop_pred]
    disease_name = disease_labels[disease_pred]

    return crop_name, crop_confidence, disease_name, disease_confidence

# Image preprocessing function
def preprocess_image(image_path, augment=False):
    image = Image.open(image_path).convert("RGB")
    if augment:
        return augmentation_transforms(image).unsqueeze(0).to(device)
    return preprocess(image).unsqueeze(0).to(device)


# Single image prediction
def predict_image(image_path, clip_model, classification_model, label_map):
    # Preprocess the image
    image = preprocess_image(image_path)
    with torch.no_grad():
        # Extract image features
        image_features = clip_model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    # Convert to TensorFlow format
    image_features_tf = tf.convert_to_tensor(image_features.cpu().numpy())
    image_features_tf = tf.squeeze(image_features_tf, axis=0)

    # Use the classification model to predict
    predictions = classification_model.predict(tf.expand_dims(image_features_tf, axis=0))
    predicted_class = tf.argmax(predictions[0]).numpy()
    confidence = float(predictions[0][predicted_class])
    

    # Return predicted class
    return predicted_class, label_map[predicted_class], confidence


# Helper functions
def predict_with_model_3(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform_3(image).unsqueeze(0).to(device)

    with torch.no_grad():
        crop_logits, disease_logits, domain_logits = model_3(image)
        crop_pred = torch.argmax(crop_logits, dim=1).item()
        disease_pred = torch.argmax(disease_logits, dim=1).item()
        domain_pred = torch.argmax(domain_logits, dim=1).item()

        crop_confidence = F.softmax(crop_logits, dim=1)[0, crop_pred].item()
        disease_confidence = F.softmax(disease_logits, dim=1)[0, disease_pred].item()

    return crop_labels[crop_pred], crop_confidence, disease_labels[disease_pred], disease_confidence, domain_labels[domain_pred]

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files or "model" not in request.form:
        return jsonify({"error": "No file or model selected"}), 400

    file = request.files["file"]
    selected_model = request.form["model"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        # Save uploaded file
        unique_filename = f"{uuid.uuid4().hex}.jpg"
        image_path = os.path.join("uploads", unique_filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(image_path)

        # Perform predictions
        if selected_model == "1":
            # Use Model 1
            crop_class, crop_name, crop_confidence = predict_image(image_path, clip_model, crop_model, crop_labels)
            disease_class, disease_name, disease_confidence = predict_image(image_path, clip_model, disease_model, disease_labels)

            crop_class = int(crop_class)  # Convert to Python int
            disease_class = int(disease_class)  # Convert to Python int
        elif selected_model == "2":
            # Use Model 2
            crop_name, crop_confidence, disease_name, disease_confidence = predict_with_model_2(image_path)
            crop_class = int(list(crop_labels.values()).index(crop_name))  # Convert to Python int
            disease_class = int(list(disease_labels.values()).index(disease_name))  # Convert to Python int
        elif selected_model == "3":
            crop_name, crop_confidence, disease_name, disease_confidence, domain_name = predict_with_model_3(image_path)
            crop_class = int(list(crop_labels.values()).index(crop_name))  # Convert to Python int
            disease_class = int(list(disease_labels.values()).index(disease_name))  # Convert to Python int
        else:
            return jsonify({"error": "Invalid model selected"}), 400

        # Clean up uploaded file
        if os.path.exists(image_path):
            os.remove(image_path)

        # Prepare the response
        response = {
            "crop": {"class": crop_class, "name": crop_name, "confidence": crop_confidence},
            "disease": {"class": disease_class, "name": disease_name, "confidence": disease_confidence},
        }

        if selected_model == "3":
            response["domain"] = domain_name

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/fetch-matching-images", methods=["POST"])
def fetch_matching_images():
    try:
        dataset_path = "dataset/PV train.csv"
        data = pd.read_csv(dataset_path, header=None, names=["ImagePath", "CropLabel", "DiseaseLabel"])

        request_data = request.get_json()
        crop_label = int(request_data.get("crop_label"))
        disease_label = int(request_data.get("disease_label"))

        filtered_data = data[
            (data["CropLabel"] == crop_label) & (data["DiseaseLabel"] == disease_label)
        ]
        image_paths = filtered_data["ImagePath"].tolist()

        # Log response for debugging
        print("Filtered image paths:", image_paths)
        print("Image paths sent to frontend:", image_paths)

        return jsonify({"images": image_paths}), 200
    except Exception as e:
        print("Error in /fetch-matching-images:", str(e))  # Log the error
        return jsonify({"error": str(e)}), 500

# Serve static images from the dataset folder
@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('dataset/plantvillage', filename)

@app.route("/favicon.ico")
def favicon():
    return jsonify({"message": "No favicon provided"})


if __name__ == "__main__":
    app.run(debug=True)
