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

# 初始化 Flask 应用
app = Flask(__name__)
CORS(app)

# 检查设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 CLIP 模型
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# 加载模型
crop_model = tf.keras.models.load_model("model/VisualLanguageClipCrop.h5")
disease_model = tf.keras.models.load_model("model/VisualLanguageClipDisease.h5")

# 类别映射
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

# 数据增强和预处理
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


# 图像预处理
def preprocess_image(image_path, augment=False):
    image = Image.open(image_path).convert("RGB")
    if augment:
        return augmentation_transforms(image).unsqueeze(0).to(device)
    return preprocess(image).unsqueeze(0).to(device)


# 单张图像预测
def predict_image(image_path, clip_model, classification_model, label_map):
    # 预处理图像
    image = preprocess_image(image_path)
    with torch.no_grad():
        # 提取图像特征
        image_features = clip_model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    # 转换为 TensorFlow 格式
    image_features_tf = tf.convert_to_tensor(image_features.cpu().numpy())
    image_features_tf = tf.squeeze(image_features_tf, axis=0)

    # 使用分类模型预测
    predictions = classification_model.predict(tf.expand_dims(image_features_tf, axis=0))
    predicted_class = tf.argmax(predictions[0]).numpy()

    # 返回预测结果
    return predicted_class, label_map[predicted_class]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        # Save uploaded file
        unique_filename = f"{uuid.uuid4().hex}.jpg"
        image_path = os.path.join("uploads", unique_filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(image_path)

        # Perform predictions
        crop_class, crop_name = predict_image(image_path, clip_model, crop_model, crop_labels)
        disease_class, disease_name = predict_image(image_path, clip_model, disease_model, disease_labels)

        # Clean up uploaded file
        if os.path.exists(image_path):
            os.remove(image_path)

        # Return both class labels and names
        return jsonify({
            "crop": {
                "class": int(crop_class),  # Label
                "name": crop_name          # Name
            },
            "disease": {
                "class": int(disease_class),  # Label
                "name": disease_name          # Name
            }
        })
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
