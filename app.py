import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import io
from werkzeug.utils import secure_filename

from flask_cors import CORS

app = Flask(__name__)
CORS(app)
model = None
class_names = None

# Load model and class indices
def load_model():
    global model, class_names
    model = tf.keras.models.load_model("nsfw.h5")
    with open("class_indices.pkl", "rb") as f:
        class_indices = pickle.load(f)
    class_names = [None] * len(class_indices)
    for label, idx in class_indices.items():
        class_names[idx] = label

load_model()

# Middleware-like function to check NSFW content
def is_nsfw_image(file):
    try:
        image = Image.open(file.stream).convert("RGB")
        image = image.resize((224, 224))
        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

        prediction = model.predict(img_array)
        predicted_idx = np.argmax(prediction[0])
        predicted_label = class_names[predicted_idx]
        confidence = float(prediction[0][predicted_idx])

        return predicted_label.lower() in ["porn", "nsfw", "sexy", "hentai"], predicted_label, confidence * 100
    except Exception as e:
        print("Error filtering image:", e)
        return True, "Error", 0.0  # Assume unsafe on error

@app.route("/upload-profile", methods=["POST"])
def upload_profile():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    is_illicit, label, confidence = is_nsfw_image(file)

    if is_illicit:
        return jsonify({
            "status": "rejected",
            "reason": f"Image classified as '{label}' ({confidence:.2f}%)"
        }), 400

    # Save the image (optional: save to static/profile_images/...)
    # file.save(os.path.join("static/profile_images", secure_filename(file.filename)))

    filename = secure_filename(file.filename)
    save_path = os.path.join("static/profile", filename)
    file.save(save_path)

    return jsonify({
        "status": "accepted",
        "label": label,
        "confidence": f"{confidence:.2f}%"
    }), 200

if __name__ == "__main__":
    app.run(debug=True)
