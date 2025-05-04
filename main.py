import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle

st.title("MASELANG IMAHE")

#Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("nsfw.h5")
    with open("class_indices.pkl", "rb") as f:
        class_indices = pickle.load(f)
    return model, class_indices

model, class_indices = load_model()

#Create a list of class names
class_names = [None] * len(class_indices)
for label, idx in class_indices.items():
    class_names[idx] = label

#Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

#Use EfficientNet preprocessing
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    # Predict
    prediction = model.predict(img_array)
    predicted_class_idx = np.argmax(prediction[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = prediction[0][predicted_class_idx] * 100

    st.write(f"### Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")