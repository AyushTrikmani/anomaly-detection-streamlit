import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("model/model.h5")

# Class names
class_names = ['Normal', 'Anomaly']  # adjust based on your classes

st.title("Anomaly Detection App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image, caption="Uploaded Image")

    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 224, 224, 3)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"Prediction: {predicted_class}")
