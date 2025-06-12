import streamlit as st
import io
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Set up the page layout
st.set_page_config(page_title="Display Damage Inspector", page_icon="🔍")

st.title("Display Damage Inspector")

st.caption(
    "AI-Powered Display Damage Detection - Detect Normal vs Damaged Displays"
)

st.write(
    "Upload or capture an image of a display and watch how our AI model classifies it as Normal or Damaged."
)

# Sidebar information
with st.sidebar:
    st.subheader("About Display Damage Inspector")
    st.write(
        "This AI-powered application helps identify damage in display screens using computer vision. "
        "The model can detect various types of display damage including cracks, dead pixels, "
        "discoloration, and other visual defects."
    )
    
    st.write(
        "Simply upload an image or use your camera to capture a display image, "
        "and our AI will classify it as either Normal or Damaged with confidence scores."
    )

# Load the model
@st.cache_resource
def load_model():
    try:
        # Load the Keras model
        model = keras.models.load_model('keras_model.h5')
        
        # Load class names
        with open('labels.txt', 'r') as f:
            class_names = [line.strip().split(' ', 1)[1] for line in f.readlines()]
        
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Load model and class names
model, class_names = load_model()

if model is None:
    st.error("Failed to load the model. Please check if 'keras_model.h5' and 'labels.txt' are in the correct directory.")
    st.stop()

def preprocess_image(image):
    """
    Preprocess the image for the Teachable Machine model
    """
    # Resize image to 224x224 as required by Teachable Machine
    image = image.resize((224, 224))
    
    # Convert to numpy array
    image_array = np.asarray(image)
    
    # Normalize the image (Teachable Machine uses values between 0 and 1)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    # Reshape for the model (add batch dimension)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    return data

def predict_damage(image):
    """
    Predict if the display is normal or damaged
    """
    try:
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(processed_image)
        
        # Get the predicted class and confidence
        predicted_class_index = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class_index]
        predicted_class = class_names[predicted_class_index]
        
        return predicted_class, confidence, prediction[0]
    
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None

# Function to load uploaded image
def load_uploaded_image(file):
    img = Image.open(file)
    return img

# Image input selection
st.subheader("Select Image Input Method")
input_method = st.radio(
    "Choose input method:", 
    ["File Uploader", "Camera Input"], 
    label_visibility="collapsed"
)

# Initialize variables
uploaded_file_img = None
camera_file_img = None

# Handle file upload
if input_method == "File Uploader":
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        uploaded_file_img = load_uploaded_image(uploaded_file)
        st.image(uploaded_file_img, caption="Uploaded Image", width=300)
        st.success("Image uploaded successfully!")
    else:
        st.warning("Please upload an image file.")

# Handle camera input
elif input_method == "Camera Input":
    st.warning("Please allow access to your camera.")
    camera_image_file = st.camera_input("Capture Display Image")
    if camera_image_file is not None:
        camera_file_img = load_uploaded_image(camera_image_file)
        st.image(camera_file_img, caption="Camera Input Image", width=300)
        st.success("Image captured successfully!")
    else:
        st.warning("Please capture an image.")

# Prediction button
submit = st.button(label="Analyze Display for Damage", type="primary")

if submit:
    # Determine which image to use
    if input_method == "File Uploader" and uploaded_file_img is not None:
        img_to_analyze = uploaded_file_img
    elif input_method == "Camera Input" and camera_file_img is not None:
        img_to_analyze = camera_file_img
    else:
        st.error("Please provide an image before analyzing.")
        st.stop()
    
    # Show analysis results
    st.subheader("Analysis Results")
    
    with st.spinner("Analyzing display for damage..."):
        predicted_class, confidence, all_predictions = predict_damage(img_to_analyze)
        
        if predicted_class is not None:
            # Display main result
            if predicted_class.lower() == "normal":
                st.success(f"✅ **Result: {predicted_class}**")
                st.success("Great news! No damage detected in this display.")
            else:
                st.error(f"⚠️ **Result: {predicted_class}**")
                st.error("Damage detected! This display may require inspection or repair.")
            
            # Show confidence
            st.write(f"**Confidence:** {confidence:.2%}")
            
            # Show detailed predictions
            st.subheader("Detailed Analysis")
            col1, col2 = st.columns(2)
            
            for i, (class_name, prob) in enumerate(zip(class_names, all_predictions)):
                if i % 2 == 0:
                    with col1:
                        st.metric(class_name, f"{prob:.2%}")
                else:
                    with col2:
                        st.metric(class_name, f"{prob:.2%}")
            
            # Recommendations
            st.subheader("Recommendations")
            if predicted_class.lower() == "normal":
                st.info("🔍 The display appears to be in good condition. Continue regular monitoring.")
            else:
                st.warning("🔧 Consider having this display inspected by a technician. "
                          "Document the damage for warranty or insurance purposes if applicable.")

# Footer
st.markdown("---")
st.markdown("*Powered by AI and Computer Vision Technology*")