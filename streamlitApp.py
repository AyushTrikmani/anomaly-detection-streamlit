import streamlit as st
import numpy as np
from PIL import Image
import base64
import io

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

# Simple rule-based prediction function for demonstration
def predict_damage(image):
    """
    Simple rule-based prediction - analyzes image characteristics
    Replace this with your actual Teachable Machine model integration
    """
    # Convert image to numpy array for analysis
    img_array = np.array(image)
    
    # Simple analysis based on image properties
    # Calculate brightness and contrast metrics
    gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
    brightness = np.mean(gray)
    contrast = np.std(gray)
    
    # Simple rule-based classification (you can adjust these thresholds)
    # This is just for demonstration - replace with your actual model
    
    # Check for very dark areas (possible damage)
    dark_pixels = np.sum(gray < 50) / gray.size
    
    # Check for high contrast variations (possible cracks)
    edge_variation = np.sum(np.abs(np.diff(gray, axis=0))) + np.sum(np.abs(np.diff(gray, axis=1)))
    edge_variation = edge_variation / gray.size
    
    # Simple scoring system
    damage_score = 0
    
    if dark_pixels > 0.1:  # More than 10% dark pixels
        damage_score += 0.3
    
    if edge_variation > 30:  # High edge variation
        damage_score += 0.4
    
    if brightness < 80:  # Very low brightness
        damage_score += 0.2
    
    if contrast > 60:  # High contrast (possible damage)
        damage_score += 0.1
    
    # Determine prediction
    if damage_score > 0.5:
        predicted_class = "Damage"
        confidence = min(0.6 + damage_score * 0.3, 0.95)
    else:
        predicted_class = "Normal"
        confidence = max(0.6 + (1 - damage_score) * 0.3, 0.6)
    
    # Create probability distribution
    if predicted_class == "Normal":
        normal_prob = confidence
        damage_prob = 1 - confidence
    else:
        damage_prob = confidence
        normal_prob = 1 - confidence
    
    return predicted_class, confidence, [normal_prob, damage_prob]

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
        
        class_names = ["Normal", "Damage"]
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

# Instructions for adding your model
st.markdown("---")
st.info("""
**To integrate your Teachable Machine model:**

Since Streamlit Cloud doesn't support TensorFlow, here are alternative approaches:

1. **Use Teachable Machine's REST API**: Deploy your model on Teachable Machine and call it via API
2. **Convert to ONNX**: Export your model to ONNX format and use onnxruntime
3. **Use HuggingFace Spaces**: Deploy there instead for full TensorFlow support

For now, this app uses simple image analysis rules for demonstration.
""")

st.markdown("*Powered by Computer Vision Technology*")
