import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import gdown
import os

# Add title and creator info with enhanced styling
st.markdown("""
    <h1 style='text-align: center; font-size: 48px; color: #1E88E5;'>Brain MRI Classification Demo</h1>
    <p style='text-align: center; color: #424242; font-style: italic; font-size: 24px; margin-top: 20px;'>Created by Eng. Mojtba Allam</p>
    <hr style='height: 3px; background-color: #1E88E5; border: none; margin: 30px 0;'>
    """, unsafe_allow_html=True)

# -------------------------------
# Load the trained model
# -------------------------------
@st.cache_resource
def load_model():
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    model_path = "models/model.h5"
    
    # Download model if it doesn't exist
    if not os.path.exists(model_path):
        # Convert the Google Drive link to direct download format
        file_id = "1Zx6qmSnamGZTxR8vceLvV9HBw8a3mpZH"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)
    
    # Load the model
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# Define the image size used during training
IMG_WIDTH, IMG_HEIGHT = 128, 128

# -------------------------------
# Preprocess uploaded image
# -------------------------------
def preprocess_image(image_file):
    """
    Reads an uploaded image file, converts it to grayscale,
    resizes it, normalizes it, and prepares it for prediction.
    """
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

# -------------------------------
# Prediction function
# -------------------------------
def predict(model, processed_img):
    preds = model.predict(processed_img)
    class_idx = np.argmax(preds, axis=1)[0]
    confidence = preds[0][class_idx] * 100
    result = "Healthy Brain" if class_idx == 0 else "Diseased Brain"
    return result, confidence

# -------------------------------
# Streamlit UI
# -------------------------------
def main():
    st.title("Brain MRI Classification Demo")
    st.write("Upload a brain MRI image to see the classification result.")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        processed_img = preprocess_image(uploaded_file)
        if processed_img is not None:
            # Reset file pointer for display
            uploaded_file.seek(0)
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Uploaded MRI Image", use_container_width=True)
            
            if st.button("Classify"):
                result, confidence = predict(model, processed_img)
                # Use different colors based on prediction
                if "Diseased" in result:
                    st.error(f"Prediction: **{result}** (Confidence: {confidence:.2f}%)")
                else:
                    st.success(f"Prediction: **{result}** (Confidence: {confidence:.2f}%)")
        else:
            st.error("Error processing the image. Please try another file.")

if __name__ == "__main__":
    main()
