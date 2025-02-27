import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import gdown
import os

# Add title and creator info with enhanced styling
st.markdown("""
    <h1 style='text-align: center; font-size: 48px; color: #1E88E5;'>Brain MRI Classification Demo</h1>
    <p style='text-align: center; color: #1E88E5; font-style: italic; font-size: 24px; margin-top: 20px; font-weight: bold;'>Created by Mojtba Allam</p>
    <hr style='height: 3px; background-color: #1E88E5; border: none; margin: 30px 0;'>
    """, unsafe_allow_html=True)

# Create columns for language selection
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    lang = st.radio("Language / اللغة", ["English", "العربية"], horizontal=True)

# Model explanation text
explanation_en = """
This deep learning model uses a Convolutional Neural Network (CNN) to analyze brain MRI scans and classify them as either healthy or diseased. 
The model has been trained on a dataset of brain MRI images and achieves high accuracy in detecting abnormalities.

Key Features:
- Real-time classification
- High accuracy predictions
- Confidence score for each prediction
- Support for common image formats (JPG, PNG)

Please note: This tool is for educational purposes only and should not be used for medical diagnosis.
"""

explanation_ar = """
<div dir="rtl" style="text-align: right;">
يستخدم هذا النموذج تقنية التعلم العميق وشبكة عصبية التفافية (CNN) لتحليل صور الرنين المغناطيسي للدماغ وتصنيفها إما كدماغ سليم أو مريض.
تم تدريب النموذج على مجموعة من صور الرنين المغناطيسي للدماغ ويحقق دقة عالية في اكتشاف الحالات غير الطبيعية.

الميزات الرئيسية:
- تصنيف فوري
- تنبؤات عالية الدقة
- درجة الثقة لكل تنبؤ
- دعم لصيغ الصور الشائعة (JPG, PNG)

ملاحظة: هذه الأداة لأغراض تعليمية فقط ولا ينبغي استخدامها للتشخيص الطبي.
</div>
"""

# Display explanation based on language selection
if lang == "English":
    st.write(explanation_en)
else:
    st.markdown(explanation_ar, unsafe_allow_html=True)

# -------------------------------
# Load the trained model
# -------------------------------
@st.cache_resource
def load_model():
    """
    Loads the trained CNN model from disk or downloads it from Google Drive if not present.
    Uses Streamlit caching to prevent reloading on each run.
    
    Returns:
        tensorflow.keras.Model: The loaded brain MRI classification model
    """
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
    Preprocesses an uploaded MRI image for model prediction.
    
    Args:
        image_file: StreamletUploadedFile containing the MRI image
        
    Returns:
        numpy.ndarray: Preprocessed image array ready for model prediction
        None: If image processing fails
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
    """
    Makes a prediction on the preprocessed MRI image using the loaded model.
    
    Args:
        model: tensorflow.keras.Model for classification
        processed_img: numpy.ndarray of the preprocessed image
        
    Returns:
        tuple: (prediction result string, confidence percentage)
    """
    preds = model.predict(processed_img)
    class_idx = np.argmax(preds, axis=1)[0]
    confidence = preds[0][class_idx] * 100
    result = "Healthy Brain" if class_idx == 0 else "Diseased Brain"
    return result, confidence

# -------------------------------
# Streamlit UI
# -------------------------------
def main():
    """
    Main application function that handles the Streamlit UI and orchestrates
    the image upload, processing, and prediction workflow. Supports both
    English and Arabic languages.
    """
    st.title("Brain MRI Classification Demo")
    st.write("Upload a brain MRI image to see the classification result.")

    # File uploader widget with language support
    upload_text = "Choose an MRI image..." if lang == "English" else "اختر صورة الرنين المغناطيسي..."
    uploaded_file = st.file_uploader(upload_text, type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        processed_img = preprocess_image(uploaded_file)
        if processed_img is not None:
            # Reset file pointer for display
            uploaded_file.seek(0)
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), 
                    caption="Uploaded MRI Image" if lang == "English" else "صورة الرنين المغناطيسي المحملة", 
                    use_container_width=True)
            
            # Classify button with language support
            button_text = "Classify" if lang == "English" else "تصنيف"
            if st.button(button_text):
                result, confidence = predict(model, processed_img)
                # Use different colors based on prediction
                if "Diseased" in result:
                    result_ar = "دماغ مريض" if lang == "العربية" else result
                    st.error(f"{'Prediction' if lang == 'English' else 'النتيجة'}: **{result_ar}** ({'Confidence' if lang == 'English' else 'نسبة الثقة'}: {confidence:.2f}%)")
                else:
                    result_ar = "دماغ سليم" if lang == "العربية" else result
                    st.success(f"{'Prediction' if lang == 'English' else 'النتيجة'}: **{result_ar}** ({'Confidence' if lang == 'English' else 'نسبة الثقة'}: {confidence:.2f}%)")
        else:
            error_text = "Error processing the image. Please try another file." if lang == "English" else "خطأ في معالجة الصورة. الرجاء المحاولة بملف آخر."
            st.error(error_text)

if __name__ == "__main__":
    main()
