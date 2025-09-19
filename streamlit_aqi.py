import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import datetime

# -----------------------------
# Helper functions
# -----------------------------
def preprocess_image(image, target_size=(128, 128)):
    """Resize and normalize image for model prediction"""
    image = image.resize(target_size)
    image_array = np.array(image)
    if image_array.shape[-1] == 4:
        image_array = image_array[..., :3]  # Remove alpha channel if exists
    image_array = image_array / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Batch dimension
    return image_array

def get_aqi_info(aqi_value):
    """Return AQI class, category, color, and health message"""
    if aqi_value <= 50:
        return 'a_Good', 'Good', 'Green', 'Air quality is considered satisfactory.'
    elif aqi_value <= 100:
        return 'b_Moderate', 'Moderate', 'Yellow', 'Air quality is acceptable.'
    elif aqi_value <= 150:
        return 'c_Unhealthy for Sensitive', 'Unhealthy for Sensitive Groups', 'Orange', 'Sensitive groups may experience health effects.'
    elif aqi_value <= 200:
        return 'd_Unhealthy', 'Unhealthy', 'Red', 'Everyone may begin to experience health effects.'
    elif aqi_value <= 300:
        return 'e_Very Unhealthy', 'Very Unhealthy', 'Purple', 'Health alert: everyone may experience more serious effects.'
    else:
        return 'f_Hazardous', 'Hazardous', 'Maroon', 'Health warnings of emergency conditions.'

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Satellite AQI Predictor", layout="wide")
st.title("🌫️ Satellite Air Quality Index (AQI) Predictor")

# Upload image
uploaded_file = st.file_uploader("Upload a satellite image...", type=["jpg", "png", "jpeg"])
location = st.text_input("Location", "Biratnagar, Nepal")
true_aqi = st.number_input("True AQI (if known)", min_value=0, max_value=500, value=0)
date = st.date_input("Date", datetime.date.today())

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)  # width in pixels
    
    # Preprocess
    img_array = preprocess_image(image, target_size=(128, 128))  # Match model input size
    
    # Load model (replace path with your trained model)
    model = load_model(r"C:\Users\kotaa\Downloads\AQI\AQI\aqi_combine_two_model.h5")
    
    # Predict AQI
    predicted_aqi = float(model.predict(img_array)[0][0])
    
    # Get AQI info
    aqi_class, aqi_category, aqi_color, health_msg = get_aqi_info(predicted_aqi)
    
    # Display results
    st.markdown(f"**🔢 Predicted AQI:** {predicted_aqi:.2f}")
    st.markdown(f"**🏷️ AQI Class:** {aqi_class}")
    st.markdown(f"**📊 AQI Category:** {aqi_category}")
    st.markdown(f"**💬 Health Message:** {health_msg}")
    st.markdown(f"**🎨 AQI Color:** {aqi_color}")
    
    st.markdown("---")
    st.markdown(f"📍 **Location:** {location}, 📅 **Date:** {date}")
    if true_aqi > 0:
        true_class, _, _, _ = get_aqi_info(true_aqi)
        st.markdown(f"🎯 **True AQI:** {true_aqi}, Class: {true_class}")
