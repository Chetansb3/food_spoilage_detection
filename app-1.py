import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
import os
import json
import requests

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
MODEL_URL = "https://huggingface.co/Latishhp/food_spoilage_detection/resolve/main/food_spoilage_model.keras"
CLASS_URL = "https://huggingface.co/Latishhp/food_spoilage_detection/resolve/main/class_names.json"

MODEL_FILE = "food_spoilage_model.keras"
CLASS_FILE = "class_names.json"

# ---------------------------------------------------------
# DOWNLOAD MODEL & CLASS NAMES
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        with st.spinner("🔽 Downloading model..."):
            r = requests.get(MODEL_URL)
            r.raise_for_status()
            with open(MODEL_FILE, "wb") as f:
                f.write(r.content)
    return tf.keras.models.load_model(MODEL_FILE)

@st.cache_resource
def load_class_names():
    if not os.path.exists(CLASS_FILE):
        try:
            r = requests.get(CLASS_URL)
            r.raise_for_status()
            class_names = r.json()  # Your JSON is a list of class labels
            with open(CLASS_FILE, "w", encoding="utf-8") as f:
                json.dump(class_names, f)
            return class_names
        except Exception as e:
            st.error(f"⚠️ Failed to download class names: {e}")
            return ["fresh", "spoiled"]
    
    with open(CLASS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------------------------------------------------
# PERSON DETECTION
# ---------------------------------------------------------
@st.cache_resource
def load_person_detector():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

def person_detected(img):
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    body_cascade = load_person_detector()
    bodies = body_cascade.detectMultiScale(gray, 1.1, 4)
    return len(bodies) > 0

# ---------------------------------------------------------
# IMAGE PREPROCESSING AND PREDICTION
# ---------------------------------------------------------
def preprocess_image(img, target_size=(224, 224)):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def predict(model, img, class_names):
    img_array = preprocess_image(img)
    preds = model.predict(img_array)
    idx = np.argmax(preds[0])
    confidence = float(preds[0][idx])
    return class_names[idx], confidence

# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.set_page_config(page_title="🍎 Food Spoilage Detection", layout="centered")
st.title("🍎 Real-Time Food Spoilage Detection")
st.write("This app skips frames if a **person** is detected and only classifies **food images**.")

model = load_model()
class_names = load_class_names()

camera_image = st.camera_input("📸 Capture food image")

if camera_image is not None:
    img = Image.open(camera_image)
    st.image(img, caption="Captured Frame", use_container_width=True)

    if person_detected(img):
        st.error("🚫 Person detected in frame. Please ensure only food is visible.")
    else:
        label, confidence = predict(model, img, class_names)

        st.markdown(
            f"""
            <div style="padding:15px; border-radius:10px; background-color:#f0f2f6; border:2px solid #4CAF50; text-align:center;">
                <h3 style="margin:0;">🍽️ Prediction: <b style="color:#2e7d32;">{label.replace('_', ' ').capitalize()}</b></h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div style="padding:15px; border-radius:10px; background-color:#fff3cd; border:2px solid #ff9800; text-align:center; margin-top:10px;">
                <h3 style="margin:0;">📊 Confidence: <b>{confidence*100:.2f}%</b></h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if "spoiled" in label.lower():
            st.warning("⚠️ The food appears to be spoiled. Please avoid consuming it.")
        else:
            st.success("✅ The food appears fresh and safe to eat.")

st.markdown("---")
st.caption("🤖 Powered by TensorFlow & OpenCV | Model hosted on Hugging Face")


