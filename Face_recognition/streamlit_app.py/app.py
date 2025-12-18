import streamlit as st
import pickle
import numpy as np
from PIL import Image
import cv2

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Face Recognition App",
    layout="centered"
)

# -------------------------------
# Load model
# -------------------------------
@st.cache_resource
def load_model():
    with open("face_recognition_cnn.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# -------------------------------
# App title
# -------------------------------
st.title("😊 Face Recognition App")
st.write("Upload an image to predict the face using the trained CNN model")

# -------------------------------
# Image uploader
# -------------------------------
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------
# Image preprocessing
# -------------------------------
def preprocess_image(image):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (64, 64))
    image = image / 255.0
    image = image.reshape(1, 64, 64, 1)
    return image

# -------------------------------
# Prediction
# -
