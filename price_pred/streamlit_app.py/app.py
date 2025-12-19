import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Phone Price Prediction", layout="centered")

st.title("📱 Phone Price Prediction App")
st.write("Predict the price of a mobile phone using Machine Learning")

# -------------------------------
# Load Model
# -------------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "price_pred.pkl")

model = joblib.load(MODEL_PATH)

# -------------------------------
# Feature Names (MUST match training)
# -------------------------------
FEATURES = [
    "RAM",              # in GB
    "Storage",          # in GB
    "Battery",          # in mAh
    "Camera",           # in MP
    "Processor_Score"   # benchmark score
]

# -------------------------------
# User Input
# -------------------------------
st.header("Enter Phone Specifications")

ram = st.number_input("RAM (GB)", min_value=1, max_value=32, value=8)
storage = st.number_input("Storage (GB)", min_value=8, max_value=512, value=128)
battery = st.number_input("Battery (mAh)", min_value=1000, max_value=7000, value=4500)
camera = st.number_input("Camera (MP)", min_value=5, max_value=200, value=64)
processor = st.number_input("Processor Score", min_value=10, max_value=300, value=150)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Price 💰"):

    input_data = pd.DataFrame(
        [[ram, storage, battery, camera, processor]],
        columns=FEATURES
    )

    prediction = model.predict(input_data)

    st.success(f"📌 Estimated Phone Price: ₹ {prediction[0]:,.2f}")
