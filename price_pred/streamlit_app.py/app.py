import streamlit as st
import pandas as pd
import pickle
import os

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Mobile Price Prediction", layout="centered")

st.title("📱 Mobile Phone Price Prediction")
st.write("Predict mobile phone price using Machine Learning")

# -------------------------------
# Load Model
# -------------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "price_pred.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# -------------------------------
# User Inputs
# -------------------------------
st.header("Enter Phone Specifications")

ram = st.number_input("RAM (GB)", min_value=1, max_value=24, value=8)
rom = st.number_input("ROM (GB)", min_value=8, max_value=1024, value=128)
battery = st.number_input("Battery Power (mAh)", min_value=1000, max_value=7000, value=4500)
primary_cam = st.number_input("Primary Camera (MP)", min_value=5, max_value=200, value=64)
selfie_cam = st.number_input("Selfie Camera (MP)", min_value=2, max_value=100, value=16)
mobile_size = st.number_input("Mobile Size (inches)", min_value=4.0, max_value=8.0, value=6.5)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Price 💰"):

    input_data = pd.DataFrame(
        [[ram, rom, battery, primary_cam, selfie_cam, mobile_size]],
        columns=[
            "RAM",
            "ROM",
            "Battery_Power",
            "Primary_Cam",
            "Selfi_Cam",
            "Mobile_Size"
        ]
    )

    prediction = model.predict(input_data)

    st.success(f"📊 Estimated Mobile Price: ₹ {prediction[0]:,.2f}")
