import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Traffic Flow Prediction", layout="centered")

st.title("🚦 Traffic Flow Prediction")
st.write("Prediction using trained PKL model")

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    with open("traffic.pkl", "rb") as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
    st.success("Model loaded successfully ✅")
except:
    st.error("traffic.pkl file not found ❌")
    st.stop()

st.subheader("🔢 Enter Total Vehicle Count")

total = st.number_input(
    "Total Vehicles",
    min_value=0,
    max_value=10000,
    step=1
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Traffic Situation"):
    prediction = model.predict([[total]])
    prediction = int(round(prediction[0]))

    # Clip prediction to valid class range
    prediction = np.clip(prediction, 0, len(le.classes_) - 1)

    traffic_status = le.inverse_transform([prediction])[0]

    st.subheader("📌 Prediction Result")
    st.success(f"**Traffic Situation:** {traffic_status}")
