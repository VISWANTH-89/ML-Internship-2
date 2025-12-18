import streamlit as st
import pickle
import os
import numpy as np
import pandas as pd

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Mobile Price Prediction", layout="centered")

# -------------------------------
# Load Model (SAFE)
# -------------------------------
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "price_pred.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


model = load_model()

# -------------------------------
# App UI
# -------------------------------
st.title("📱 Mobile Price Prediction App")
st.write("Enter mobile specifications to predict the price.")

# -------------------------------
# User Inputs (6 FEATURES)
# -------------------------------
ram = st.number_input("RAM (GB)", min_value=1, max_value=32, value=4)
storage = st.number_input("Storage (GB)", min_value=8, max_value=512, value=64)
battery = st.number_input("Battery Capacity (mAh)", min_value=1000, max_value=7000, value=4000)
camera = st.number_input("Camera (MP)", min_value=5, max_value=200, value=48)
screen_size = st.number_input("Screen Size (inches)", min_value=4.0, max_value=7.5, value=6.5)
processor_speed = st.number_input("Processor Speed (GHz)", min_value=1.0, max_value=4.0, value=2.2)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Price"):
    try:
        # Use feature names if available (BEST PRACTICE)
        if hasattr(model, "feature_names_in_"):
            input_df = pd.DataFrame(
                [[ram, storage, battery, camera, screen_size, processor_speed]],
                columns=model.feature_names_in_
            )
            prediction = model.predict(input_df)
        else:
            input_data = np.array(
                [[ram, storage, battery, camera, screen_size, processor_speed]]
            )
            prediction = model.predict(input_data)

        st.success(f"💰 Predicted Mobile Price: ₹ {int(prediction[0])}")

    except Exception as e:
        st.error("❌ Prediction failed. Feature mismatch or model issue.")
        st.write(e)

# -------------------------------
# Debug info (optional)
# -------------------------------
with st.expander("🔍 Model Info"):
    st.write("Expected number of features:", model.n_features_in_)
