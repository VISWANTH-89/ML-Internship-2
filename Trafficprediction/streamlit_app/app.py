import streamlit as st
import numpy as np
import os, pickle
st.set_page_config(page_title="Traffic Prediction", layout="centered")
st.title("🚦 Traffic Flow Prediction")
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "traffic.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


total = st.number_input(
    "Total Vehicles",
    min_value=0,
    step=1
)
if st.button("Predict"):
    pred = model.predict([[total]])
    pred = int(round(pred[0]))
    pred = np.clip(pred, 0, len(le.classes_) - 1)

    result = le.inverse_transform([pred])[0]
    st.success(f"🚥 Traffic Situation: **{result}**")
