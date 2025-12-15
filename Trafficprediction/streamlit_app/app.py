import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Traffic Prediction", layout="centered")
st.title("🚦 Traffic Flow Prediction")
st.write(
    "This application uses **Linear Regression** to predict the "
    "**Traffic Flow Situation** based on the total number of vehicles."
)
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "traffic.pkl")
CSV_PATH = os.path.join(BASE_DIR, "Traffic.csv")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
try:
    df = pd.read_csv(CSV_PATH)
except Exception as e:
    st.error("❌ Traffic.csv not found")
    st.error(e)
    st.stop()

le = LabelEncoder()
le.fit(df["Traffic Situation"])

total = st.number_input(
    "Total Vehicles",
    min_value=0,
    max_value=10000,   # 🔹 added limit
    step=1
)
total = min(total, 10000)

if st.button("Predict"):
    pred = model.predict([[total]])
    pred = int(round(pred[0]))
    pred = np.clip(pred, 0, len(le.classes_) - 1)

    result = le.inverse_transform([pred])[0]
    st.success(f"🚥 Traffic Situation: **{result}**")
