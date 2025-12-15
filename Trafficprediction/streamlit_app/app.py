import os
import pickle
import streamlit as st

st.write("Current Directory:", os.getcwd())
st.write("Files in directory:", os.listdir())

import os, pickle

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "traffic.pkl")

with open("trafficprediction.pkl", "rb") as f:
    model = pickle.load(f)


st.subheader("🔢 Enter Total Vehicle Count")
try:
    df = pd.read_csv("/content/Traffic.csv")
except:
    st.error("Traffic.csv not found ❌")
    st.stop()

le = LabelEncoder()
le.fit(df["Traffic Situation"])
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
