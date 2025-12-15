import streamlit as st
import numpy as np
import pickle
import math

st.set_page_config(page_title="Pass or Fail Prediction", layout="centered")

st.title("🎓 Student Pass or Fail Prediction")
st.subheader("Pass or Fail Prediction using Decision Tree with Gamma")

# ✅ Correct file name
MODEL_PATH = "Decision_tree.pkl"

# Load model safely
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found. Please upload decision_tree.pkl")
    st.stop()

study_hours = st.number_input("Enter Study Hours", min_value=0.0, step=0.1)
previous_score = st.number_input("Enter Previous Exam Score", min_value=0.0, step=1.0)

if st.button("Predict"):

    if study_hours < 2 or previous_score < 200:
        st.error("❌ Prediction: FAIL")
        st.info("Reason: Study Hours < 2 OR Previous Exam Score < 200")
    else:
        study_gamma = math.gamma(study_hours + 1)
        prev_score_gamma = math.gamma(previous_score / 10 + 1)

        prediction = model.predict([[study_gamma, prev_score_gamma]])

        if prediction[0] == 1:
            st.success("✅ Prediction: PASS")
        else:
            st.error("❌ Prediction: FAIL")
