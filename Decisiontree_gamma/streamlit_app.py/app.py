import streamlit as st
import numpy as np
import pickle
import math

# Page config
st.set_page_config(page_title="Pass or Fail Prediction", layout="centered")

# Title & subheading
st.title("🎓 Student Pass or Fail Prediction")
st.subheader("Pass or Fail Prediction using Decision Tree with Gamma")

# Load trained model
with open("Decision Tree.pkl", "rb") as f:
    model = pickle.load(f)

st.markdown("---")

# User inputs
study_hours = st.number_input(
    "Enter Study Hours",
    min_value=0.0,
    step=0.1
)

previous_score = st.number_input(
    "Enter Previous Exam Score",
    min_value=0.0,
    step=1.0
)

# Predict button
if st.button("Predict"):
    
    # Automatic fail rules
    if study_hours < 2 or previous_score < 200:
        st.error("❌ Prediction: FAIL (Automatic Rule Applied)")
        st.info("Reason: Study Hours < 2 or Previous Exam Score < 200")
    else:
        # Apply gamma transformation (same as training)
        study_gamma = gamma(study_hours + 1)
        prev_score_gamma = gamma(previous_score / 10 + 1)

        # Prediction
        prediction = model.predict([[study_gamma, prev_score_gamma]])

        if prediction[0] == 1:
            st.success("✅ Prediction: PASS")
        else:
            st.error("❌ Prediction: FAIL")

st.markdown("---")
st.caption("Model: Decision Tree | Feature Transformation: Gamma Function")
