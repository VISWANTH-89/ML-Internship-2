import streamlit as st
import pickle
import numpy as np
import os

st.title("🩺 Diabetes Prediction App")
st.write(
    "This Diabetes Prediction App uses **Logistic Regression** to predict "
    "whether a patient is **Diabetes Positive or Negative**."
)

# Get correct path of pkl file
MODEL_PATH = os.path.join(os.path.dirname(__file__), "Logistic_Regression.pkl")

# Load model safely
try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("❌ Model file not found. Please upload Logistic_Regression.pkl")
    st.stop()
st.write("Enter patient details to predict diabetes")
# Patient inputs
age = st.number_input("Age", min_value=1, max_value=120, value=30)
mass = st.number_input(
    "BMI (Body Mass Index)",
    min_value=0.0,
    max_value=70.0,
    value=25.0
)

insu = st.number_input(
    "Insulin Level",
    min_value=0.0,
    max_value=900.0,
    value=80.0
)

plas = st.number_input(
    "Plasma Glucose Level",
    min_value=0.0,
    max_value=300.0,
    value=120.0
)


# Prediction
if st.button("Predict"):
    input_data = np.array([[age, mass, insu, plas]])
    prediction = model.predict(input_data)[0]

    if prediction == "tested_positive":
        st.error("⚠️ Prediction: Diabetes POSITIVE")
    else:
        st.success("✅ Prediction: Diabetes NEGATIVE")
