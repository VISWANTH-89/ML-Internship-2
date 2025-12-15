import streamlit as st
import pickle
import numpy as np

# Load trained logistic regression model
with open("Logistic_Regression.pkl", "rb") as file:
    model = pickle.load(file)

st.title("🩺 Diabetes Prediction App")
st.write(
    "This Diabetes Prediction App uses **Logistic Regression** to predict "
    "whether a patient is **Diabetes Positive or Negative** based on patient details."
)
st.write("Enter patient details to predict diabetes")
# Patient input details
age = st.number_input("Age", min_value=1, max_value=120, value=30)
mass = st.number_input("BMI (Body Mass Index)", min_value=0.0, value=25.0)
insu = st.number_input("Insulin Level", min_value=0.0, value=80.0)
plas = st.number_input("Plasma Glucose Level", min_value=0.0, value=120.0)

# Prediction
if st.button("Predict"):
    input_data = np.array([[age, mass, insu, plas]])
    prediction = model.predict(input_data)[0]

    if prediction == "tested_positive":
        st.error("⚠️ Prediction: Diabetes POSITIVE")
    else:
        st.success("✅ Prediction: Diabetes NEGATIVE")
