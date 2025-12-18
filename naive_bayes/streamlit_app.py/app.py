import streamlit as st
import pickle
import numpy as np
import os

# -------------------------------
# Load model
# -------------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "naive_bayes_model.pkl")
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model

# IMPORTANT: model is defined HERE
model = load_model()

# -------------------------------
# App UI
# -------------------------------
st.title("Diabetes Prediction App")
st.write("Enter patient details")

pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0, step=1)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):
    try:
        input_data = np.array([[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            dpf,
            age
        ]])

        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.error("🟥 Result: Diabetic")
        else:
            st.success("🟩 Result: Non-Diabetic")

    except Exception as e:
        st.error(f"Error: {e}")
