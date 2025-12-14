import streamlit as st
import pandas as pd
import pickle

st.title("💼 Salary Prediction App")

# Load trained model
with open("Salaryprediction.pkl", "rb") as file:
    model = pickle.load(file)

st.subheader("🧑‍💻 Enter Details")

# Example input (change feature names if needed)
years_experience = st.number_input("Years of Experience", min_value=0.0, value=1.0)

# Create DataFrame for prediction
input_data = pd.DataFrame([[years_experience]], columns=["YearsExperience"])

# Predict salary
if st.button("Predict Salary"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Predicted Salary: ₹ {prediction:,.2f}")

