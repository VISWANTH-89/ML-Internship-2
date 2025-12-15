import streamlit as st
import pickle
import os

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "Salaryprediction.pkl")
    with open(model_path, "rb") as file:
        return pickle.load(file)

model = load_model()

st.subheader("🧑‍💻 Salary Prediction")

years = st.number_input("Enter the Years of Experience", min_value=0.0, step=0.1)



# Predict salary
if st.button("Predict Salary"):
    prediction = model.predict([[years]])[0]
    st.success(f"💰 Predicted Salary: ₹ {prediction:,.2f}")
