import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(
    page_title="Loan Approval Prediction",
    layout="centered"
)

# ----------------------------------
# Title & Subtitle
# ----------------------------------
st.title("🏦 Loan Approval Prediction")
st.subheader("Model Used: Logistic Regression")

st.write("Enter applicant details to predict loan approval status.")

# ----------------------------------
# Load Model
# ----------------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "Loanapproval.pkl")

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# ----------------------------------
# User Inputs
# ----------------------------------
person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
person_income = st.number_input("Annual Income", min_value=0, value=50000)

person_home_ownership = st.selectbox(
    "Home Ownership",
    ["MORTGAGE", "OTHER", "OWN", "RENT"]
)

loan_amnt = st.number_input("Loan Amount", min_value=0, value=10000)

loan_intent = st.selectbox(
    "Loan Intent",
    ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT",
     "MEDICAL", "PERSONAL", "VENTURE"]
)

credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)

# ----------------------------------
# Prediction Button
# ----------------------------------
if st.button("Predict Loan Status"):
    
    # Create input DataFrame
    input_data = pd.DataFrame([{
        "person_age": person_age,
        "person_income": person_income,
        "person_home_ownership": person_home_ownership,
        "loan_amnt": loan_amnt,
        "loan_intent": loan_intent,
        "credit_score": credit_score
    }])

    # One-hot encoding
    input_encoded = pd.get_dummies(
        input_data,
        columns=["person_home_ownership", "loan_intent"],
        drop_first=True
    )

    # Expected columns from training
    expected_columns = model.feature_names_in_
    input_encoded = input_encoded.reindex(
        columns=expected_columns,
        fill_value=0
    )

    # Prediction
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)

    # ----------------------------------
    # Output
    # ----------------------------------
    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")

    st.write(f"Approval Probability: **{probability[0][1]*100:.2f}%**")
