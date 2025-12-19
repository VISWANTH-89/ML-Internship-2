import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------------
# Load model safely
# -------------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "price_pred.pkl")

model = joblib.load(MODEL_PATH)

# -------------------------------
# User Input
# -------------------------------
st.title("Price Prediction App")

feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")

input_data = pd.DataFrame([[feature1, feature2]],
                          columns=model.feature_names_in_)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Price"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Price: {prediction[0]:.2f}")

    st.write("Expected number of features:", model.n_features_in_)
