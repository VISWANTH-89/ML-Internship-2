import streamlit as st
import pickle
import numpy as np
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# -------------------------------
# Load model (robust)
# -------------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "advertising_poly_model.pkl")
    with open(model_path, "rb") as file:
        obj1, obj2 = pickle.load(file)

    # Detect which is PolynomialFeatures
    if hasattr(obj1, "transform"):
        poly = obj1
        model = obj2
    else:
        model = obj1
        poly = obj2

    return poly, model

poly, model = load_model()

# -------------------------------
# App UI
# -------------------------------
st.title("Advertising Sales Prediction")
st.write("Enter advertising budget values")

tv = st.number_input("TV Advertising Budget", min_value=0.0)
radio = st.number_input("Radio Advertising Budget", min_value=0.0)
newspaper = st.number_input("Newspaper Advertising Budget", min_value=0.0)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Sales"):
    try:
        input_data = np.array([[tv, radio, newspaper]])

        # Apply polynomial transform
        input_poly = poly.transform(input_data)

        # Predict
        prediction = model.predict(input_poly)

        st.success(f"📈 Predicted Sales: {prediction[0]:.2f}")

    except Exception as e:
        st.error(f"Error: {e}")
