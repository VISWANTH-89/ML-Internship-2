import streamlit as st
import pickle
import numpy as np
import os

# -----------------------------
# Load model safely
# -----------------------------
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "svm_iris_model.pkl")

    with open(model_path, "rb") as file:
        obj = pickle.load(file)

    # Handle (model, scaler) or model-only
    if isinstance(obj, tuple):
        model = obj[0]
        scaler = obj[1] if len(obj) > 1 else None
    else:
        model = obj
        scaler = None

    return model, scaler


# ✅ DEFINE model and scaler HERE
model, scaler = load_model()

# -----------------------------
# UI
# -----------------------------
st.title("Iris Flower Prediction 🌸")

sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0)
sepal_width  = st.number_input("Sepal Width (cm)", min_value=0.0)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0)
petal_width  = st.number_input("Petal Width (cm)", min_value=0.0)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    try:
        input_data = np.array([[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]])

        # Apply scaler ONLY if it exists
        if scaler is not None:
            input_data = scaler.transform(input_data)

        prediction = model.predict(input_data)[0]

        class_names = {
            0: "Setosa",
            1: "Versicolor",
            2: "Virginica"
        }

        st.success(f"🌸 Predicted Flower: **{class_names.get(prediction, prediction)}**")

    except Exception as e:
        st.error(f"Prediction error: {e}")
