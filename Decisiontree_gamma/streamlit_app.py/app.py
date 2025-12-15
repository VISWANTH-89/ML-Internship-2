import streamlit as st
import pickle
import math
import os

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Pass or Fail Prediction",
    layout="centered"
)

# ---------------- Title ----------------
st.title("🎓 Student Pass or Fail Prediction")
st.subheader("Pass or Fail Prediction using Decision Tree with Gamma")

st.markdown("---")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Decision_tree.pkl")
# ---------------- Load Model ----------------
MODEL_PATH = "Decision_tree.pkl"   # ✅ updated file name

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found")
    st.write("Looking for:", MODEL_PATH)
    st.write("Files in folder:", os.listdir(BASE_DIR))
    st.stop()

# ---------------- User Input ----------------
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

# ---------------- Prediction ----------------
if st.button("Predict"):

    # Automatic fail rules
    if study_hours < 2 or previous_score < 200:
        st.error("❌ Prediction: FAIL")
        st.warning("Reason: Study Hours < 2 or Previous Exam Score < 200")

    else:
        # Gamma feature transformation
        study_gamma = math.gamma(study_hours + 1)
        score_gamma = math.gamma(previous_score / 10 + 1)

        prediction = model.predict([[study_gamma, score_gamma]])

        if prediction[0] == 1:
            st.success("✅ Prediction: PASS")
        else:
            st.error("❌ Prediction: FAIL")

st.markdown("---")
st.caption("Decision Tree Model | Gamma Function Feature Transformation")
