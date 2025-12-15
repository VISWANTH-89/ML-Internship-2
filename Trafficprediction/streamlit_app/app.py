import os, pickle
st.set_page_config(page_title="Traffic Prediction", layout="centered")
st.title("🚦 Traffic Flow Prediction")
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "traffic.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


total = st.number_input(
    "Total Vehicles",
    min_value=0,
    step=1
)
if st.button("Predict Traffic Situation"):
    prediction = model.predict([[total]])
    prediction = int(round(prediction[0]))

    # Keep prediction in valid range
    prediction = np.clip(prediction, 0, len(le.classes_) - 1)

    result = le.inverse_transform([prediction])[0]

    st.success(f"🚥 Traffic Situation: **{result}**")
