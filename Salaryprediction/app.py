import streamlit as st
import pickle

st.title("💼 Salary Prediction App")

model=pickle.load(open("Salaryprediction.pkl", "rb")) 
st.subheader("🧑‍💻 Enter Details")

years = st.number_input("Enter the Years of Experience", min_value=0.0, step=0.1)

# Create DataFrame for prediction
input_data = pd.DataFrame([[years_experience]], columns=["YearsExperience"])

# Predict salary
if st.button("Predict Salary"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Predicted Salary: ₹ {prediction:,.2f}")

