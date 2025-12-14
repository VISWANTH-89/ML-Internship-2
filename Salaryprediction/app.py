import streamlit as st
import pandas as pd
import pickle

model=pickle.load(open("Salaryprediction.pkl","rb"))
st.title("Salary Prediction App")

years=st.number_input("Enter Years of Experience",min_value=0.0,step=0.1)

if st.button("Predict Salary"):
  prediction=model.predict([[years]])
  st.success(f"Predicted Salary: ₹
{prediction[0]:,.2f}")
