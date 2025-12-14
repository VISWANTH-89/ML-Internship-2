import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# App title
st.title("💼 Salary Prediction App")

# Load dataset from repository
@st.cache_data
def load_data():
    return pd.read_csv("Salary Data.csv")

data = load_data()

st.subheader("📊 Dataset Preview")
st.dataframe(data.head())

# Features and target
X = data.drop("Salary", axis=1)
y = data["Salary"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

st.success("✅ Model trained successfully!")

st.subheader("🧑‍💻 Enter Employee Details")

# User input fields
user_input = {}
for col in X.columns:
    user_input[col] = st.number_input(
        f"Enter {col}", 
        value=float(X[col].mean())
    )

input_df = pd.DataFrame([user_input])

# Predict salary
if st.button("Predict Salary"):
    prediction = model.predict(input_df)[0]

    st.subheader("📢 Predicted Salary")
    st.success(f"₹ {prediction:,.2f}")
