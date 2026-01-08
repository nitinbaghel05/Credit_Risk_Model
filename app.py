import streamlit as st
import joblib
import pandas as pd

model = joblib.load("credit_risk_model.pkl")
encoders = joblib.load("encoders.pkl")

st.title("Credit Risk Prediction App")

age = st.number_input("Age", 18, 100)
sex = st.selectbox("Sex", ["male", "female"])
job = st.selectbox("Job", [0, 1, 2, 3])
housing = st.selectbox("Housing", ["own", "rent", "free"])
saving = st.selectbox("Saving Account", ["little", "moderate", "rich", "quite rich"])
checking = st.selectbox("Checking Account", ["little", "moderate", "rich"])
credit = st.number_input("Credit Amount")
duration = st.number_input("Duration (months)")

input_df = pd.DataFrame([{
    "Age": age,
    "Sex": encoders["Sex"].transform([sex])[0],
    "Job": job,
    "Housing": encoders["Housing"].transform([housing])[0],
    "Saving accounts": encoders["Saving accounts"].transform([saving])[0],
    "Checking account": encoders["Checking account"].transform([checking])[0],
    "Credit amount": credit,
    "Duration": duration
}])

if st.button("Predict Risk"):
    pred = model.predict(input_df)[0]
    st.success("Good Credit Risk" if pred == 1 else "Bad Credit Risk")
