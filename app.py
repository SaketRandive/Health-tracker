# app.py
import streamlit as st
import requests

st.title("AI-Powered Health Tracker")

st.header("Fitness Tracking")
steps = st.number_input("Enter Steps Walked", min_value=0)
if st.button("Predict Calories Burned"):
    response = requests.post("http://127.0.0.1:5000/predict_calories", json={"steps": steps})
    st.write("Calories Burned:", response.json()['calories_burned'])

st.header("Diet Recommendation")
goal = st.selectbox("Choose Goal", ["weight_loss", "muscle_gain", "balanced"])
if st.button("Get Diet Plan"):
    response = requests.get(f"http://127.0.0.1:5000/diet?goal={goal}")
    st.write("Diet Plan:", response.json()['diet_plan'])

st.header("Sleep Analysis")
hours = st.number_input("Enter Sleep Hours", min_value=0)
if st.button("Check Sleep Quality"):
    response = requests.post("http://127.0.0.1:5000/sleep", json={"hours": hours})
    st.write("Sleep Quality:", response.json()['sleep_quality'])

st.header("Disease Risk Prediction")
age = st.number_input("Enter Age", min_value=0)
bmi = st.number_input("Enter BMI", min_value=0.0)
if st.button("Check Disease Risk"):
    response = requests.post("http://127.0.0.1:5000/disease", json={"age": age, "bmi": bmi})
    st.write("Risk Level:", response.json()['disease_risk'])
