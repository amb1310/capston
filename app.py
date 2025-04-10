import streamlit as st
import joblib
import numpy as np

model = joblib.load('calorie_predictor_model.pkl')

st.title("Smart Calorie Needs Predictor")

age = st.slider("Age", 18, 65)
gender = st.selectbox("Gender", ["Male", "Female"])
height = st.slider("Height (cm)", 150, 200)
weight = st.slider("Weight (kg)", 50, 120)
steps = st.slider("Daily Steps", 3000, 15000)
sleep = st.slider("Sleep Hours", 4.0, 10.0, step=0.1)
activity = st.selectbox("Activity Level", ["Low", "Moderate", "High"])

gender = 0 if gender == "Female" else 1
activity_dict = {"Low": 0, "Moderate": 1, "High": 2}
activity = activity_dict[activity]

input_data = np.array([[age, gender, height, weight, steps, sleep, activity]])
prediction = model.predict(input_data)

st.success(f"Estimated Daily Calorie Need: {int(prediction[0])} kcal")