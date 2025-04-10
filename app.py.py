import streamlit as st
import numpy as np
import pandas as pd
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Function to generate synthetic IoT data
def generate_sample_data(n=500):
    data = {
        'Age': [random.randint(18, 65) for _ in range(n)],
        'Gender': [random.choice(['Male', 'Female']) for _ in range(n)],
        'Height_cm': [random.randint(150, 200) for _ in range(n)],
        'Weight_kg': [random.randint(50, 120) for _ in range(n)],
        'Steps': [random.randint(3000, 15000) for _ in range(n)],
        'Sleep_hours': [round(random.uniform(4, 10), 1) for _ in range(n)],
        'Activity_level': [random.choice(['Low', 'Moderate', 'High']) for _ in range(n)],
        'Calories': [random.randint(1500, 3500) for _ in range(n)]
    }
    return pd.DataFrame(data)

# Train the model from generated data
@st.cache_data
def train_model():
    df = generate_sample_data()
    df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
    df['Activity_level'] = LabelEncoder().fit_transform(df['Activity_level'])

    features = df.drop('Calories', axis=1)
    labels = df['Calories']

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(scaled_features, labels, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, scaler

# UI starts here
st.title("Smart Calorie Needs Predictor")

# Train model
model, scaler = train_model()

# User Inputs
age = st.slider("Age", 18, 65)
gender = st.selectbox("Gender", ["Male", "Female"])
height = st.slider("Height (cm)", 150, 200)
weight = st.slider("Weight (kg)", 50, 120)
steps = st.slider("Daily Steps", 3000, 15000)
sleep = st.slider("Sleep Hours", 4.0, 10.0, step=0.1)
activity = st.selectbox("Activity Level", ["Low", "Moderate", "High"])

# Encode Inputs
gender_val = 0 if gender == "Female" else 1
activity_dict = {"Low": 0, "Moderate": 1, "High": 2}
activity_val = activity_dict[activity]

# Prepare input data
input_data = np.array([[age, gender_val, height, weight, steps, sleep, activity_val]])
scaled_input = scaler.transform(input_data)

# Prediction
prediction = model.predict(scaled_input)

st.success(f"Estimated Daily Calorie Need: {int(prediction[0])} kcal")