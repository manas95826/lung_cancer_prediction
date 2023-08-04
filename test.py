import numpy as np
import pandas as pd
import streamlit as st
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Load the trained model
with open("lung_cancer.pkl", "rb") as f:
    model = pickle.load(f)
# st.set_theme("dark")
# Define default values for mean and variance
default_mean = 0.0
default_var = 1.0

# Create a StandardScaler instance with default values
# scaler = StandardScaler()
# scaler.mean_ = default_mean
# scaler.var_ = default_var

def predict_lung_cancer_status(input_data):
    input_df = pd.DataFrame([input_data])
    input_df.GENDER.replace({'M': '1', 'F': '0'}, inplace=True)
    # input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_df)
    return int(prediction[0])

def main():
    st.title("Lung Cancer Prediction")

    st.sidebar.header("User Input")
    age = st.sidebar.number_input("Age", min_value=1, max_value=100, step=1, key="age")
    gender = st.sidebar.selectbox("Gender", ["M", "F"], key="gender")
    smoking = st.sidebar.selectbox("Smoking Frequency in a Day", ["0","1", "2"], key="smoking")
    yellow_fingers = st.sidebar.selectbox("How many Yellow Fingers", ["0","1", "2"], key="yellow_fingers")
    anxiety = st.sidebar.selectbox("Feeling of Anxiety in a Day", ["0","1", "2"],key="anxiety")
    chest_pain = st.sidebar.selectbox("Chest Pain Frequency in a Day", ["0","1", "2"], key="chest_pain")
    swallowing_difficulty = st.sidebar.selectbox("Swallowing Difficulty in a Day", ["0","1", "2"], key="swallowing_difficulty")
    shortness_of_breath = st.sidebar.selectbox("Shortness of Breath in a Day", ["0","1", "2"], key="shortness_of_breath")
    peer_pressure = st.sidebar.selectbox("Feeling of Peer Pressure Frequency in a Day", ["0","1", "2"],key="peer_pressure")
    chronic_disease = st.sidebar.selectbox("How many Chronic Diseases", ["0","1", "2"], key="chronic_disease")
    coughing = st.sidebar.selectbox("Coughing in a Day", ["0","1", "2"],key="coughing")
    alcohol_consumption = st.sidebar.selectbox("Alcohol Consumption in a Day", ["0","1", "2"], key="alcohol_consumption")
    wheezing = st.sidebar.selectbox("Wheezing Frequency in a Day", ["0","1", "2"], key="wheezing")
    allergy = st.sidebar.selectbox("Number of Allergies", ["0","1", "2"],key="allergy")
    fatigue = st.sidebar.selectbox("Feeling of Fatigue in a Day", ["0","1", "2"], key="fatigue")

    input_data = {
    'GENDER': gender,
    'AGE': age,
    'SMOKING': smoking,
    'YELLOW_FINGERS': yellow_fingers,
    'ANXIETY': anxiety,
    'PEER_PRESSURE': peer_pressure,
    'CHRONIC DISEASE': chronic_disease,
    'WHEEZING': wheezing,
    'ALCOHOL CONSUMING': alcohol_consumption,
    'COUGHING': coughing,
    'SHORTNESS OF BREATH': shortness_of_breath,
    'SWALLOWING DIFFICULTY': swallowing_difficulty,
    'CHEST PAIN': chest_pain
}

    result = predict_lung_cancer_status(input_data)
    probability = model.predict_proba([input_data])
    if result == 1:
        st.write("Prediction: You've Lung Cancer! Bye bye tata, good bye gaya!")
    else:
        st.write("Prediction: You don't have lung cancer chill, have good sex!")

if __name__ == "__main__":
    main()
