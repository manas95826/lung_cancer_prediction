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
    age = st.sidebar.number_input("Age", min_value=0, max_value=100, step=1, key="age")
    gender = st.sidebar.selectbox("Gender", ["M", "F"], key="gender")
    smoking = st.sidebar.number_input("Smoking", min_value=0, max_value=2, step=1, key="smoking")
    yellow_fingers = st.sidebar.number_input("yellow_fingers", min_value=0, max_value=2, step=1, key="yellow_fingers")
    anxiety = st.sidebar.number_input("anxiety",  min_value=0, max_value=2, step=1,key="anxiety")
    chest_pain = st.sidebar.number_input("chest_pain", min_value=0, max_value=2, step=1, key="chest_pain")
    swallowing_difficulty = st.sidebar.number_input("swallowing_difficulty", min_value=0, max_value=2, step=1, key="swallowing_difficulty")
    shortness_of_breath = st.sidebar.number_input("shortness_of_breath", min_value=0, max_value=2, step=1, key="shortness_of_breath")
    peer_pressure = st.sidebar.number_input("peer_pressure",  min_value=0, max_value=2, step=1,key="peer_pressure")
    chronic_disease = st.sidebar.number_input("chronic_disease", min_value=0, max_value=2, step=1, key="chronic_disease")
    coughing = st.sidebar.number_input("coughing",  min_value=0, max_value=2, step=1,key="coughing")
    alcohol_consumption = st.sidebar.number_input("alcohol_consumption", min_value=0, max_value=2, step=1, key="alcohol_consumption")
    wheezing = st.sidebar.number_input("wheezing", min_value=0, max_value=2, step=1, key="wheezing")
    allergy = st.sidebar.number_input("allergy",  min_value=0, max_value=2, step=1,key="allergy")
    fatigue = st.sidebar.number_input("fatigue", min_value=0, max_value=2, step=1, key="fatigue")

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

    if result == 1:
        st.write("Prediction: You've Lung Cancer! Bye bye tata, good bye gaya!")
    else:
        st.write("Prediction: You don't have lung cancer chill, have good sex!")

if __name__ == "__main__":
    main()
