import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model
with open("lung_cancer.pkl", "rb") as f:
    model = pickle.load(f)

def predict_lung_cancer_status(input_data):
    input_df = pd.DataFrame(input_data)
    input_df['GENDER'].replace({'M': 1, 'F': 0}, inplace=True)
    prediction = model.predict(input_df)
    return int(prediction[0])

def main():
    st.title("Lung Cancer Prediction")

    st.sidebar.header("User Input")
    age = st.sidebar.number_input("Age", min_value=0, max_value=100, step=1, key="age")
    gender = st.sidebar.radio("Gender", ["M", "F"], key="gender")
    smoking = st.sidebar.selectbox("Smoking Frequency in a Day", ["0", "1", "2"], key="smoking")
    yellow_fingers = st.sidebar.selectbox("How many Yellow Fingers", ["0", "1", "2"], key="yellow_fingers")
    anxiety = st.sidebar.selectbox("Feeling of Anxiety in a Day", ["0", "1", "2"], key="anxiety")
    chest_pain = st.sidebar.selectbox("Chest Pain Frequency in a Day", ["0", "1", "2"], key="chest_pain")
    swallowing_difficulty = st.sidebar.selectbox("Swallowing Difficulty in a Day", ["0", "1", "2"], key="swallowing_difficulty")
    shortness_of_breath = st.sidebar.selectbox("Shortness of Breath in a Day", ["0", "1", "2"], key="shortness_of_breath")
    peer_pressure = st.sidebar.selectbox("Feeling of Peer Pressure Frequency in a Day", ["0", "1", "2"], key="peer_pressure")
    chronic_disease = st.sidebar.selectbox("How many Chronic Diseases", ["0", "1", "2"], key="chronic_disease")
    coughing = st.sidebar.selectbox("Coughing in a Day", ["0", "1", "2"], key="coughing")
    alcohol_consumption = st.sidebar.selectbox("Alcohol Consumption in a Day", ["0", "1", "2"], key="alcohol_consumption")
    wheezing = st.sidebar.selectbox("Wheezing Frequency in a Day", ["0", "1", "2"], key="wheezing")
    allergy = st.sidebar.selectbox("Number of Allergies", ["0", "1", "2"], key="allergy")
    fatigue = st.sidebar.selectbox("Feeling of Fatigue in a Day", ["0", "1", "2"], key="fatigue")

    input_data = {
        'GENDER': gender,
        'AGE': age,
        'SMOKING': smoking,
        'YELLOW_FINGERS': yellow_fingers,
        'ANXIETY': anxiety,
        'CHEST_PAIN': chest_pain,
        'SWALLOWING_DIFFICULTY': swallowing_difficulty,
        'SHORTNESS_OF_BREATH': shortness_of_breath,
        'PEER_PRESSURE': peer_pressure,
        'CHRONIC_DISEASE': chronic_disease,
        'COUGHING': coughing,
        'ALCOHOL_CONSUMPTION': alcohol_consumption,
        'WHEEZING': wheezing,
        'ALLERGY': allergy,
        'FATIGUE': fatigue,
    }

    probability = model.predict_proba([list(input_data.values())])
    result = predict_lung_cancer_status(input_data)

    if result == 1:
        st.write("Prediction: You have Lung Cancer! Please consult a doctor. Probability:", probability[0][1])
    else:
        st.write("Prediction: You don't have lung cancer. Probability:", probability[0][0])

if __name__ == "__main__":
    main()
