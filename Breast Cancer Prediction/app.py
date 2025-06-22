import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and top 10 features
model = joblib.load("xgb_top10_model.pkl")
top_features = joblib.load("top10_features.pkl")

st.title("Breast Cancer Diagnosis Prediction")
st.write("Input values for the top 10 features:")

# Input fields for the 10 most important features
user_input = {}
for feature in top_features:
    user_input[feature] = st.number_input(f"{feature}", format="%.4f")

# Convert input to DataFrame
input_df = pd.DataFrame([user_input])

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.success(f"Prediction: {'Malignant' if prediction == 1 else 'Benign'}")
