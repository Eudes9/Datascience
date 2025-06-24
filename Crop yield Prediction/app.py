import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# Define the input features (adjust based on your dataset)
feature_names = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']  # Replace with real names

st.title("🌾 Crop Yield Prediction App")

st.write("Enter the values below to predict the expected crop yield (Q/acre).")

# Input fields for features
input_data = {}
for feature in feature_names:
    input_data[feature] = st.number_input(f"{feature}", value=0.0)

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Predict button
if st.button("Predict Yield"):
    prediction = model.predict(input_df)[0]
    st.success(f"🌟 Predicted Yield: **{prediction:.2f} Q/acre**")
