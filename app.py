import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load your trained model
model = joblib.load('churn_xgb_model.pkl')

st.set_page_config(page_title="Telco Churn Prediction", page_icon="📊")
st.title("📊 Telco Customer Churn Prediction")
st.write("Enter customer details below to check if the customer is likely to churn.")

# Input fields (adjust based on your dataset features)
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=1)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=50.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=500.0)

# Encode categorical variables
gender_num = 1 if gender == "Male" else 0
partner_num = 1 if partner == "Yes" else 0
dependents_num = 1 if dependents == "Yes" else 0

# Arrange features in the same order as your training dataset
features = np.array([[gender_num, senior_citizen, partner_num, dependents_num,
                      tenure, monthly_charges, total_charges]])

# Predict button
if st.button("Predict Churn"):
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is likely to stay.")
