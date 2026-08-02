# Streamlit app - loads the model committed by the GitHub Actions pipeline
import os
import streamlit as st
import pandas as pd
import joblib

# The pipeline commits the trained model right next to this file
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_model_v1.joblib")
model = joblib.load(model_path)

st.set_page_config(page_title="Tourism Package Prediction", layout="centered")
st.title("Wellness Tourism Package Prediction")
st.write("Predict whether a customer is likely to purchase the new Wellness Tourism Package.")
st.write("Enter the customer details below and click Predict.")

# --- Demographics ---
st.header("Customer Demographics")
Age = st.number_input("Age", min_value=18, max_value=100, value=35)
Gender = st.selectbox("Gender", ["Male", "Female"])
MaritalStatus = st.selectbox("Marital Status", ["Married", "Divorced", "Unmarried", "Single"])
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
MonthlyIncome = st.number_input("Monthly Income (gross)", min_value=1000, max_value=100000, value=20000)

# --- Trip preferences ---
st.header("Trip Preferences")
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=5, value=3)
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting (under 5)", min_value=0, max_value=3, value=0)
NumberOfTrips = st.number_input("Average Trips per Year", min_value=1, max_value=25, value=2)
PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])
Passport = st.selectbox("Has Passport?", ["Yes", "No"])
OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])

# --- Interaction details ---
st.header("Interaction Details")
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=5, max_value=60, value=15)
NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=1, max_value=6, value=3)
PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])

# Build a one-row DataFrame with the SAME columns used during training
input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": 1 if Passport == "Yes" else 0,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": 1 if OwnCar == "Yes" else 0,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome,
}])

if st.button("Predict"):
    proba = model.predict_proba(input_data)[0, 1]
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success(f"The customer is LIKELY to purchase the package. (Probability: {proba:.2%})")
    else:
        st.error(f"The customer is UNLIKELY to purchase the package. (Probability: {proba:.2%})")
