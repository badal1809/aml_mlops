CATEGORIES = {"TypeofContact": ["Company Invited", "Self Enquiry"], "Occupation": ["Free Lancer", "Large Business", "Salaried", "Small Business"], "Gender": ["Female", "Male"], "ProductPitched": ["Basic", "Deluxe", "King", "Standard", "Super Deluxe"], "MaritalStatus": ["Divorced", "Married", "Single", "Unmarried"], "Designation": ["AVP", "Executive", "Manager", "Senior Manager", "VP"]}

import streamlit as st
import pandas as pd
import joblib

# Load the serialized pipeline (preprocessing + model) committed to the repository
@st.cache_resource
def load_model():
    return joblib.load("tourism_model.joblib")

model = load_model()

st.set_page_config(page_title="Wellness Tourism Package Prediction", page_icon="🏖️")
st.title("Wellness Tourism Package Prediction 🏖️")
st.markdown("Predict whether a customer will purchase the Wellness Tourism Package.")

with st.form("customer_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 70, 35)
        gender = st.selectbox("Gender", CATEGORIES["Gender"])
        marital_status = st.selectbox("Marital Status", CATEGORIES["MaritalStatus"])
        occupation = st.selectbox("Occupation", CATEGORIES["Occupation"])
        designation = st.selectbox("Designation", CATEGORIES["Designation"])
        monthly_income = st.number_input("Monthly Income", 1000, 200000, 23000)
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        own_car = st.selectbox("Own Car (0 = No, 1 = Yes)", [0, 1])
        passport = st.selectbox("Passport (0 = No, 1 = Yes)", [0, 1])

    with col2:
        type_of_contact = st.selectbox("Type of Contact", CATEGORIES["TypeofContact"])
        duration_of_pitch = st.slider("Duration of Pitch (minutes)", 1, 40, 15)
        pitch_satisfaction = st.slider("Pitch Satisfaction Score", 1, 5, 3)
        number_of_person_visiting = st.slider("Number of Persons Visiting", 1, 5, 3)
        number_of_children = st.slider("Number of Children Visiting", 0, 3, 1)
        number_of_followups = st.slider("Number of Follow-ups", 1, 6, 4)
        number_of_trips = st.slider("Number of Trips per Year", 0, 20, 3)
        preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
        product_pitched = st.selectbox("Product Pitched", CATEGORIES["ProductPitched"])

    submitted = st.form_submit_button("Predict Purchase Likelihood")

if submitted:
    input_dict = {
        "Age": age,
        "TypeofContact": type_of_contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": number_of_person_visiting,
        "NumberOfFollowups": number_of_followups,
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": preferred_property_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": number_of_trips,
        "Passport": passport,
        "PitchSatisfactionScore": pitch_satisfaction,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": number_of_children,
        "Designation": designation,
        "MonthlyIncome": monthly_income,
    }

    features = pd.DataFrame([input_dict])
    st.subheader("Input Data")
    st.dataframe(features)

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.success(f"🎯 The customer is **LIKELY** to purchase the package! (Probability: {probability:.2%})")
    else:
        st.error(f"🛑 The customer is **UNLIKELY** to purchase the package. (Probability: {probability:.2%})")
