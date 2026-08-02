import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    return joblib.load('tourism_model.joblib')

model = load_model()

st.set_page_config(page_title="Wellness Tourism Prediction", page_icon="🏖️")
st.title('Wellness Tourism Package Prediction 🏖️')
st.write("Predict whether a customer will purchase the new Wellness Tourism Package.")

st.sidebar.header('Customer Attributes')
age = st.sidebar.slider('Age', 18, 70, 35)
city_tier = st.sidebar.selectbox('City Tier', [1, 2, 3])
occupation = st.sidebar.selectbox('Occupation', ['Salaried', 'Small Business', 'Large Business', 'Free Lancer'])
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
marital_status = st.sidebar.selectbox('Marital Status', ['Single', 'Married', 'Divorced', 'Unmarried'])
income = st.sidebar.number_input('Monthly Income', 10000, 200000, 50000)
contact_type = st.sidebar.selectbox('Contact Type', ['Self Inquiry', 'Company Invited'])
pitch_duration = st.sidebar.slider('Duration of Pitch (min)', 5, 40, 15)
satisfaction = st.sidebar.slider('Pitch Satisfaction Score', 1, 5, 3)

input_dict = {
    'Age': age, 'CityTier': city_tier, 'Occupation': occupation, 'Gender': gender,
    'MaritalStatus': marital_status, 'MonthlyIncome': income, 'TypeofContact': contact_type,
    'DurationOfPitch': pitch_duration, 'PitchSatisfactionScore': satisfaction,
    'NumberOfPersonVisiting': 3, 'PreferredPropertyStar': 3, 'NumberOfTrips': 2,
    'Passport': 0, 'OwnCar': 1, 'NumberOfChildrenVisiting': 1,
    'Designation': 'Executive', 'ProductPitched': 'Basic', 'NumberOfFollowups': 3
}

input_df = pd.DataFrame([input_dict])

if st.button('Predict Purchase Likelihood'):
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    if pred == 1:
        st.success(f"🎯 Prediction: **LIKELY TO PURCHASE** (Probability: {proba:.2%})")
    else:
        st.error(f"🛑 Prediction: **UNLIKELY TO PURCHASE** (Probability: {proba:.2%})")
