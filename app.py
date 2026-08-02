import streamlit as st
import pandas as pd
import joblib

# Load the serialized pipeline
@st.cache_resource
def load_model():
    return joblib.load('tourism_model.joblib')

model = load_model()

st.set_page_config(page_title="Tourism Package Prediction", page_icon="🏖️", layout="centered")
st.title('Wellness Tourism Package Prediction 🏖️')
st.markdown("Predict whether a customer will purchase the newly introduced Wellness Tourism Package based on historical and interaction data.")

st.sidebar.header('🧑‍💼 Customer Demographics')
age = st.sidebar.slider('Age', 18, 70, 35)
city_tier = st.sidebar.selectbox('City Tier', [1, 2, 3])
occupation = st.sidebar.selectbox('Occupation', ['Salaried', 'Small Business', 'Large Business', 'Free Lancer'])
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
marital_status = st.sidebar.selectbox('Marital Status', ['Single', 'Married', 'Divorced', 'Unmarried'])
income = st.sidebar.number_input('Monthly Income (Gross)', 10000, 200000, 50000)

st.sidebar.header('📞 Interaction Details')
contact_type = st.sidebar.selectbox('Contact Type', ['Self Inquiry', 'Company Invited'])
pitch_duration = st.sidebar.slider('Duration of Pitch (mins)', 5, 40, 15)
satisfaction = st.sidebar.slider('Pitch Satisfaction Score', 1, 5, 3)

# Aggregate into dataframe dict mapping back to the Scikit-Learn transformer columns
input_data = {
    'Age': age,
    'CityTier': city_tier,
    'Occupation': occupation,
    'Gender': gender,
    'MaritalStatus': marital_status,
    'MonthlyIncome': income,
    'TypeofContact': contact_type,
    'DurationOfPitch': pitch_duration,
    'PitchSatisfactionScore': satisfaction,

    # Hardcoded typical defaults to ensure pipeline Schema checks pass
    'NumberOfPersonVisiting': 3,
    'PreferredPropertyStar': 3,
    'NumberOfTrips': 2,
    'Passport': 0,
    'OwnCar': 1,
    'NumberOfChildrenVisiting': 1,
    'Designation': 'Executive',
    'ProductPitched': 'Basic',
    'NumberOfFollowups': 3
}

features = pd.DataFrame([input_data])

if st.button('Predict Package Purchase'):
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.success(f"🎯 The customer is **LIKELY** to purchase the package! (Probability: {probability:.2%})")
    else:
        st.error(f"🛑 The customer is **UNLIKELY** to purchase the package. (Probability: {probability:.2%})")
