import streamlit as st
import requests
import json

# Define the FastAPI URL
fastapi_url = "http://127.0.0.1:8000/predict"

st.title('Medicare Claim Fraud Detection')

# Create input fields for each feature
Total_Claims_Per_Bene = st.number_input("Total Claims Per Beneficiary", value=0)
TimeInHptal = st.number_input("Time in Hospital (in days)", value=0)
Provider_Claim_Frequency = st.number_input("Provider Claim Frequency", value=0)
ChronicCond_stroke_Yes = st.selectbox("Chronic Condition Stroke (Yes=1, No=0)", [0, 1])
DeductibleAmtPaid = st.number_input("Deductible Amount Paid", value=0)
NoOfMonths_PartBCov = st.number_input("No of Months with Part B Coverage", value=0)
NoOfMonths_PartACov = st.number_input("No of Months with Part A Coverage", value=0)
OPD_Flag_Yes = st.selectbox("Outpatient Department Flag (Yes=1, No=0)", [0, 1])
Diagnosis_Count = st.number_input("Diagnosis Count", value=0)
ChronicDisease_Count = st.number_input("Chronic Disease Count", value=0)
Age = st.number_input("Age", value=65)

# Create a submit button
if st.button('Predict'):
    # Prepare the data payload to be sent to FastAPI
    data = {
        "Total_Claims_Per_Bene": Total_Claims_Per_Bene,
        "TimeInHptal": TimeInHptal,
        "Provider_Claim_Frequency": Provider_Claim_Frequency,
        "ChronicCond_stroke_Yes": ChronicCond_stroke_Yes,
        "DeductibleAmtPaid": DeductibleAmtPaid,
        "NoOfMonths_PartBCov": NoOfMonths_PartBCov,
        "NoOfMonths_PartACov": NoOfMonths_PartACov,
        "OPD_Flag_Yes": OPD_Flag_Yes,
        "Diagnosis_Count": Diagnosis_Count,
        "ChronicDisease_Count": ChronicDisease_Count,
        "Age": Age
    }

    # Send a POST request to FastAPI server
    response = requests.post(fastapi_url, data=json.dumps(data))

    # Display the prediction result
    if response.status_code == 200:
        result = response.json()['fraud_risk_category']
        probability = response.json()['prediction_probability']
        st.success(f"Fraud Risk Category: {result}")
        st.success(f"The probability of the claim being fraudulent is: {round(probability, 3)*100}{'%'}")
    else:
        st.error(f"Error: {response.status_code} - {response.text}")