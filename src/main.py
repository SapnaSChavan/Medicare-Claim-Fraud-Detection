from fastapi import FastAPI
import pandas as pd
import numpy as np
from pydantic import BaseModel
import joblib
import os
print(os.getcwd())

# Load your pre-trained model (update the path as necessary)
model = joblib.load("fraud_claim_logistic_model.pkl")  # Update the path

app = FastAPI()

def categorize_probabilities(prob_column):
    # Convert the probability to a pandas Series if it is not already
    prob_series = pd.Series(prob_column)
    
    # Apply the categorization based on predefined conditions
    return pd.cut(prob_series, bins=[0, 0.148682, 0.354855, 1.0], labels=['Low', 'Medium', 'High'], right=False)

# Define the request body structure
# Define the input data schema
class ClaimData(BaseModel):
    Total_Claims_Per_Bene: float
    TimeInHptal: int
    Provider_Claim_Frequency: int
    ChronicCond_stroke_Yes: bool
    DeductibleAmtPaid: float
    NoOfMonths_PartBCov: int
    NoOfMonths_PartACov: int
    OPD_Flag_Yes: bool
    Diagnosis_Count: int
    ChronicDisease_Count: int
    Age: int
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Medicare Claim Fraud Detection API!"}

@app.post("/predict")
async def predict_claim(data: ClaimData):
    # Remove ClaimID and Provider, use only the necessary features for prediction
    input_data = np.array([[
        data.Total_Claims_Per_Bene,
        data.TimeInHptal,
        data.Provider_Claim_Frequency,
        data.ChronicCond_stroke_Yes,
        data.DeductibleAmtPaid,
        data.NoOfMonths_PartBCov,
        data.NoOfMonths_PartACov,
        data.OPD_Flag_Yes,
        data.Diagnosis_Count,
        data.ChronicDisease_Count,
        data.Age
    ]])

    try:
        # Perform the prediction using the model, predict_proba gives probabilities
        prediction_prob = model.predict_proba(input_data)[0][1]  # Get the probability of class 1 (Fraud)

        # Categorize the probability into Low, Medium, High
        fraud_risk_category = categorize_probabilities([prediction_prob])[0]
        
        # Return both probability and fraud risk category
        return {
            "prediction_probability": prediction_prob,
            "fraud_risk_category": fraud_risk_category
        }

    except Exception as e:
        return ("Prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "Healthy"}