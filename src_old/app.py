# app.py

import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd

app = FastAPI()

# Determine the absolute path of the current file (app.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to the current directory
models_dir = os.path.join(current_dir, '..', 'models')
static_dir = os.path.join(current_dir, '..', 'static')
templates_dir = os.path.join(current_dir, '..', 'templates')

# Mount the static directory for serving CSS, JS, images, etc.
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Initialize the templates directory
templates = Jinja2Templates(directory=templates_dir)

# Load the trained pipeline
model_path = os.path.join(models_dir, "fraud_claim_logistic_model.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at path: {model_path}")

model = joblib.load(model_path)

# Define feature columns used in the pipeline (excluding 'ClaimID' and 'Provider')
feature_columns = [
    'Total_Claims_Per_Bene',
    'TimeInHptal',
    'Provider_Claim_Frequency',
    'ChronicCond_stroke_Yes',
    'DeductibleAmtPaid',
    'NoOfMonths_PartBCov',
    'NoOfMonths_PartACov',
    'OPD_Flag_Yes',
    'Diagnosis Count',
    'ChronicDisease_Count', 
    'Age'
]

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction_text": None})

@app.post("/predict", response_class=HTMLResponse)
async def make_prediction(request: Request):
    form = await request.form()
    
    try:
        # Extract features from the form
        input_data = {}
        for feature in feature_columns:
            value = form.get(feature)
            if feature in ['ChronicCond_stroke_Yes', 'OPD_Flag_Yes']:
                # Binary features expected to be 0 or 1
                if value not in ['0', '1']:
                    raise ValueError(f"{feature} must be 0 or 1.")
                input_data[feature] = int(value)
            else:
                # Numerical features
                try:
                    input_data[feature] = float(value)
                except ValueError:
                    raise ValueError(f"{feature} must be a numerical value.")
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        
        # Make prediction using the pipeline
        probability = model.predict_proba(input_df)[0][1]  # Probability of fraud
        probability_percentage = round(probability * 100, 2)
        
        prediction_text = f"Fraud Probability: {probability_percentage}%"
    
    except Exception as e:
        prediction_text = f"Error: {str(e)}"
    
    return templates.TemplateResponse("index.html", {"request": request, "prediction_text": prediction_text})
