from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import os

app = FastAPI(title="Customer Intelligence Platform")

# -----------------------------
# Load trained ML model
# -----------------------------
MODEL_PATH = os.path.join("app", "ml", "model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# -----------------------------
# Input Schema (MUST MATCH TRAINING FEATURES)
# -----------------------------
class CustomerData(BaseModel):
    account_length: int
    international_plan: int   # 0 = No, 1 = Yes
    voice_mail_plan: int      # 0 = No, 1 = Yes
    number_vmail_messages: int

    total_day_minutes: float
    total_day_calls: int
    total_day_charge: float

    total_eve_minutes: float
    total_eve_calls: int
    total_eve_charge: float

    total_night_minutes: float
    total_night_calls: int
    total_night_charge: float

    total_intl_minutes: float
    total_intl_calls: int
    total_intl_charge: float

    customer_service_calls: int

# -----------------------------
# Health Check
# -----------------------------
@app.get("/")
def health_check():
    return {"status": "API running successfully"}

# -----------------------------
# Churn Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict_churn(data: CustomerData):

    features = np.array([[  
        data.account_length,
        data.international_plan,
        data.voice_mail_plan,
        data.number_vmail_messages,

        data.total_day_minutes,
        data.total_day_calls,
        data.total_day_charge,

        data.total_eve_minutes,
        data.total_eve_calls,
        data.total_eve_charge,

        data.total_night_minutes,
        data.total_night_calls,
        data.total_night_charge,

        data.total_intl_minutes,
        data.total_intl_calls,
        data.total_intl_charge,

        data.customer_service_calls
    ]])

    prediction = model.predict(features)[0]

    return {
        "churn_prediction": "Yes" if prediction == 1 else "No"
    }
