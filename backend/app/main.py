from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Customer Intelligence Platform")

# -----------------------------
# Input Schema (Request Body)
# -----------------------------
class CustomerInput(BaseModel):
    age: int
    monthly_spend: float
    tenure_months: int

# -----------------------------
# Health Check
# -----------------------------
@app.get("/")
def health_check():
    return {"status": "API running successfully"}

# -----------------------------
# Prediction Endpoint (Mock)
# -----------------------------
@app.post("/predict")
def predict_customer_behavior(data: CustomerInput):
    # Dummy logic (we’ll replace with ML model)
    score = (data.monthly_spend * 0.6) + (data.tenure_months * 0.4)

    risk = "High" if score < 500 else "Low"

    return {
        "customer_score": score,
        "risk_level": risk
    }

