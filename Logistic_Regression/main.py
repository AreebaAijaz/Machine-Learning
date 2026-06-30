from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import io

app = FastAPI(title="Customer Churn Predictor API")

# Load model once at startup
model = joblib.load("model/churn_model.pkl")

# --- Single customer input schema ---
class CustomerData(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float

# --- Endpoint 1: Single prediction ---
@app.post("/predict")
def predict(customer: CustomerData):
    data = pd.DataFrame([customer.dict()])
    probability = model.predict_proba(data)[0][1]
    prediction = model.predict(data)[0]

    risk = "High" if probability >= 0.7 else "Medium" if probability >= 0.4 else "Low"

    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 3),
        "risk_level": risk
    }

# --- Endpoint 2: Batch prediction (CSV upload) ---
@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    probabilities = model.predict_proba(df)[:, 1]
    predictions = model.predict(df)

    df['churn_prediction'] = predictions
    df['churn_probability'] = probabilities.round(3)
    df['risk_level'] = df['churn_probability'].apply(
        lambda p: "High" if p >= 0.7 else "Medium" if p >= 0.4 else "Low"
    )

    return df[['churn_prediction', 'churn_probability', 'risk_level']].to_dict(orient='records')

# --- Health check ---
@app.get("/")
def root():
    return {"status": "Churn Predictor API is running"}