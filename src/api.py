from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(
    title="IoT Predictive Maintenance API",
    description="API for Anomaly Detection and RUL Prediction based on sensor data.",
    version="1.0"
)

# Load trained models
model_dir = os.path.join(BASE_DIR, 'models')
try:
    anomaly_model = joblib.load(os.path.join(model_dir, 'anomaly_model.joblib'))
    rul_model = joblib.load(os.path.join(model_dir, 'rul_model.joblib'))
    scaler = joblib.load(os.path.join(model_dir, 'feature_scaler.joblib'))
    feature_cols = joblib.load(os.path.join(model_dir, 'feature_cols.joblib'))
except Exception as e:
    print(f"Error loading models. Did you run train.py? Error: {e}")
    anomaly_model, rul_model, scaler, feature_cols = None, None, None, None

class SensorData(BaseModel):
    temperature: float
    vibration: float
    pressure: float
    temperature_rolling_mean: float
    temperature_rolling_std: float
    vibration_rolling_mean: float
    vibration_rolling_std: float
    pressure_rolling_mean: float
    pressure_rolling_std: float

class PredictionResponse(BaseModel):
    is_anomaly: bool
    anomaly_score: float # Optional: Isolation Forest score
    predicted_rul_days: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Predictive Maintenance API"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: SensorData):
    if anomaly_model is None or rul_model is None:
        raise HTTPException(status_code=500, detail="Models are not loaded.")

    # Convert incoming data to DataFrame for model consumption
    input_data = pd.DataFrame([data.dict()])
    
    # Ensure correct column order
    input_data = input_data[feature_cols]
    
    # 1. Anomaly Prediction
    scaled_data = scaler.transform(input_data)
    # Returns -1 for anomaly, 1 for normal
    anomaly_pred = anomaly_model.predict(scaled_data)[0]
    
    # Get the raw anomaly score (negative is usually more anomalous)
    anomaly_score = float(anomaly_model.decision_function(scaled_data)[0])
    is_anomaly = bool(anomaly_pred == -1)
    
    # 2. RUL Prediction
    # Note: RUL is typically only relevant if the machine is starting to degrade.
    rul_pred = float(rul_model.predict(input_data)[0])
    
    # Optional logic: If not an anomaly and healthy, cap RUL at a high number
    # For this demo we just return model prediction
    
    return PredictionResponse(
        is_anomaly=is_anomaly,
        anomaly_score=anomaly_score,
        predicted_rul_days=max(0.0, round(rul_pred, 2)) # Don't return negative days
    )
