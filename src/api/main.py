from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

# Setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "src" / "models" / "aqi_model.joblib"

app = FastAPI(title="LA AQI Predictor API", version="1.0")

# Load Model
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None
    print("WARNING: Model not found. Predictions will fail.")

class PredictionRequest(BaseModel):
    temperature: float
    humidity: float

class PredictionResponse(BaseModel):
    predicted_pm2_5: float
    air_quality_category: str

def get_aqi_category(pm2_5):
    if pm2_5 <= 12.0: return "Good"
    elif pm2_5 <= 35.4: return "Moderate"
    elif pm2_5 <= 55.4: return "Unhealthy for Sensitive Groups"
    elif pm2_5 <= 150.4: return "Unhealthy"
    elif pm2_5 <= 250.4: return "Very Unhealthy"
    else: return "Hazardous"

@app.get("/")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Feature Engineering on the fly (must match training)
    features = pd.DataFrame([{
        'temperature': request.temperature,
        'humidity': request.humidity,
        'temp_humidity_interaction': request.temperature * request.humidity
    }])
    
    prediction = model.predict(features)[0]
    category = get_aqi_category(prediction)
    
    return {
        "predicted_pm2_5": round(prediction, 2),
        "air_quality_category": category
    }
