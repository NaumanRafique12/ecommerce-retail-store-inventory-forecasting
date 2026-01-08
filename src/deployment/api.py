from fastapi import FastAPI, HTTPException
import mlflow
import pandas as pd
import numpy as np
import os
import sys
import dagshub
import uvicorn
import joblib
from contextlib import asynccontextmanager

# Add project root to path for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.forecaster import generate_multi_horizon_forecast

# Global references
MODEL = None
SCALER = None
MODEL_NAME = "Ecom_Demand_Forecast_WAPE_Model"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan event handler for FastAPI."""
    global MODEL, SCALER
    print("--- Initializing API (Loading Model & Scaler) ---")
    try:
        # 1. Load Scaler
        if os.path.exists("models/scaler.joblib"):
            SCALER = joblib.load("models/scaler.joblib")
            print("Scaler loaded successfully.")
        else:
            print("Warning: models/scaler.joblib not found. Predictions will be in normalized scale.")

        # 2. Setup MLflow tracking with token-based auth
        dagshub_token = os.getenv("ABARK_MLOPS") or os.getenv("DAGSHUB_PAT")
        if dagshub_token:
            os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
            os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        
        mlflow.set_tracking_uri("https://dagshub.com/NaumanRafique12/ecommerce-retail-store-inventory-forecasting.mlflow")
        
        # 3. Try loading Production stage
        try:
            model_uri = f"models:/{MODEL_NAME}/Production"
            MODEL = mlflow.sklearn.load_model(model_uri)
            print(f"Production model '{MODEL_NAME}' loaded successfully.")
        except Exception as e_prod:
            print(f"Production stage empty, falling back to latest version: {e_prod}")
            model_uri = f"models:/{MODEL_NAME}/latest"
            MODEL = mlflow.sklearn.load_model(model_uri)
            print(f"Latest model '{MODEL_NAME}' loaded successfully as fallback.")
            
    except Exception as e:
        print(f"Failed to load registry model: {e}")
        if os.path.exists("models/lightgbm_model.joblib"):
            MODEL = joblib.load("models/lightgbm_model.joblib")
            print("Loaded local fallback model (lightgbm).")
    
    yield
    print("--- Shutting down API ---")

app = FastAPI(title="E-Commerce Demand Forecast API", version="1.2", lifespan=lifespan)

@app.get("/")
def health_check():
    return {
        "status": "operational", 
        "model": MODEL_NAME if MODEL else "none",
        "scaler": "loaded" if SCALER else "missing"
    }

@app.get("/predict")
def predict_forecast(horizon: int = 7):
    """
    Generate forecasts for the next N days.
    """
    if horizon not in [7, 14, 30]:
        raise HTTPException(status_code=400, detail="Horizon must be 7, 14, or 30 days.")
    
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is currently unavailable.")
    
    if not os.path.exists("data/processed/processed.csv"):
        raise HTTPException(status_code=500, detail="Processed historical data missing.")
        
    df = pd.read_csv("data/processed/processed.csv", index_col=0)
    
    # Generate recursive forecast
    raw_forecasts = generate_multi_horizon_forecast(MODEL, df, horizon_days=horizon)
    
    # Inverse Transform if scaler is available
    final_forecasts = raw_forecasts
    if SCALER:
        try:
            # We need to create a dummy array matching the scaler's input dimension
            # We know 'Units_Sold' is at index 1
            n_features = SCALER.n_features_in_
            dummy = np.zeros((len(raw_forecasts), n_features))
            
            # Find the actual index of Units_Sold dynamically
            target_idx = list(SCALER.feature_names_in_).index('Units_Sold')
            dummy[:, target_idx] = raw_forecasts
            
            unscaled = SCALER.inverse_transform(dummy)
            final_forecasts = unscaled[:, target_idx]
        except Exception as e:
            print(f"Inverse transform failed: {e}")
    
    # Format dates
    last_date = pd.to_datetime(df.index[-1])
    forecast_dates = [(last_date + pd.Timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(len(final_forecasts))]
    
    response = {
        "model_version": "Active",
        "horizon_days": horizon,
        "predictions": dict(zip(forecast_dates, [int(round(float(p))) for p in final_forecasts])) # Return as integers (actual units)
    }
    
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
