import mlflow
import pandas as pd
import os
import sys
import joblib
import yaml
import dagshub

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.forecaster import generate_multi_horizon_forecast

def batch_inference_stage():
    print("--- Batch Inference Stage ---")
    
    # Initialize Dagshub & MLflow to fetch the production model
    dagshub.init(repo_owner='NaumanRafique12', repo_name='ecommerce-retail-store-inventory-forecasting', mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/NaumanRafique12/ecommerce-retail-store-inventory-forecasting.mlflow")
    
    model_name = "Ecom_Demand_Forecast_WAPE_Model"
    
    try:
        # Load the production model from the registry
        print(f"Fetching latest production model: {model_name}")
        model_uri = f"models:/{model_name}/Production"
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        print(f"Could not load Production model from registry: {e}")
        print("Falling back to local joblib model if available...")
        if os.path.exists("models/lightgbm_model.joblib"):
            model = joblib.load("models/lightgbm_model.joblib")
        else:
            print("No models found. Please run training first.")
            return

    # Load the latest processed data to start forecasting from
    if not os.path.exists("data/processed/processed.csv"):
        print("Processed data not found. Please run preprocessing first.")
        return

    processed_df = pd.read_csv("data/processed/processed.csv", index_col=0)
    
    # Generate forecasts for different horizons
    horizons = [7, 14, 30]
    results = {}
    
    print("Generating batch forecasts...")
    for days in horizons:
        forecasts = generate_multi_horizon_forecast(model, processed_df, horizon_days=days)
        results[f"next_{days}_days"] = [float(x) for x in forecasts]
        print(f"Generated {days}-day forecast.")

    # Save results
    os.makedirs("data/forecasts", exist_ok=True)
    import json
    with open("data/forecasts/batch_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("Batch inference results saved to data/forecasts/batch_results.json")

if __name__ == "__main__":
    batch_inference_stage()
