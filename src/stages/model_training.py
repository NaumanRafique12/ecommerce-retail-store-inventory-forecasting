import yaml
import json
import os
import sys
import pandas as pd
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.traditional_models import train_xgboost, train_lightgbm, evaluate_model

def training_stage():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("--- Training Stage ---")
    df = pd.read_csv("data/processed/processed.csv", index_col=0)
    
    # Split data
    train_size = int(len(df) * 0.8)
    train, test = df[:train_size], df[train_size:]
    
    X_train = train.drop(columns=['Units Sold'])
    y_train = train['Units Sold']
    X_test = test.drop(columns=['Units Sold'])
    y_test = test['Units Sold']

    os.makedirs("models", exist_ok=True)

    # Train & Save XGBoost
    print("Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train, config['models']['xgboost'])
    xgb_metrics, _ = evaluate_model(xgb_model, X_test, y_test)
    print(f"XGBoost Metrics: {xgb_metrics}")
    joblib.dump(xgb_model, "models/xgboost_model.joblib")

    # Train & Save LightGBM
    print("Training LightGBM...")
    lgbm_model = train_lightgbm(X_train, y_train, config['models']['lightgbm'])
    lgbm_metrics, _ = evaluate_model(lgbm_model, X_test, y_test)
    print(f"LightGBM Metrics: {lgbm_metrics}")
    joblib.dump(lgbm_model, "models/lightgbm_model.joblib")

    # Save metrics in JSON
    all_metrics = {
        "xgboost": xgb_metrics,
        "lightgbm": lgbm_metrics
    }
    
    with open("metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)

    print("Models and metrics.json saved successfully.")

if __name__ == "__main__":
    training_stage()
