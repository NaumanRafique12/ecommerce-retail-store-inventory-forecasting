import yaml
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
    xgb_mae, _ = evaluate_model(xgb_model, X_test, y_test)
    print(f"XGBoost MAE: {xgb_mae}")
    joblib.dump(xgb_model, "models/xgboost_model.joblib")

    # Train & Save LightGBM
    print("Training LightGBM...")
    lgbm_model = train_lightgbm(X_train, y_train, config['models']['lightgbm'])
    lgbm_mae, _ = evaluate_model(lgbm_model, X_test, y_test)
    print(f"LightGBM MAE: {lgbm_mae}")
    joblib.dump(lgbm_model, "models/lightgbm_model.joblib")

    # Save metrics
    with open("metrics.yaml", "w") as f:
        yaml.dump({"xgboost_mae": float(xgb_mae), "lightgbm_mae": float(lgbm_mae)}, f)

    print("Models saved to models/ directory.")

if __name__ == "__main__":
    training_stage()
