import yaml
import json
import os
import sys
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.traditional_models import train_lightgbm, train_xgboost, evaluate_model

def setup_mlflow_tracking():
    """Setup MLflow tracking with Dagshub using environment variables."""
    dagshub_token = os.getenv("ABARK_MLOPS") or os.getenv("DAGSHUB_PAT")
    if dagshub_token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        print("Using token-based authentication for Dagshub")
    
    dagshub_url = "https://dagshub.com"
    repo_owner = "NaumanRafique12"
    repo_name = "ecommerce-retail-store-inventory-forecasting"
    
    mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

def training_stage():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("--- Training Stage ---")
    
    # Initialize MLflow with token-based auth
    setup_mlflow_tracking()
    mlflow.set_experiment('Production_Pipeline_Runs')

    if not os.path.exists("data/processed/processed.csv"):
        print("Processed data not found. Please run preprocessing first.")
        return

    df = pd.read_csv("data/processed/processed.csv", index_col=0)
    
    target_col = 'Units_Sold'
    if target_col not in df.columns:
        target_col = 'Units Sold'

    # Split data (Temporal split is better for time series context)
    train_size = int(len(df) * 0.8)
    train, test = df[:train_size], df[train_size:]
    
    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]
    X_test = test.drop(columns=[target_col])
    y_test = test[target_col]

    os.makedirs("models", exist_ok=True)

    # Dictionary to hold metrics for JSON
    all_metrics = {}

    # 1. Train & Save XGBoost
    print("Training XGBoost...")
    xgb_params = config['models'].get('xgboost', {'n_estimators': 100, 'max_depth': 6})
    with mlflow.start_run(run_name="XGB_Main_Pipeline", nested=True):
        xgb_model = train_xgboost(X_train, y_train, xgb_params)
        xgb_metrics, _ = evaluate_model(xgb_model, X_test, y_test)
        
        mlflow.log_params(xgb_params)
        for m_name, m_val in xgb_metrics.items():
            mlflow.log_metric(m_name, m_val)
        
        signature = mlflow.models.infer_signature(X_train, xgb_model.predict(X_train))
        mlflow.sklearn.log_model(xgb_model, "xgboost_model", signature=signature)
        mlflow.set_tag("model_type", "XGBoost")
        
        joblib.dump(xgb_model, "models/xgboost_model.joblib")
        all_metrics["xgboost"] = xgb_metrics

    # 2. Train & Save LightGBM
    print("Training LightGBM...")
    lgbm_params = config['models'].get('lightgbm', {'n_estimators': 100, 'num_leaves': 31})
    with mlflow.start_run(run_name="LGBM_Main_Pipeline", nested=True):
        lgbm_model = train_lightgbm(X_train, y_train, lgbm_params)
        lgbm_metrics, _ = evaluate_model(lgbm_model, X_test, y_test)
        
        mlflow.log_params(lgbm_params)
        for m_name, m_val in lgbm_metrics.items():
            mlflow.log_metric(m_name, m_val)
        
        signature_lgbm = mlflow.models.infer_signature(X_train, lgbm_model.predict(X_train))
        mlflow.sklearn.log_model(lgbm_model, "lightgbm_model", signature=signature_lgbm)
        mlflow.set_tag("model_type", "LightGBM")
        
        joblib.dump(lgbm_model, "models/lightgbm_model.joblib")
        all_metrics["lightgbm"] = lgbm_metrics

    # Save metrics in JSON for DVC tracking
    with open("metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)

    print("Models and metrics.json saved successfully. Results logged to MLflow.")

if __name__ == "__main__":
    training_stage()
