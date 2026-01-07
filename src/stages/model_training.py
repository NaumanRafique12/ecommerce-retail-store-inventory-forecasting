import yaml
import json
import os
import sys
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import dagshub

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.traditional_models import train_lightgbm, train_random_forest, evaluate_model

def training_stage():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("--- Training Stage ---")
    
    # Initialize Dagshub & MLflow
    dagshub.init(repo_owner='NaumanRafique12', repo_name='ecommerce-retail-store-inventory-forecasting', mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/NaumanRafique12/ecommerce-retail-store-inventory-forecasting.mlflow")
    mlflow.set_experiment('Production_Pipeline_Runs')

    if not os.path.exists("data/processed/processed.csv"):
        print("Processed data not found. Please run preprocessing first.")
        return

    df = pd.read_csv("data/processed/processed.csv", index_col=0)
    
    target_col = 'Units_Sold'
    if target_col not in df.columns:
        target_col = 'Units Sold'

    # Split data
    train_size = int(len(df) * 0.8)
    train, test = df[:train_size], df[train_size:]
    
    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]
    X_test = test.drop(columns=[target_col])
    y_test = test[target_col]

    os.makedirs("models", exist_ok=True)

    # Dictionary to hold metrics for JSON
    all_metrics = {}

    # 1. Train & Save LightGBM
    print("Training LightGBM...")
    lgbm_params = config['models'].get('lightgbm', {'n_estimators': 100, 'num_leaves': 31})
    
    with mlflow.start_run(run_name="LGBM_Main_Pipeline"):
        lgbm_model = train_lightgbm(X_train, y_train, lgbm_params)
        lgbm_metrics, _ = evaluate_model(lgbm_model, X_test, y_test)
        
        # Log to MLflow
        mlflow.log_params(lgbm_params)
        for m_name, m_val in lgbm_metrics.items():
            mlflow.log_metric(m_name, m_val)
        
        signature = mlflow.models.infer_signature(X_train, lgbm_model.predict(X_train))
        mlflow.sklearn.log_model(lgbm_model, "lightgbm_model", signature=signature)
        mlflow.set_tag("model_type", "LightGBM")
        mlflow.set_tag("stage", "production")
        
        print(f"LightGBM Metrics: {lgbm_metrics}")
        joblib.dump(lgbm_model, "models/lightgbm_model.joblib")
        all_metrics["lightgbm"] = lgbm_metrics

    # 2. Train & Save Random Forest
    print("Training Random Forest...")
    rf_params = config['models'].get('random_forest', {'n_estimators': 100, 'max_depth': 20})
    
    with mlflow.start_run(run_name="RF_Main_Pipeline"):
        rf_model = train_random_forest(X_train, y_train, rf_params)
        rf_metrics, _ = evaluate_model(rf_model, X_test, y_test)
        
        # Log to MLflow
        mlflow.log_params(rf_params)
        for m_name, m_val in rf_metrics.items():
            mlflow.log_metric(m_name, m_val)
        
        signature_rf = mlflow.models.infer_signature(X_train, rf_model.predict(X_train))
        mlflow.sklearn.log_model(rf_model, "random_forest_model", signature=signature_rf)
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("stage", "production")

        print(f"Random Forest Metrics: {rf_metrics}")
        joblib.dump(rf_model, "models/random_forest_model.joblib")
        all_metrics["random_forest"] = rf_metrics

    # Save metrics in JSON for DVC tracking
    with open("metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)

    print("Models and metrics.json saved successfully. Results logged to MLflow.")

if __name__ == "__main__":
    training_stage()
