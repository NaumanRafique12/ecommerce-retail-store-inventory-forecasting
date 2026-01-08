import yaml
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import make_scorer
import lightgbm as lgb
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.data

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.traditional_models import calculate_metrics

def wape_scorer(y_true, y_pred):
    """Custom scorer for WAPE."""
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)

def tune_hyperparameters():
    print("--- Hyperparameter Tuning based on WAPE ---")
    
    # Setup MLflow tracking with token-based auth
    dagshub_token = os.getenv("ABARK_MLOPS") or os.getenv("DAGSHUB_PAT")
    if dagshub_token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    
    mlflow.set_tracking_uri("https://dagshub.com/NaumanRafique12/ecommerce-retail-store-inventory-forecasting.mlflow")
    mlflow.set_experiment('Demand_Forecasting_WAPE_Tuning')

    # Load processed data
    if not os.path.exists("data/processed/processed.csv"):
        print("Processed data not found. Please run preprocessing first.")
        return

    df = pd.read_csv("data/processed/processed.csv", index_col=0)
    
    # Sampling for speed
    sample_size = 15000
    if len(df) > sample_size:
        print(f"Sampling {sample_size} rows for tuning.")
        df = df.sample(n=sample_size, random_state=42).sort_index()

    # Split data
    target_col = 'Units_Sold'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Prepare datasets for MLflow
    train_data_info = X_train.copy()
    train_data_info[target_col] = y_train
    mlflow_train_df = mlflow.data.from_pandas(train_data_info, name="training_dataset")

    # Custom WAPE scorer
    wape_scoring = make_scorer(wape_scorer, greater_is_better=False)
    scoring = {
        'MAE': 'neg_mean_absolute_error',
        'WAPE': wape_scoring
    }

    # Safe n_jobs for Windows
    SAFE_N_JOBS = 2

    # 1. Tuning XGBoost
    print("\nTuning XGBoost (Refit on WAPE)...")
    xgb_param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [4, 6],
        'learning_rate': [0.01, 0.1]
    }
    xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=SAFE_N_JOBS)
    grid_xgb = GridSearchCV(xgb_model, xgb_param_grid, cv=3, scoring=scoring, refit='WAPE', n_jobs=SAFE_N_JOBS)

    with mlflow.start_run(run_name="XGBoost_WAPE_GridSearch") as parent:
        grid_xgb.fit(X_train, y_train)

        for i in range(len(grid_xgb.cv_results_['params'])):
            with mlflow.start_run(run_name=f"XGB_Trial_{i}", nested=True):
                mlflow.log_params(grid_xgb.cv_results_['params'][i])
                mlflow.log_metric("MAE", -grid_xgb.cv_results_['mean_test_MAE'][i])
                mlflow.log_metric("WAPE", -grid_xgb.cv_results_['mean_test_WAPE'][i])
                mlflow.set_tag("model_type", "XGBoost")
        
        # Log best model in parent
        best_metrics_xgb, _ = calculate_metrics(y_test, grid_xgb.best_estimator_.predict(X_test))
        mlflow.log_params(grid_xgb.best_params_)
        for m_name, m_val in best_metrics_xgb.items():
            mlflow.log_metric(m_name, m_val)
        
        signature_xgb = mlflow.models.infer_signature(X_train, grid_xgb.best_estimator_.predict(X_train))
        mlflow.sklearn.log_model(grid_xgb.best_estimator_, "xgboost_model", signature=signature_xgb)
        mlflow.set_tag("author", "Nauman")

    # 2. Tuning LightGBM
    print("\nTuning LightGBM (Refit on WAPE)...")
    lgbm_param_grid = {
        'n_estimators': [50, 100],
        'num_leaves': [20, 31],
        'learning_rate': [0.01, 0.1]
    }
    lgbm = lgb.LGBMRegressor(random_state=42, verbose=-1, n_jobs=1)
    grid_lgbm = GridSearchCV(lgbm, lgbm_param_grid, cv=3, scoring=scoring, refit='WAPE', n_jobs=SAFE_N_JOBS)

    with mlflow.start_run(run_name="LightGBM_WAPE_GridSearch") as parent:
        grid_lgbm.fit(X_train, y_train)

        for i in range(len(grid_lgbm.cv_results_['params'])):
            with mlflow.start_run(run_name=f"LGBM_Trial_{i}", nested=True):
                mlflow.log_params(grid_lgbm.cv_results_['params'][i])
                mlflow.log_metric("MAE", -grid_lgbm.cv_results_['mean_test_MAE'][i])
                mlflow.log_metric("WAPE", -grid_lgbm.cv_results_['mean_test_WAPE'][i])
                mlflow.set_tag("model_type", "LightGBM")
        
        # Log best model in parent
        best_metrics, _ = calculate_metrics(y_test, grid_lgbm.best_estimator_.predict(X_test))
        mlflow.log_params(grid_lgbm.best_params_)
        for m_name, m_val in best_metrics.items():
            mlflow.log_metric(m_name, m_val)
        
        signature = mlflow.models.infer_signature(X_train, grid_lgbm.best_estimator_.predict(X_train))
        mlflow.sklearn.log_model(grid_lgbm.best_estimator_, "lightgbm_model", signature=signature)
        mlflow.set_tag("author", "Nauman")

    print("\nHyperparameter tuning complete for XGBoost and LightGBM (WAPE-optimized).")

if __name__ == "__main__":
    tune_hyperparameters()
