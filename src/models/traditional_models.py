from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import sys

def train_random_forest(X_train, y_train, params):
    """Train Random Forest model."""
    print(f"Starting Random Forest training (n_jobs={params.get('n_jobs', 1)})...")
    sys.stdout.flush()
    model = RandomForestRegressor(**params, random_state=42, verbose=1)
    model.fit(X_train, y_train)
    print("Random Forest training complete.")
    sys.stdout.flush()
    return model

def train_lightgbm(X_train, y_train, params):
    """Train LightGBM model."""
    print("Starting LightGBM training...")
    sys.stdout.flush()
    model = lgb.LGBMRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    print("LightGBM training complete.")
    sys.stdout.flush()
    return model

def train_xgboost(X_train, y_train, params):
    """Train XGBoost model."""
    print("Starting XGBoost training...")
    sys.stdout.flush()
    # Explicitly set verbosity to avoid clutter but show progress
    model = xgb.XGBRegressor(**params, random_state=42, verbosity=1)
    model.fit(X_train, y_train)
    print("XGBoost training complete.")
    sys.stdout.flush()
    return model

def calculate_metrics(y_true, y_pred):
    """Calculate MAE, MAPE, WAPE, and RMSE."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    y_true_safe = np.where(y_true == 0, 1e-10, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100
    
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "WAPE": float(wape)
    }

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance across multiple metrics."""
    print(f"Evaluating model: {type(model).__name__}")
    sys.stdout.flush()
    predictions = model.predict(X_test)
    metrics = calculate_metrics(y_test, predictions)
    print(f"Evaluation complete. Metrics: {metrics}")
    sys.stdout.flush()
    return metrics, predictions
