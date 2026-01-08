import yaml
import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.preprocessing.preprocess import handle_dates, encode_categorical, normalize_numerical, sanitize_columns
from src.preprocessing.outlier_handler import remove_outliers
from src.features.feature_engineering import add_calendar_features, add_lag_features, add_rolling_features

def preprocess_stage():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("--- Preprocessing Stage ---")
    df = pd.read_csv("data/raw/raw.csv")
    
    # 1. Sanitize column names (remove whitespace)
    df = sanitize_columns(df)
    
    # 2. Preprocessing
    df = handle_dates(df, date_col='Date')
    
    # Use sanitized names for outlier removal
    outlier_cols = ['Units_Sold', 'Demand_Forecast', 'Inventory_Level']
    # If sanitization renamed others, adjust
    df = remove_outliers(df, outlier_cols)
    
    df = encode_categorical(df)
    df, scaler = normalize_numerical(df)
    
    # Save scaler for inverse transform
    os.makedirs("models", exist_ok=True)
    import joblib
    joblib.dump(scaler, "models/scaler.joblib")

    # 3. Feature Engineering
    # Feature engineering will also need to handle sanitized names
    df = add_calendar_features(df)
    df = add_lag_features(df, lags=config['features']['lags'])
    df = add_rolling_features(df, windows=config['features']['rolling_windows'])
    df = df.dropna()

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/processed.csv")
    print("Processed data saved to data/processed/processed.csv")

if __name__ == "__main__":
    preprocess_stage()
