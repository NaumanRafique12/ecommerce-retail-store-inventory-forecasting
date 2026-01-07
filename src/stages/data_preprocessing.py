import yaml
import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.preprocessing.preprocess import handle_dates, encode_categorical, normalize_numerical
from src.preprocessing.outlier_handler import remove_outliers
from src.features.feature_engineering import add_calendar_features, add_lag_features, add_rolling_features

def preprocess_stage():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("--- Preprocessing Stage ---")
    df = pd.read_csv("data/raw/raw.csv")
    
    # Preprocessing
    df = handle_dates(df)
    df = remove_outliers(df, ['Units Sold', 'Demand Forecast', 'Inventory Level'])
    df = encode_categorical(df)
    df, _ = normalize_numerical(df)

    # Feature Engineering
    df = add_calendar_features(df)
    df = add_lag_features(df, lags=config['features']['lags'])
    df = add_rolling_features(df, windows=config['features']['rolling_windows'])
    df = df.dropna()

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/processed.csv")
    print("Processed data saved to data/processed/processed.csv")

if __name__ == "__main__":
    preprocess_stage()
