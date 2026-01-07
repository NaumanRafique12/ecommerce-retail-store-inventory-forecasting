import yaml
import pandas as pd
import os
from src.ingestion.ingest_data import load_data, version_data
from src.ingestion.validate_data import validate_schema, data_quality_checks
from src.preprocessing.preprocess import handle_dates, encode_categorical, normalize_numerical
from src.preprocessing.outlier_handler import remove_outliers
from src.features.feature_engineering import add_calendar_features, add_lag_features, add_rolling_features
from src.models.traditional_models import train_xgboost, train_lightgbm, evaluate_model

def run_pipeline():
    # Load config
    if not os.path.exists("config/config.yaml"):
        raise FileNotFoundError("Config file not found.")
        
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Ingestion
    print("Ingesting data...")
    df = load_data(config['paths']['raw_data'])
    version_data(df, config['paths']['raw_snapshots_dir'])

    # Validation
    print("Validating data...")
    expected_cols = ['Date', 'Store ID', 'Product ID', 'Units Sold', 'Price']
    validate_schema(df, expected_cols)
    quality_report = data_quality_checks(df)
    print(f"Quality Report: {quality_report}")

    # Preprocessing
    print("Preprocessing data...")
    df = handle_dates(df)
    df = remove_outliers(df, ['Units Sold', 'Demand Forecast', 'Inventory Level'])
    df = encode_categorical(df)
    df, scaler = normalize_numerical(df)

    # Feature Engineering
    print("Feature Engineering...")
    df = add_calendar_features(df)
    df = add_lag_features(df, lags=config['features']['lags'])
    df = add_rolling_features(df, windows=config['features']['rolling_windows'])
    df = df.dropna()

    # Split data
    train_size = int(len(df) * 0.8)
    train, test = df[:train_size], df[train_size:]
    
    X_train = train.drop(columns=['Units Sold'])
    y_train = train['Units Sold']
    X_test = test.drop(columns=['Units Sold'])
    y_test = test['Units Sold']

    # Train & Evaluate XGBoost
    print("\nTraining XGBoost...")
    xgb_model = train_xgboost(X_train, y_train, config['models']['xgboost'])
    xgb_mae, _ = evaluate_model(xgb_model, X_test, y_test)
    print(f"XGBoost MAE: {xgb_mae}")

    # Train & Evaluate LightGBM
    print("\nTraining LightGBM...")
    lgbm_model = train_lightgbm(X_train, y_train, config['models']['lightgbm'])
    lgbm_mae, _ = evaluate_model(lgbm_model, X_test, y_test)
    print(f"LightGBM MAE: {lgbm_mae}")

    os.makedirs(config['paths']['processed_dir'], exist_ok=True)
    df.to_csv(os.path.join(config['paths']['processed_dir'], "processed_data.csv"))
    print("\nPipeline completed successfully.")

if __name__ == "__main__":
    run_pipeline()
