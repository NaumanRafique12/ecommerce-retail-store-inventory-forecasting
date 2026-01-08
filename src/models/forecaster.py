import pandas as pd
import numpy as np
import datetime

class RecursiveForecaster:
    def __init__(self, model, feature_pipeline_config):
        """
        model: Trained model (XGB/LGBM)
        feature_pipeline_config: Dictionary with lags and rolling windows
        """
        self.model = model
        self.lags = feature_pipeline_config.get('lags', [1, 7, 30])
        self.windows = feature_pipeline_config.get('rolling_windows', [7, 30])

    def forecast(self, current_data_row, horizon_days=30):
        """
        Generates predictions for the next N days.
        current_data_row: The most recent row from the processed dataset (normalized).
        """
        predictions = []
        current_features = current_data_row.copy()
        
        # We need a history of sliding values to calculate lags and windows
        # For simplicity in this recursive demo, we'll maintain a buffer
        history = [current_features['Units_Sold']] if 'Units_Sold' in current_features else [0.0]
        
        for i in range(horizon_days):
            # 1. Prepare feature vector for prediction
            # Note: In a real production system, you'd need to re-apply the full 
            # feature engineering logic for each step. 
            # Here we simulate the update of key temporal features.
            
            # Extract features for model (excluding the target column if present)
            X_df = current_features.drop('Units_Sold', errors='ignore').to_frame().T
            
            # 2. Predict next day
            pred = self.model.predict(X_df)[0]
            pred = max(0, pred) # Demand can't be negative
            predictions.append(pred)
            
            # 3. Update features for the next step (Recursive)
            history.append(pred)
            
            # Update Lags (Simplified: only Lag_1 is updated for deep recursion)
            if 'Lag_1' in current_features:
                current_features['Lag_1'] = pred
            
            # In a more advanced version, we would update all lags and rolling stats:
            # for lag in self.lags:
            #     if len(history) > lag:
            #         current_features[f'Lag_{lag}'] = history[-lag]
            
            # Update Calendar features (Simulation)
            # This would normally involve incrementing the index and re-extracting Month, Day, etc.
            
        return predictions

def generate_multi_horizon_forecast(model, last_processed_data, horizon_days=30):
    """
    Wrapper to handle the forecasting for all products.
    """
    # In a real batch scenario, we iterate per Product/Store
    # Here we show the logic for a single series
    forecaster = RecursiveForecaster(model, {'lags': [1, 7, 30], 'rolling_windows': [7, 30]})
    
    # Take the last available data point
    latest_row = last_processed_data.iloc[-1]
    
    forecasts = forecaster.forecast(latest_row, horizon_days=horizon_days)
    return forecasts
