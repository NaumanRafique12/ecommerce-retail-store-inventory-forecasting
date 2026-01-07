import pandas as pd

def add_calendar_features(df):
    """Extract calendar-based features from datetime index."""
    df['Month'] = df.index.month
    df['Day'] = df.index.day
    df['Week'] = df.index.isocalendar().week
    df['Year'] = df.index.year
    return df

def add_lag_features(df, target_col='Units_Sold', lags=[1, 7, 30]):
    """Add lag features for the target column."""
    if target_col not in df.columns and 'Units Sold' in df.columns:
        target_col = 'Units Sold'
        
    for lag in lags:
        df[f'Lag_{lag}'] = df[target_col].shift(lag)
    return df

def add_rolling_features(df, target_col='Units_Sold', windows=[7, 30]):
    """Add rolling mean and std features."""
    if target_col not in df.columns and 'Units Sold' in df.columns:
        target_col = 'Units Sold'

    for window in windows:
        df[f'Rolling_Mean_{window}'] = df[target_col].rolling(window=window).mean()
        df[f'Rolling_Std_{window}'] = df[target_col].rolling(window=window).std()
    return df
