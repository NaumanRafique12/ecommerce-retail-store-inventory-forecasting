import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def handle_dates(df, date_col='Date'):
    """Convert column to datetime and set as index."""
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    return df

def encode_categorical(df):
    """Convert categorical columns to dummy variables."""
    categorical_cols = df.select_dtypes(include=['object']).columns
    # Drop IDs before encoding if they are not useful for modeling directly
    cols_to_drop = [col for col in ['Store ID', 'Product ID'] if col in categorical_cols]
    df = df.drop(columns=cols_to_drop)
    categorical_cols = [c for c in categorical_cols if c not in cols_to_drop]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df

def normalize_numerical(df):
    """Normalize numerical features."""
    scaler = MinMaxScaler()
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df, scaler
