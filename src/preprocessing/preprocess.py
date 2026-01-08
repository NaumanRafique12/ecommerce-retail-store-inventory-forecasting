import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def sanitize_columns(df):
    """Replace whitespaces and special characters in column names."""
    df.columns = [col.replace(' ', '_').replace('/', '_') for col in df.columns]
    return df

def handle_dates(df, date_col='Date'):
    """Convert column to datetime and set as index."""
    # After sanitization, 'Date' is likely still 'Date' unless it had spaces.
    # Be flexible.
    if date_col not in df.columns and date_col.replace(' ', '_') in df.columns:
        date_col = date_col.replace(' ', '_')
    
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
    return df

def encode_categorical(df):
    """Convert categorical columns to dummy variables."""
    # After sanitization, 'Store ID' becomes 'Store_ID'
    cols_to_drop = [col for col in ['Store_ID', 'Product_ID', 'Store ID', 'Product ID'] if col in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    return df

def normalize_numerical(df):
    """Normalize numerical features."""
    scaler = MinMaxScaler()
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df, scaler
