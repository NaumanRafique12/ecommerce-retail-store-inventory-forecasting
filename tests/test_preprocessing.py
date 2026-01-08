import pandas as pd
import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.preprocessing.preprocess import sanitize_columns, handle_dates

def test_sanitize_columns():
    df = pd.DataFrame(columns=['Units Sold', 'Product ID/Category'])
    df_sanitized = sanitize_columns(df)
    assert 'Units_Sold' in df_sanitized.columns
    assert 'Product_ID_Category' in df_sanitized.columns

def test_handle_dates():
    df = pd.DataFrame({'Date': ['2022-01-01', '2022-01-02'], 'Sales': [10, 20]})
    df_handled = handle_dates(df)
    assert isinstance(df_handled.index, pd.DatetimeIndex)
    assert df_handled.index[0] == pd.Timestamp('2022-01-01')
