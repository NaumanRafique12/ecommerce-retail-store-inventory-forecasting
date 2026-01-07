import pandas as pd
import numpy as np

def validate_schema(df, expected_columns):
    """Validate that all expected columns are present."""
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return True

def data_quality_checks(df):
    """Perform basic data quality checks."""
    report = {
        "null_counts": df.isnull().sum().to_dict(),
        "negative_sales": (df['Units Sold'] < 0).sum() if 'Units Sold' in df.columns else 0,
        "negative_prices": (df['Price'] < 0).sum() if 'Price' in df.columns else 0
    }
    
    # Check date continuity (if Date is index or column)
    if 'Date' in df.columns:
        dates = pd.to_datetime(df['Date'])
        report['date_range'] = f"{dates.min()} to {dates.max()}"
        report['is_continuous'] = dates.is_monotonic_increasing
    
    return report

if __name__ == "__main__":
    # Example usage
    file_path = "c:/Users/dell/Desktop/Ecommerce_Project/Dataset/retail_store_inventory.csv"
    df = pd.read_csv(file_path)
    expected = ['Date', 'Store ID', 'Product ID', 'Category', 'Units Sold', 'Price']
    validate_schema(df, expected)
    report = data_quality_checks(df)
    print("Data Quality Report:", report)
