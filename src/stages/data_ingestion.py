import yaml
import os
import sys

# Add src to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.ingestion.ingest_data import load_data, version_data
from src.ingestion.validate_data import validate_schema, data_quality_checks

def ingest_stage():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("--- Ingestion Stage ---")
    df = load_data(config['paths']['raw_data'])
    
    # Validation
    expected_cols = ['Date', 'Store ID', 'Product ID', 'Units Sold', 'Price']
    validate_schema(df, expected_cols)
    quality_report = data_quality_checks(df)
    print(f"Validation complete. Quality Report: {quality_report}")

    # Versioning/Snapshot is handled within ingest_data, but for DVC 
    # we usually want to save to a specific output file.
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/raw.csv", index=False)
    print("Raw data saved to data/raw/raw.csv")

if __name__ == "__main__":
    ingest_stage()
