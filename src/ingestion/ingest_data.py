import pandas as pd
import os
import shutil
from datetime import datetime

def load_data(file_path):
    """Load historical sales data from CSV."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)

def version_data(df, snapshot_dir):
    """Assign version ID and save snapshot of raw data."""
    version_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(snapshot_dir, exist_ok=True)
    snapshot_path = os.path.join(snapshot_dir, f"raw_data_{version_id}.csv")
    df.to_csv(snapshot_path, index=False)
    print(f"Data versioned: {version_id} -> {snapshot_path}")
    return version_id, snapshot_path

if __name__ == "__main__":
    # Example usage for testing
    file_path = "c:/Users/dell/Desktop/Ecommerce_Project/Dataset/retail_store_inventory.csv"
    snapshot_dir = "c:/Users/dell/Desktop/Ecommerce_Project/data/raw"
    data = load_data(file_path)
    version_data(data, snapshot_dir)
