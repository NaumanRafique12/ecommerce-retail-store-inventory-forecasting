import mlflow
from mlflow.tracking import MlflowClient
import dagshub
import pandas as pd
import os
import sys

# Add root to path for local imports if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

def validate_model(model, X_test, y_test):
    """Perform smoke test on the model."""
    try:
        preds = model.predict(X_test)
        if len(preds) > 0 and not any(pd.isna(preds)):
            return True
        return False
    except Exception as e:
        print(f"Validation failed: {e}")
        return False

def promote_model():
    print("--- Model Promotion Stage (Staging -> Production) ---")
    
    # Setup MLflow tracking with token-based auth
    dagshub_token = os.getenv("ABARK_MLOPS") or os.getenv("DAGSHUB_PAT")
    if dagshub_token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    
    mlflow.set_tracking_uri("https://dagshub.com/NaumanRafique12/ecommerce-retail-store-inventory-forecasting.mlflow")
    
    client = MlflowClient()
    model_name = "Ecom_Demand_Forecast_WAPE_Model"
    
    # 1. Fetch Staging and Production models
    staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
    production_versions = client.get_latest_versions(model_name, stages=["Production"])
    
    if not staging_versions:
        print("No model found in Staging. Skipping promotion logic.")
        return

    staging_v = staging_versions[0]
    print(f"Candidate Model found in Staging: Version {staging_v.version}")
    
    # 2. Get metrics for comparison
    staging_metrics = client.get_run(staging_v.run_id).data.metrics
    staging_wape = staging_metrics.get("WAPE", float('inf'))
    
    prod_wape = float('inf')
    if production_versions:
        prod_v = production_versions[0]
        prod_metrics = client.get_run(prod_v.run_id).data.metrics
        prod_wape = prod_metrics.get("WAPE", float('inf'))
        print(f"Current Production Model: Version {prod_v.version} with WAPE: {prod_wape}")
    else:
        print("No Production model found. Staging will be the first Production model.")

    # 3. Validation Logic (Smoke Test)
    print("Running smoke tests on Staging model...")
    if not os.path.exists("data/processed/processed.csv"):
        print("Processed data not found. Skipping detailed validation, relying on registry metrics.")
        validation_passed = True
    else:
        df = pd.read_csv("data/processed/processed.csv", index_col=0).tail(10)
        X = df.drop(columns=['Units_Sold'], errors='ignore')
        y = df['Units_Sold'] if 'Units_Sold' in df.columns else None
        
        try:
            model_uri = f"models:/{model_name}/Staging"
            model = mlflow.sklearn.load_model(model_uri)
            validation_passed = validate_model(model, X, y)
        except Exception as e:
            print(f"Error loading Staging model for validation: {e}")
            validation_passed = False

    # 4. Promotion Decision
    if validation_passed and staging_wape <= prod_wape:
        print(f"SUCCESS: Staging WAPE ({staging_wape}) is better than or equal to Production WAPE ({prod_wape}).")
        print(f"Promoting Version {staging_v.version} to Production...")
        
        client.transition_model_version_stage(
            name=model_name,
            version=staging_v.version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"Model Version {staging_v.version} is now in Production. Old versions archived.")
    else:
        if not validation_passed:
            print("FAILURE: Staging model failed validation smoke tests.")
        else:
            print(f"SKIPPED: Staging WAPE ({staging_wape}) is not better than Production WAPE ({prod_wape}).")
        
        # Optionally archive the failed staging version
        print(f"Archiving Staging Version {staging_v.version}...")
        client.transition_model_version_stage(
            name=model_name,
            version=staging_v.version,
            stage="Archived"
        )

if __name__ == "__main__":
    promote_model()
