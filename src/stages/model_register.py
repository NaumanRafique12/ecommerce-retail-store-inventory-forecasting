import mlflow
from mlflow.tracking import MlflowClient
import dagshub
import os

def get_production_model_wape(client, model_name):
    """Fetch WAPE of the current production model version."""
    try:
        versions = client.get_latest_versions(model_name, stages=["Production", "None"])
        if not versions:
            return float('inf')
        
        # Get the latest version (even if not in Production stage yet, for comparison)
        latest_version = versions[0]
        run_id = latest_version.run_id
        run_data = client.get_run(run_id).data
        wape = run_data.metrics.get("WAPE", float('inf'))
        print(f"Current Registered Model (v{latest_version.version}) WAPE: {wape}")
        return wape
    except Exception as e:
        print(f"No existing registered model found or error fetching WAPE: {e}")
        return float('inf')

def register_model():
    print("--- Model Registration Stage with Quality Gate (WAPE) ---")
    
    # Setup MLflow tracking with token-based auth
    dagshub_token = os.getenv("ABARK_MLOPS") or os.getenv("DAGSHUB_PAT")
    if dagshub_token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    
    mlflow.set_tracking_uri("https://dagshub.com/NaumanRafique12/ecommerce-retail-store-inventory-forecasting.mlflow")
    
    client = MlflowClient()
    model_name = "Ecom_Demand_Forecast_WAPE_Model"
    
    # 1. Find the best new run across all experiments
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id for exp in client.search_experiments()],
        filter_string="metrics.WAPE > 0",
        order_by=["metrics.WAPE ASC"],
        max_results=1
    )

    if not runs:
        print("No successful runs found with WAPE metrics in experiments.")
        return

    best_new_run = runs[0]
    new_wape = best_new_run.data.metrics.get("WAPE")
    run_id = best_new_run.info.run_id
    print(f"Best New Run found: {run_id} with WAPE: {new_wape}")

    # 2. Get current production model performance
    current_production_wape = get_production_model_wape(client, model_name)

    # 3. Quality Gate: Only register if performance is better
    if new_wape < current_production_wape:
        print(f"Quality Gate Passed: New WAPE ({new_wape}) is better than Current WAPE ({current_production_wape}).")
        
        model_type = best_new_run.data.tags.get("model_type", "model")
        if "XGBoost" in model_type:
            model_artifact_path = "xgboost_model"
        elif "LightGBM" in model_type:
            model_artifact_path = "lightgbm_model"
        else:
            model_artifact_path = "model"
            
        model_uri = f"runs:/{run_id}/{model_artifact_path}"
        
        # Register the model
        result = mlflow.register_model(model_uri, model_name)
        
        # Promote to Staging for validation
        client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage="Staging",
            archive_existing_versions=False
        )
        
        # Add metadata
        client.update_model_version(
            name=model_name,
            version=result.version,
            description=f"Improved model based on WAPE comparison. Type: {model_type}."
        )
        
        client.set_model_version_tag(name=model_name, version=result.version, key="WAPE", value=str(new_wape))
        client.set_model_version_tag(name=model_name, version=result.version, key="Selection_Metric", value="WAPE")
        
        print(f"Successfully registered model: {model_name}, Version: {result.version}")
    else:
        print(f"Quality Gate Failed: New WAPE ({new_wape}) is NOT better than Current WAPE ({current_production_wape}).")
        print("Skipping registration to maintain production stability.")

if __name__ == "__main__":
    register_model()