import mlflow
from mlflow.tracking import MlflowClient
import dagshub
import os
import yaml

def register_model():
    print("--- Model Registration Stage (Based on WAPE) ---")
    
    # Initialize Dagshub & MLflow
    dagshub.init(repo_owner='NaumanRafique12', repo_name='ecommerce-retail-store-inventory-forecasting', mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/NaumanRafique12/ecommerce-retail-store-inventory-forecasting.mlflow")
    
    client = MlflowClient()
    
    # Search for runs with WAPE across experiments
    # (Using a broader search to catch runs from different tuning attempts if needed)
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id for exp in client.search_experiments()],
        filter_string="metrics.WAPE > 0",
        order_by=["metrics.WAPE ASC"], # Lower WAPE is better
        max_results=1
    )

    if not runs:
        print("No successful runs found with WAPE metrics.")
        return

    best_run = runs[0]
    run_id = best_run.info.run_id
    model_name = "Ecom_Demand_Forecast_WAPE_Model"
    
    # Identify which model was better
    model_type = best_run.data.tags.get("model_type", "model")
    # Check artifact path
    model_artifact_path = "lightgbm_model" if "LightGBM" in model_type else "random_forest_model"
    
    model_uri = f"runs:/{run_id}/{model_artifact_path}"
    
    print(f"Registering best model from run {run_id} ({model_type}) with WAPE: {best_run.data.metrics.get('WAPE')}")
    
    # Register the model
    result = mlflow.register_model(model_uri, model_name)
    
    # Add descriptions and tags
    client.update_model_version(
        name=model_name,
        version=result.version,
        description=f"Automated registration based on WAPE. Model Type: {model_type}."
    )
    
    client.set_model_version_tag(
        name=model_name,
        version=result.version,
        key="WAPE",
        value=str(best_run.data.metrics.get("WAPE"))
    )
    
    client.set_model_version_tag(
        name=model_name,
        version=result.version,
        key="Selection_Metric",
        value="WAPE"
    )

    print(f"Successfully registered model: {model_name}, Version: {result.version}")

if __name__ == "__main__":
    register_model()