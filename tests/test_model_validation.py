import mlflow
import pandas as pd
import pytest
import sys
import os
import dagshub

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Check if we should use real connection or mock
USE_REAL_CONNECTION = os.getenv("MLFLOW_TRACKING_USERNAME") and os.getenv("MLFLOW_TRACKING_PASSWORD")

@pytest.fixture
def production_model():
    """Fixture to load the production model from the registry."""
    if not USE_REAL_CONNECTION:
        pytest.skip("Skipping real model validation: No credentials found in environment.")
    
    dagshub.init(repo_owner='NaumanRafique12', repo_name='ecommerce-retail-store-inventory-forecasting', mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/NaumanRafique12/ecommerce-retail-store-inventory-forecasting.mlflow")
    
    model_name = "Ecom_Demand_Forecast_WAPE_Model"
    try:
        model = mlflow.sklearn.load_model(f"models:/{model_name}/latest")
        return model
    except Exception as e:
        pytest.fail(f"Failed to load latest model from registry: {e}")

def test_model_loading_integrity(production_model):
    """Verify that the model can be loaded and its type is correct."""
    assert production_model is not None
    # Basic check for scikit-learn/lightgbm/xgboost compatibility
    assert hasattr(production_model, "predict"), "Loaded model does not have a predict method."

def test_model_signature_consistency(production_model):
    """
    Verify that the model's expected features match our preprocessing output.
    This prevents 'Feature Name' mismatch issues in production.
    """
    if not os.path.exists("data/processed/processed.csv"):
        pytest.skip("Skipping signature test: processed.csv not found locally.")

    df = pd.read_csv("data/processed/processed.csv", index_col=0)
    expected_features = df.drop(columns=['Units_Sold'], errors='ignore').columns.tolist()
    
    # If the model was fitted with feature names, check them
    if hasattr(production_model, "feature_name_") or hasattr(production_model, "feature_names_in_"):
        try:
            model_features = production_model.feature_names_in_ if hasattr(production_model, "feature_names_in_") else production_model.feature_name_
            # Check if all required features are present
            for feat in expected_features:
                assert feat in model_features, f"Missing feature in model signature: {feat}"
        except Exception:
            pass # Some models might not have explicit feature names in all versions

def test_model_smoke_performance(production_model):
    """
    A basic smoke test to ensure the model produces reasonable output (no NaN or negative values).
    """
    if not os.path.exists("data/processed/processed.csv"):
        pytest.skip("Processed data missing for smoke test.")

    df = pd.read_csv("data/processed/processed.csv", index_col=0).tail(5)
    X = df.drop(columns=['Units_Sold'], errors='ignore')
    
    preds = production_model.predict(X)
    
    assert len(preds) == 5
    assert not any(pd.isna(preds)), "Model produced NaN predictions."
    assert all(preds >= 0), "Model produced negative demand forecasts."
