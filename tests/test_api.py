import pytest
from fastapi.testclient import TestClient
import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Check if we should use real connection or mock
USE_REAL_CONNECTION = os.getenv("MLFLOW_TRACKING_USERNAME") and os.getenv("MLFLOW_TRACKING_PASSWORD")

if not USE_REAL_CONNECTION:
    print("CI: Credentials not found, using Mocking for API tests.")
    mock_mlflow = MagicMock()
    mock_dagshub = MagicMock()
    with patch.dict('sys.modules', {'mlflow': mock_mlflow, 'mlflow.sklearn': mock_mlflow.sklearn, 'dagshub': mock_dagshub}):
        from src.deployment.api import app
else:
    print("CI: Credentials found, testing with REAL Dagshub connection.")
    from src.deployment.api import app

client = TestClient(app)

def test_health_check():
    """Test the health check endpoint."""
    if not USE_REAL_CONNECTION:
        # Mock the global MODEL in api.py if we are in mock mode
        with patch('src.deployment.api.MODEL', MagicMock()):
            response = client.get("/")
            assert response.status_code == 200
            assert response.json()["status"] == "operational"
    else:
        # If real, just check if it's operational (even if model is none)
        response = client.get("/")
        assert response.status_code == 200
        assert "status" in response.json()

def test_predict_endpoint_params():
    """Test the predict endpoint with invalid parameters."""
    response = client.get("/predict?horizon=10") # 10 is not allowed
    assert response.status_code == 400
    assert "Horizon must be 7, 14, or 30" in response.json()["detail"]

def test_predict_service_unavailable_logic():
    """Verify 503 if model is missing (Only in Mock mode or if registry is empty)."""
    # This specifically tests the logic of the endpoint
    with patch('src.deployment.api.MODEL', None):
        response = client.get("/predict?horizon=7")
        assert response.status_code == 503
        assert "Model is currently unavailable" in response.json()["detail"]
