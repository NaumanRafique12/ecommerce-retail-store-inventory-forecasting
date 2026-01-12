#!/bin/bash
set -e

# Start FastAPI in the background
echo "Starting FastAPI Backend..."
python src/deployment/api.py &

# Start Streamlit in the foreground
echo "Starting Streamlit Dashboard..."
python -m streamlit run src/deployment/app.py --server.port 8501 --server.address 0.0.0.0
