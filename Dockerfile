# Base Image: Python 3.10 Slim for smaller size
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (only if needed)
# Using --no-install-recommends to keep size down
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY deploy_requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r deploy_requirements.txt

# Copy application code
# Copy application code (Important: api.py needs src.models so we copy all of src)
COPY src/ ./src/
COPY config/ ./config/
# We might need data or models if they are local, but ideally models come from MLflow
# For this PoC, we'll copy the models dir if it exists, but ignore large files if using .dockerignore
# Creating models dir just in case
# Creating models dir just in case
RUN mkdir -p models data/processed

# Copy necessary data files (required for feature engineering during inference)
COPY data/processed/processed.csv ./data/processed/
COPY models/scaler.joblib ./models/

COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Create a non-root user for security with home directory (fixes permission errors)
RUN groupadd -r appuser && useradd -r -m -g appuser appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose ports
EXPOSE 8000 8501

ENTRYPOINT ["./entrypoint.sh"]
