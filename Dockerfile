# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements/environment dependencies and install
COPY environment.yml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir pandas==2.2.3 numpy==1.26.4 scikit-learn==1.5.2 mlflow==2.19.0 fastapi==0.115.6 uvicorn==0.34.0 pydantic==2.10.5 joblib==1.4.2 pyarrow==18.0.0 marshmallow==3.23.2

# Copy source code and trained model artifacts
COPY src/ /app/src/
COPY outputs/ /app/outputs/
COPY config.json /app/config.json

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Start Uvicorn server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]