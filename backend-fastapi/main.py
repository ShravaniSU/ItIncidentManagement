import os
import joblib
import pandas as pd
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

app = FastAPI(title="AI Incident Triage & MLOps API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus Metrics Instrumentation
REQUEST_COUNT = Counter(
    "model_predictions_total",
    "Total number of incident prediction requests received",
    ["category"]
)
CONFIDENCE_GAUGE = Gauge(
    "model_prediction_confidence_score",
    "Confidence score of the latest incident prediction"
)

MODEL_PATH = os.path.join("models", "incident_classifier.pkl")

# Load trained model
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

class IncidentRequest(BaseModel):
    incident_id: str = Field(default="INC-101", description="Unique identifier for the incident")
    short_description: str = Field(..., description="Brief summary of the incident")
    description: str = Field(..., description="Detailed technical description of the incident")

@app.get("/")
def read_root():
    return {"status": "healthy", "message": "IT Incident Triage FastAPI Backend is running!"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "ai-incident-triage-api"}

@app.post("/predict")
def predict_incident(application: IncidentRequest):
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="Model file not found. Please run 'python train.py' first to generate models/incident_classifier.pkl."
        )

    combined_text = f"{application.short_description} {application.description}"

    try:
        prediction = model.predict([combined_text])[0]
        probabilities = model.predict_proba([combined_text])[0]
        confidence = float(max(probabilities))

        # Update Prometheus metrics
        REQUEST_COUNT.labels(category=str(prediction)).inc()
        CONFIDENCE_GAUGE.set(confidence)

        return {
            "trace_id": str(uuid.uuid4()),
            "incident_id": application.incident_id,
            "predicted_category": str(prediction),
            "confidence": round(confidence, 4),
            "status": "Success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.get("/metrics")
def metrics():
    """Exposes live Prometheus metrics for observability dashboards."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)