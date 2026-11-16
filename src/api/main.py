import time
import uuid
import os
import sys
import json
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
sys.path.append(project_root)

from src.models.inference import IncidentTriageEngine

app = FastAPI(
    title="AI-Powered IT Incident Triage & Resolution API",
    description="Production-grade MLOps service for automated IT incident categorization, priority assignment, and resolution recommendation.",
    version="1.0.0"
)

# The browser sends an OPTIONS preflight before the JSON POST when the UI and
# API use different origins. Configure deployed UI origins with CORS_ORIGINS.
cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Initialize Inference Engine on startup
artifacts_path = os.path.join(project_root, "outputs")
engine = None

@app.on_event("startup")
def startup_event():
    global engine
    try:
        engine = IncidentTriageEngine(artifacts_dir=artifacts_path)
    except Exception as e:
        print(f"Failed to load inference engine: {e}")
        raise e

# --- Prometheus Metrics for AI Observability ---
PREDICTION_COUNTER = Counter(
    "model_predictions_total", 
    "Total predictions made by category and priority", 
    ["predicted_category", "predicted_priority"]
)

CONFIDENCE_GAUGE = Gauge(
    "model_prediction_confidence_score", 
    "Confidence score of the latest model predictions"
)

# --- Request / Response Schemas ---
class IncidentRequest(BaseModel):
    incident_id: Optional[str] = Field(default="INC-UNKNOWN", description="Unique incident identifier")
    short_description: str = Field(..., description="Brief summary of the incident")
    description: str = Field(..., description="Detailed description of the incident")
    environment: Optional[str] = Field(default="production", description="Environment where incident occurred")

class SimilarIncident(BaseModel):
    incident_id: str
    short_description: str
    category: str
    similarity_score: float
    resolution_notes: str

class IncidentPredictionResponse(BaseModel):
    trace_id: str
    incident_id: str
    predicted_category: str
    category_confidence: float
    predicted_priority: str
    priority_confidence: float
    estimated_resolution_time_minutes: float
    similar_incidents: List[SimilarIncident]
    troubleshooting_recommendations: List[str]
    model_version: str = "1.0.0"

# --- Helper for Logging Inferences for Drift Detection ---
def log_inference_to_disk(payload: IncidentRequest, prediction: dict):
    """Appends inference requests and outputs to a local JSONL log file for drift analysis."""
    log_dir = os.path.join(project_root, "monitoring")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "inference_logs.jsonl")
    
    log_record = {
        "timestamp": datetime.utcnow().isoformat(),
        "short_description": payload.short_description,
        "description": payload.description,
        "environment": payload.environment,
        "predicted_category": prediction["predicted_category"],
        "category_confidence": prediction["category_confidence"],
        "predicted_priority": prediction["predicted_priority"],
        "priority_confidence": prediction["priority_confidence"]
    }
    
    try:
        with open(log_file, "a") as f:
            f.write(json.dumps(log_record) + "\n")
    except Exception as e:
        print(f"Failed to write inference log: {e}")

# --- Endpoints ---
@app.get("/health/live", tags=["Health"])
def liveness_probe():
    """Kubernetes liveness probe endpoint."""
    return {"status": "alive"}

@app.get("/health/ready", tags=["Health"])
def readiness_probe():
    """Kubernetes readiness probe endpoint verifying model readiness."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not ready")
    return {"status": "ready"}

@app.get("/version", tags=["Info"])
def get_version():
    """Returns service and model versioning details."""
    return {
        "service": "it-incident-triage-api",
        "version": "1.0.0",
        "mlflow_model_name": "it-incident-triage-bundle"
    }

@app.get("/metrics", tags=["Observability"])
def metrics():
    """Exposes real-time Prometheus metrics for scraping."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/api/v1/incidents/predict", response_model=IncidentPredictionResponse, tags=["Prediction"])
def predict_incident(payload: IncidentRequest):
    """Predicts category, priority, resolution time, and retrieves similar historical incidents."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Inference engine is not initialized.")
    
    trace_id = str(uuid.uuid4())
    
    try:
        prediction = engine.predict(
            short_description=payload.short_description,
            description=payload.description
        )
        
        # --- Update Prometheus Metrics for Production Observability ---
        cat = prediction["predicted_category"]
        pri = prediction["predicted_priority"]
        conf = prediction["category_confidence"]
        
        PREDICTION_COUNTER.labels(predicted_category=cat, predicted_priority=pri).inc()
        CONFIDENCE_GAUGE.set(conf)
        # -------------------------------------------------------------
        
        # Log the request and output for Phase 2 Data/Model Drift monitoring
        log_inference_to_disk(payload, prediction)
        
        return IncidentPredictionResponse(
            trace_id=trace_id,
            incident_id=payload.incident_id,
            predicted_category=prediction["predicted_category"],
            category_confidence=prediction["category_confidence"],
            predicted_priority=prediction["predicted_priority"],
            priority_confidence=prediction["priority_confidence"],
            estimated_resolution_time_minutes=prediction["estimated_resolution_time_minutes"],
            similar_incidents=prediction["similar_incidents"],
            troubleshooting_recommendations=prediction["troubleshooting_recommendations"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
