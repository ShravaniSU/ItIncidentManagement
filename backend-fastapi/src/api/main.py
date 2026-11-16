import os
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

# Load environment variables from .env file at root of backend-fastapi
load_dotenv()

from src.models.inference import IncidentTriageEngine

# Initialize FastAPI app
app = FastAPI(
    title="AI Incident Triage & MLOps API",
    version="1.0.0",
    description="Production-grade MLOps service for automated incident triage and real-time observability."
)

# Enable CORS for frontend integration
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

# Global engine instance
engine = None

# Determine absolute path for model artifacts
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "outputs")

@app.on_event("startup")
def startup_event():
    global engine
    try:
        print("Initializing FastAPI backend startup...")
        
        # Pull Hugging Face config from .env variables
        hf_repo_id = os.getenv("HF_REPO_ID")
        hf_token = os.getenv("HF_TOKEN")
        
        if hf_repo_id:
            print(f"Downloading model artifacts from Hugging Face repo: {hf_repo_id}")
            os.makedirs(ARTIFACTS_DIR, exist_ok=True)
            snapshot_download(
                repo_id=hf_repo_id,
                local_dir=ARTIFACTS_DIR,
                repo_type="model",
                token=hf_token
            )
        
        # Initialize the triage engine
        engine = IncidentTriageEngine(artifacts_dir=ARTIFACTS_DIR)
        print("Incident Triage Engine loaded successfully!")
    except Exception as e:
        print(f"Error during startup model loading: {e}")
        engine = IncidentTriageEngine(artifacts_dir=ARTIFACTS_DIR)

# Request & Response Pydantic Schemas
class IncidentRequest(BaseModel):
    incident_id: str = Field(default="INC-101", description="Unique identifier for the incident")
    short_description: str = Field(..., description="Brief summary of the incident")
    description: str = Field(..., description="Detailed technical description of the incident")
    environment: str = Field(default="production", description="Deployment environment")

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "ai-incident-triage-api"}

@app.post("/api/v1/incidents/predict")
def predict_incident(payload: IncidentRequest):
    if engine is None:
        raise HTTPException(status_code=500, detail="Inference engine not initialized.")
    
    try:
        result = engine.predict(
            short_description=payload.short_description,
            description=payload.description
        )
        
        category = result["predicted_category"]
        confidence = result["category_confidence"]
        
        REQUEST_COUNT.labels(category=category).inc()
        CONFIDENCE_GAUGE.set(confidence)
        
        response_data = {
            "trace_id": str(uuid.uuid4()),
            "incident_id": payload.incident_id,
            **result,
            "model_version": "1.0.0"
        }
        return response_data
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)