import os
import uuid
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI(title="AI Incident Triage & MLOps API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus Metrics
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
DATA_PATH = os.path.join("data", "incidents.csv")

# Load trained pipeline
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# Load historical incidents and precompute TF-IDF matrix for similarity lookup
historical_df = None
historical_tfidf = None

if model is not None and os.path.exists(DATA_PATH):
    try:
        df = pd.read_csv(DATA_PATH)
        df["short_description"] = df["short_description"].fillna("").astype(str)
        df["description"] = df["description"].fillna("").astype(str)
        df["category"] = df["category"].fillna("Unknown").astype(str)
        if "incident_id" not in df.columns:
            df["incident_id"] = [f"INC-{1000 + i}" for i in range(len(df))]
        
        df["combined_text"] = df["short_description"] + " " + df["description"]
        historical_df = df
        historical_tfidf = model.named_steps["tfidf"].transform(df["combined_text"])
    except Exception as e:
        print(f"Warning: Could not initialize historical similarity index: {e}")


class IncidentRequest(BaseModel):
    incident_id: str = Field(default="INC-101", description="Unique identifier for the incident")
    short_description: str = Field(..., description="Brief summary of the incident")
    description: str = Field(default="", description="Detailed technical description of the incident")
    top_k: int = Field(default=3, description="Number of similar historical incidents to return")


def get_related_incidents(query_text: str, top_k: int = 3):
    if historical_df is None or historical_tfidf is None or model is None:
        return []
    
    query_vec = model.named_steps["tfidf"].transform([query_text])
    similarities = cosine_similarity(query_vec, historical_tfidf).flatten()
    top_indices = similarities.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = float(similarities[idx])
        if score <= 0.0:
            continue
        row = historical_df.iloc[idx]
        item = {
            "incident_id": str(row.get("incident_id", f"INC-{idx}")),
            "short_description": str(row.get("short_description", "")),
            "description": str(row.get("description", "")),
            "category": str(row.get("category", "")),
            "similarity_score": round(score, 4),
        }
        # Include resolution or priority if present in your CSV
        if "resolution" in historical_df.columns and pd.notna(row["resolution"]):
            item["resolution"] = str(row["resolution"])
        if "priority" in historical_df.columns and pd.notna(row["priority"]):
            item["priority"] = str(row["priority"])
        results.append(item)

    return results


@app.get("/")
def read_root():
    return {"status": "healthy", "message": "IT Incident Triage FastAPI Backend is running!"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "historical_records": len(historical_df) if historical_df is not None else 0
    }


@app.get("/health/live")
def liveness_probe():
    return {"status": "alive"}


@app.get("/health/ready")
def readiness_probe():
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifact not loaded")
    return {
        "status": "ready",
        "model_loaded": True,
        "similarity_index_ready": historical_tfidf is not None
    }


@app.post("/predict")
def predict_incident(application: IncidentRequest):
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model file not found. Please run 'python train.py' first to generate models/incident_classifier.pkl."
        )

    combined_text = f"{application.short_description} {application.description}".strip()

    try:
        prediction = model.predict([combined_text])[0]
        probabilities = model.predict_proba([combined_text])[0]
        confidence = float(max(probabilities))
        related = get_related_incidents(combined_text, top_k=application.top_k)

        REQUEST_COUNT.labels(category=str(prediction)).inc()
        CONFIDENCE_GAUGE.set(confidence)

        return {
            "trace_id": str(uuid.uuid4()),
            "incident_id": application.incident_id,
            "predicted_category": str(prediction),
            "confidence": round(confidence, 4),
            "related_incidents": related,
            "similar_incidents": related,  # Alias in case your frontend checks either key
            "status": "Success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)