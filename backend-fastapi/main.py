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

REQUEST_COUNT = Counter(
    "model_predictions_total",
    "Total number of incident prediction requests received",
    ["category"]
)
CONFIDENCE_GAUGE = Gauge(
    "model_prediction_confidence_score",
    "Confidence score of the latest incident prediction"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "incident_classifier.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "incidents.csv")

model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

historical_df = None
historical_tfidf = None
startup_warning = None

if model is not None and os.path.exists(DATA_PATH):
    try:
        df = pd.read_csv(DATA_PATH)
        text_cols = [c for c in ["short_description", "description", "summary", "title", "details", "text"] if c in df.columns]
        if not text_cols:
            text_cols = df.select_dtypes(include=["object"]).columns.tolist()

        df["combined_text"] = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
        if "incident_id" not in df.columns:
            df["incident_id"] = [f"INC-{1000 + i}" for i in range(len(df))]

        historical_df = df
        historical_tfidf = model.named_steps["tfidf"].transform(df["combined_text"])
    except Exception as e:
        startup_warning = str(e)
elif not os.path.exists(DATA_PATH):
    startup_warning = f"Dataset not found at {DATA_PATH}"


class IncidentRequest(BaseModel):
    incident_id: str = Field(default="INC-101")
    short_description: str = Field(...)
    description: str = Field(default="")
    environment: str = Field(default="Production")
    top_k: int = Field(default=3)


def analyze_related_and_metadata(query_text: str, environment: str, top_k: int = 3):
    if historical_df is None or historical_tfidf is None or model is None:
        default_prio = "P1 - Critical" if environment.lower() == "production" else "P2 - High"
        return [], default_prio, 0.85, 45

    query_vec = model.named_steps["tfidf"].transform([query_text])
    similarities = cosine_similarity(query_vec, historical_tfidf).flatten()
    top_indices = similarities.argsort()[::-1][:top_k]

    results = []
    priorities = []
    res_times = []

    for idx in top_indices:
        score = float(similarities[idx])
        row = historical_df.iloc[idx]
        item = {
            "incident_id": str(row.get("incident_id", f"INC-{idx}")),
            "short_description": str(row.get("short_description", row.get("summary", row.get("title", "")))),
            "description": str(row.get("description", row.get("details", ""))),
            "category": str(row.get("category", "")),
            "similarity_score": round(score, 4),
        }
        if "resolution" in historical_df.columns and pd.notna(row["resolution"]):
            item["resolution"] = str(row["resolution"])
        if "priority" in historical_df.columns and pd.notna(row["priority"]):
            prio_val = str(row["priority"])
            item["priority"] = prio_val
            priorities.append(prio_val)

        # Check common resolution time column names in incidents.csv
        for col in ["resolution_time_minutes", "resolution_minutes", "mttr_minutes", "resolution_time", "duration_minutes"]:
            if col in historical_df.columns and pd.notna(row[col]):
                try:
                    res_times.append(float(row[col]))
                except ValueError:
                    pass

        results.append(item)

    # Derive priority from top matching historical incidents (or environment fallback)
    if priorities:
        predicted_priority = pd.Series(priorities).mode()[0]
        priority_conf = round(priorities.count(predicted_priority) / len(priorities), 4)
    else:
        predicted_priority = "P1 - High" if environment.lower() == "production" else "P2 - Medium"
        priority_conf = 0.82

    # Derive estimated resolution time in minutes
    est_minutes = int(round(sum(res_times) / len(res_times))) if res_times else (45 if "P1" in predicted_priority else 90)

    return results, predicted_priority, priority_conf, est_minutes


@app.get("/")
def read_root():
    return {"status": "healthy", "message": "IT Incident Triage FastAPI Backend is running!"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "historical_records": len(historical_df) if historical_df is not None else 0,
        "warning": startup_warning
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
        "similarity_index_ready": historical_tfidf is not None,
        "historical_records": len(historical_df) if historical_df is not None else 0
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
        prediction = str(model.predict([combined_text])[0])
        probabilities = model.predict_proba([combined_text])[0]
        cat_confidence = round(float(max(probabilities)), 4)

        related, pred_priority, prio_confidence, est_minutes = analyze_related_and_metadata(
            combined_text, application.environment, top_k=application.top_k
        )

        REQUEST_COUNT.labels(category=prediction).inc()
        CONFIDENCE_GAUGE.set(cat_confidence)

        return {
            "trace_id": str(uuid.uuid4()),
            "incident_id": application.incident_id,
            "predicted_category": prediction,
            "category_confidence": cat_confidence,
            "confidence": cat_confidence,
            "predicted_priority": pred_priority,
            "priority": pred_priority,
            "priority_confidence": prio_confidence,
            "estimated_resolution_minutes": est_minutes,
            "resolution_time_minutes": est_minutes,
            "predicted_resolution_time": est_minutes,
            "estimated_resolution_time": est_minutes,
            "estimated_resolution": est_minutes,
            "resolution_time": est_minutes,
            "mttr": est_minutes,
            "related_incidents": related,
            "similar_incidents": related,
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