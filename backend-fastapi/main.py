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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "incident_classifier.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "incidents.csv")

# Load trained model
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# Load historical incidents and precompute TF-IDF matrix for similarity search
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
        default_prio = "P1" if environment.lower() == "production" else "P2"
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

        for col in ["resolution_time_minutes", "resolution_minutes", "mttr_minutes", "resolution_time", "duration_minutes"]:
            if col in historical_df.columns and pd.notna(row[col]):
                try:
                    res_times.append(float(row[col]))
                except ValueError:
                    pass

        results.append(item)

    if priorities:
        predicted_priority = pd.Series(priorities).mode()[0]
        priority_conf = round(priorities.count(predicted_priority) / len(priorities), 4)
    else:
        predicted_priority = "P1" if environment.lower() == "production" else "P2"
        priority_conf = 0.82

    est_minutes = int(round(sum(res_times) / len(res_times))) if res_times else (45 if "P1" in predicted_priority else 90)

    return results, predicted_priority, priority_conf, est_minutes


def get_troubleshooting_steps(category: str, related_incidents: list) -> list[str]:
    steps = []
    for inc in related_incidents:
        res = inc.get("resolution", "").strip()
        if res and res not in steps:
            steps.append(res)

    if steps:
        return steps[:4]

    runbooks = {
        "Performance": [
            "Check CPU, memory, and I/O utilization metrics on affected nodes or services.",
            "Inspect upstream API gateway latency, connection pool saturation, and slow queries.",
            "Scale replicas horizontally or clear stale cache entries if resource thresholds exceed 85%."
        ],
        "Storage": [
            "Inspect disk usage (df -h) and inode exhaustion across persistent volumes.",
            "Rotate or archive uncompressed application logs and orphaned temporary files.",
            "Expand storage volume capacity or detach unused snapshots."
        ],
        "Network": [
            "Verify DNS resolution, ingress routing rules, and firewall security group policies.",
            "Run packet loss and latency diagnostics between source and target endpoints.",
            "Restart affected load balancer health checks or failover to secondary route."
        ],
        "Database": [
            "Check active database locks, deadlocks, and long-running queries.",
            "Verify connection pool limits and replication lag across read replicas.",
            "Optimize missing indexes or kill blocking sessions after validation."
        ],
        "Security": [
            "Isolate affected credentials, tokens, or compromised service accounts immediately.",
            "Audit access logs for anomalous IP addresses or unauthorized privilege escalations.",
            "Rotate secrets and enforce updated WAF/IAM policies."
        ]
    }
    return runbooks.get(category, [
        "Inspect recent deployment changes and service logs around the incident timestamp.",
        "Verify health check endpoints and downstream service dependencies.",
        "Escalate to the on-call SME with trace ID and diagnostic logs if issue persists."
    ])


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
        recommendations = get_troubleshooting_steps(prediction, related)

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
            "estimated_resolution_time_minutes": est_minutes,
            "estimated_resolution_minutes": est_minutes,
            "troubleshooting_recommendations": recommendations,
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