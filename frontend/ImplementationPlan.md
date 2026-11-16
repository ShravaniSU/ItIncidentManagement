Act as an expert frontend engineer. Build a sleek, professional, production-grade React (Vite + JavaScript + Tailwind CSS + Lucide React + Recharts) dashboard for an AI-Powered IT Incident Triage & MLOps System. 

The dashboard must feature a single, cohesive modern UI theme (clean light enterprise style) with two primary top-navigation tabs:

---
### TAB 1: Incident Triage Portal
- An input form with fields for:
  - Short Description (text input, e.g., "High latency on API gateway")
  - Detailed Description (textarea, e.g., "Gateway returning 504 timeouts under load")
  - Priority / Environment (optional dropdowns if needed)
- A "Run Triage Prediction" submit button with loading states.
- When submitted, it must make a POST request to the FastAPI backend:
  - Endpoint: `POST /api/v1/incidents/predict`
  - Request Payload JSON structure:
    {
      "incident_id": "INC-101",
      "short_description": "<user_input>",
      "description": "<user_input>",
      "environment": "production"
    }
  - Response JSON structure expected from backend:
    {
      "trace_id": "uuid-string",
      "incident_id": "INC-101",
      "predicted_category": "Database / Network / Infrastructure / Application / Security",
      "category_confidence": 0.92,
      "predicted_priority": "high",
      "priority_confidence": 0.88,
      "estimated_resolution_time_minutes": 45.0,
      "similar_incidents": [
        {
          "incident_id": "INC-042",
          "short_description": "Connection pool exhaustion",
          "category": "Database",
          "similarity_score": 0.89,
          "resolution_notes": "Restarted connection pool and scaled worker threads."
        }
      ],
      "troubleshooting_recommendations": [
        "Check database connection pool limits.",
        "Review slow query logs for bottlenecks."
      ],
      "model_version": "1.0.0"
    }
- A structured results view displaying the prediction badges, confidence scores, estimated resolution time, troubleshooting steps, and a card grid of similar historical incidents.

---
### TAB 2: Observability & Drift Dashboard
- A live MLOps monitoring view that polls the FastAPI backend `/metrics` endpoint (`GET /metrics`) or fetches live telemetry.
- Real-time stat cards displaying:
  - Total prediction requests (`model_predictions_total`).
  - Current model confidence score (`model_prediction_confidence_score`).
- A live visual chart (using Recharts) mapping prediction category distributions and confidence trends in real time as triage requests are processed.
- An auto-refresh toggle (polling every 3 seconds) and a manual "Refresh Metrics" button.

---
Requirements:
- Include clean error handling (e.g., if the FastAPI backend at `http://localhost:8000` is offline, show a polite warning banner).
- Provide clean, modular React component code with all necessary imports.