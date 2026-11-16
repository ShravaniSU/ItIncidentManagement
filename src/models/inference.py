import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class IncidentTriageEngine:
    def __init__(self, artifacts_dir: str = "outputs"):
        print(f"Loading model artifacts from {artifacts_dir}...")
        self.vectorizer = joblib.load(os.path.join(artifacts_dir, "tfidf_vectorizer.pkl"))
        self.cat_model = joblib.load(os.path.join(artifacts_dir, "category_model.pkl"))
        self.pri_model = joblib.load(os.path.join(artifacts_dir, "priority_model.pkl"))
        self.time_model = joblib.load(os.path.join(artifacts_dir, "resolution_time_model.pkl"))
        
        # Load historical incidents for similarity search
        history_path = os.path.join(artifacts_dir, "historical_incidents.parquet")
        if os.path.exists(history_path):
            self.history_df = pd.read_parquet(history_path)
            # Precompute TF-IDF matrix for history texts for fast cosine similarity
            history_texts = self.history_df['short_description'].fillna('')
            self.history_matrix = self.vectorizer.transform(history_texts)
        else:
            self.history_df = pd.DataFrame()
            self.history_matrix = None
        print("Inference engine initialized successfully.")

    def predict(self, short_description: str, description: str) -> dict:
        text_input = [f"{short_description} {description}"]
        text_vec = self.vectorizer.transform(text_input)
        
        # 1. Category Prediction
        category = self.cat_model.predict(text_vec)[0]
        cat_proba = np.max(self.cat_model.predict_proba(text_vec))
        
        # 2. Priority Prediction
        priority = self.pri_model.predict(text_vec)[0]
        pri_proba = np.max(self.pri_model.predict_proba(text_vec))
        
        # 3. Resolution Time Prediction
        resolution_time = float(self.time_model.predict(text_vec)[0])
        resolution_time = max(5.0, round(resolution_time, 2)) # Ensure positive minimum time
        
        # 4. Similar Incident Retrieval
        similar_incidents = []
        if self.history_matrix is not None and not self.history_df.empty:
            sim_scores = cosine_similarity(text_vec, self.history_matrix).flatten()
            top_indices = sim_scores.argsort()[::-1][:3] # Top 3 similar
            
            for idx in top_indices:
                row = self.history_df.iloc[idx]
                similar_incidents.append({
                    "incident_id": str(row['incident_id']),
                    "short_description": str(row['short_description']),
                    "category": str(row['category']),
                    "similarity_score": round(float(sim_scores[idx]), 4),
                    "resolution_notes": str(row['resolution_notes']) if pd.notna(row['resolution_notes']) else "None recorded"
                })

        # 5. Deterministic Troubleshooting Recommendations
        recommendations = self._generate_recommendations(category, priority)

        return {
            "predicted_category": category,
            "category_confidence": round(float(cat_proba), 4),
            "predicted_priority": priority,
            "priority_confidence": round(float(pri_proba), 4),
            "estimated_resolution_time_minutes": resolution_time,
            "similar_incidents": similar_incidents,
            "troubleshooting_recommendations": recommendations
        }

    def _generate_recommendations(self, category: str, priority: str) -> list:
        """Generates deterministic troubleshooting actions based on category."""
        recs = {
            "Application": [
                "Check application error logs in /var/log/app/error.log for stack traces.",
                "Verify recent deployments or configuration changes made in the last 2 hours.",
                "Restart application service pool if memory leak or deadlock is suspected."
            ],
            "Database": [
                "Inspect active database connections and long-running queries via pg_stat_activity / processlist.",
                "Check disk I/O, CPU utilization, and connection pool exhaustion on database instance.",
                "Verify recent migration scripts or index locks."
            ],
            "Network": [
                "Perform traceroute and DNS lookup to verify routing and connectivity.",
                "Check firewall rules, security groups, and NAT gateway metrics.",
                "Verify load balancer target health and interface status."
            ],
            "Authentication": [
                "Check Identity Provider (IdP) service health and OAuth/SAML token endpoints.",
                "Verify LDAP/Active Directory connectivity and certificate expiration dates.",
                "Inspect user authentication logs for brute-force patterns or expired credentials."
            ],
            "Infrastructure": [
                "Check node resource utilization (CPU, memory, disk pressure) via monitoring tools.",
                "Verify cloud provider status page for regional outages or hardware degradation.",
                "Inspect systemd service statuses and kernel ring buffer (dmesg)."
            ]
        }
        default_recs = [
            "Review recent system and application logs for anomalies.",
            "Verify service dependencies and downstream API health.",
            "Escalate to the appropriate secondary on-call rotation if unresolved within SLA."
        ]
        return recs.get(category, default_recs)