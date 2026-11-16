import os
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import dagshub
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()

os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI", "")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_TOKEN", "")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN", "")

mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(mlflow_tracking_uri)

print("Initializing DagsHub MLflow connection...")
# dagshub.init(repo_owner=..., repo_name=..., mlflow=True)
mlflow.set_experiment("it_incident_triage_dagshub")

# Ensure required directories exist
os.makedirs("models", exist_ok=True)
data_path = os.path.join("data", "incidents.csv")

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at '{data_path}'. Please place historical_incidents.csv in the data/ folder.")

print(f"Loading dataset from: {data_path}")
df = pd.read_csv(data_path)

# 2. Data Preprocessing & Cleaning
# Drop primary identifier if present
if "incident_id" in df.columns:
    df = df.drop(columns=["incident_id"])

# Handle missing values in text fields
df["short_description"] = df["short_description"].fillna("").astype(str)
df["description"] = df["description"].fillna("").astype(str)
df["category"] = df["category"].fillna("Infrastructure").astype(str)

# Combine text features for classification
df["combined_text"] = df["short_description"] + " " + df["description"]

X = df["combined_text"]
y = df["category"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Training & Experiment Tracking (DagsHub/MLflow + Hugging Face)
HF_REPO_ID = os.getenv("HF_REPO_ID")  # e.g., "your-username/it-incident-triage-models"
HF_TOKEN = os.getenv("HF_TOKEN")

with mlflow.start_run() as run:
    n_estimators = 100
    max_depth = 10
    
    # Build text pipeline (TF-IDF + Random Forest Classifier)
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
        ('classifier', RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42))
    ])
    
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    # Using weighted average for multi-class metrics
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"Model Training Completed. Accuracy: {acc:.4f}")

    # Log parameters & metrics to DagsHub MLflow
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Save local pickle file
    model_path = os.path.join("models", "incident_classifier.pkl")
    joblib.dump(model, model_path)
    
    # Also save a copy in the parent/outputs directory if needed by main.py
    os.makedirs("../outputs", exist_ok=True)
    joblib.dump(model, os.path.join("../outputs", "incident_classifier.pkl"))
    df.to_csv(os.path.join("../outputs", "historical_incidents.csv"), index=False)

    # Log model artifact to MLflow/DagsHub with trusted types configuration
    # Log model artifact to MLflow/DagsHub with trusted types configuration
    # Log model artifact to MLflow/DagsHub
    
    mlflow.sklearn.log_model(
        model, 
        artifact_path="incident_classifier"
    )
    print("Model successfully logged to MLflow and DagsHub!")

    # Upload to Hugging Face Hub (if configured)
    if HF_REPO_ID and HF_TOKEN:
        print(f"Uploading model to Hugging Face Hub repository: {HF_REPO_ID}...")
        try:
            local_artifact_dir = "hf_model_artifact"
            os.makedirs(local_artifact_dir, exist_ok=True)
            joblib.dump(model, os.path.join(local_artifact_dir, "incident_classifier.pkl"))
            df.to_csv(os.path.join(local_artifact_dir, "historical_incidents.csv"), index=False)
            
            api = HfApi()
            api.create_repo(repo_id=HF_REPO_ID, token=HF_TOKEN, exist_ok=True, repo_type="model")
            api.upload_folder(
                folder_path=local_artifact_dir,
                repo_id=HF_REPO_ID,
                repo_type="model",
                token=HF_TOKEN
            )
            print("Model successfully uploaded to Hugging Face Hub!")
        except Exception as e:
            print(f"Failed to upload to Hugging Face: {str(e)}")
    else:
        print("HF_REPO_ID or HF_TOKEN not provided. Skipping Hugging Face upload.")

    print(f"Experiment successfully logged to DagsHub! Run ID: {run.info.run_id}")