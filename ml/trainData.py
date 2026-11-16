import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_absolute_error, root_mean_squared_error
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

def load_data(csv_path: str) -> pd.DataFrame:
    """Loads the incidents dataset."""
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    return df

def train_and_evaluate(df: pd.DataFrame, output_dir: str = "outputs"):
    """Trains classification and regression models, logs to MLflow, and saves artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Feature engineering: combine short description and full description for text mining
    df['text_content'] = df['short_description'].fillna('') + " " + df['description'].fillna('')
    
    X_text = df['text_content']
    y_category = df['category']
    y_priority = df['priority']
    y_time = df['resolution_time_minutes']
    
    # Train-test split
    X_train, X_test, cat_train, cat_test, pri_train, pri_test, time_train, time_test = train_test_split(
        X_text, y_category, y_priority, y_time, test_size=0.2, random_state=42
    )
    
    # Setup MLflow experiment
    mlflow.set_experiment("it-incident-triage-experiment")
    
    with mlflow.start_run(run_name="baseline-sklearn-models") as run:
        run_id = run.info.run_id
        print(f"\n[MLflow] Active Run ID: {run_id}")

        print("\n--- Training TF-IDF Vectorizer ---")
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        joblib.dump(vectorizer, os.path.join(output_dir, "tfidf_vectorizer.pkl"))
        
        # 1. Category Classification
        print("Training Category Classification Model...")
        cat_model = LogisticRegression(max_iter=1000, random_state=42)
        cat_model.fit(X_train_vec, cat_train)
        cat_preds = cat_model.predict(X_test_vec)
        
        cat_acc = accuracy_score(cat_test, cat_preds)
        _, _, cat_f1, _ = precision_recall_fscore_support(cat_test, cat_preds, average='macro', zero_division=0)
        
        print(f"Category Model -> Accuracy: {cat_acc:.4f} | Macro F1: {cat_f1:.4f}")
        mlflow.log_metric("category_accuracy", cat_acc)
        mlflow.log_metric("category_macro_f1", cat_f1)
        joblib.dump(cat_model, os.path.join(output_dir, "category_model.pkl"))
        
        # 2. Priority Classification
        print("Training Priority Classification Model...")
        pri_model = RandomForestClassifier(n_estimators=100, random_state=42)
        pri_model.fit(X_train_vec, pri_train)
        pri_preds = pri_model.predict(X_test_vec)
        
        pri_acc = accuracy_score(pri_test, pri_preds)
        _, _, pri_f1, _ = precision_recall_fscore_support(pri_test, pri_preds, average='weighted', zero_division=0)
        
        print(f"Priority Model -> Accuracy: {pri_acc:.4f} | Weighted F1: {pri_f1:.4f}")
        mlflow.log_metric("priority_accuracy", pri_acc)
        mlflow.log_metric("priority_weighted_f1", pri_f1)
        joblib.dump(pri_model, os.path.join(output_dir, "priority_model.pkl"))
        
        # 3. Resolution Time Regression
        print("Training Resolution Time Regression Model...")
        time_model = Ridge(alpha=1.0)
        time_model.fit(X_train_vec, time_train)
        time_preds = time_model.predict(X_test_vec)
        
        mae = mean_absolute_error(time_test, time_preds)
        rmse = root_mean_squared_error(time_test, time_preds)
        
        print(f"Resolution Time Model -> MAE: {mae:.2f} mins | RMSE: {rmse:.2f} mins")
        mlflow.log_metric("resolution_time_mae", mae)
        mlflow.log_metric("resolution_time_rmse", rmse)
        joblib.dump(time_model, os.path.join(output_dir, "resolution_time_model.pkl"))
        
        # Save historical incident index for similarity lookup
        history_df = df[['incident_id', 'short_description', 'category', 'resolution_notes']].copy()
        history_df.to_parquet(os.path.join(output_dir, "historical_incidents.parquet"))
        
        # Log entire outputs directory as an MLflow artifact folder
        mlflow.log_artifacts(output_dir, artifact_path="model_bundle")
        print("\nTraining and MLflow artifact logging completed successfully!")
    
    return output_dir, run_id

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    csv_file_path = os.path.join(project_root, "data", "raw", "incidents.csv")
    
    df_incidents = load_data(csv_file_path)
    outputs_path, active_run_id = train_and_evaluate(df_incidents, output_dir=os.path.join(project_root, "outputs"))
    
    # Register Model in Azure ML Workspace using local folder path (Custom Model Asset)
    print("\n--- Registering Model Bundle in Azure ML ---")
    try:
        credential = DefaultAzureCredential()
        config_path = os.path.join(project_root, "config.json")
        ml_client = MLClient.from_config(credential=credential, path=config_path)
        
        model_name = "it-incident-triage-bundle"
        
        # Register the local outputs folder as a custom model asset in Azure ML
        ml_model = Model(
            path=outputs_path,
            type=AssetTypes.CUSTOM_MODEL,
            name=model_name,
            description="Bundle containing category classifier, priority classifier, resolution time regressor, TF-IDF vectorizer, and historical incident records.",
            version="1"
        )
        
        registered_model = ml_client.models.create_or_update(ml_model)
        print(f"Azure ML Model registered successfully! Name: {registered_model.name}, Version: {registered_model.version}")
        
    except Exception as e:
        print(f"\n[Azure Registration Error]: {e}")