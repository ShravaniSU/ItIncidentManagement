import os
import pandas as pd
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

def validate_and_inspect_data(csv_path: str) -> pd.DataFrame:
    """Loads and validates the raw incidents CSV dataset."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at absolute path: {os.path.abspath(csv_path)}")
    
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    print(f"Columns present: {list(df.columns)}")
    
    # Check for missing values in critical columns
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        print("Missing values summary:\n", missing_counts[missing_counts > 0])
    else:
        print("No missing values detected in dataset.")
        
    return df

def register_azure_ml_data_asset(ml_client: MLClient, local_path: str) -> Data:
    """Registers the local CSV file as a versioned Data Asset in Azure ML."""
    asset_name = "it-incidents-raw"
    print(f"Preparing to register Data Asset '{asset_name}' in Azure ML...")
    
    my_data = Data(
        path=local_path,
        type=AssetTypes.URI_FILE,
        description="Synthetic IT incident triage dataset containing 10,000 enterprise records.",
        name=asset_name,
        version="1"
    )
    
    data_asset = ml_client.data.create_or_update(my_data)
    print(f"Data Asset registered successfully! Name: {data_asset.name}, Version: {data_asset.version}")
    return data_asset

if __name__ == "__main__":
    # Dynamically resolve path relative to project root (AI-IncidentManagement/)
    current_dir = os.path.dirname(os.path.abspath(__file__))  # points to ml/
    project_root = os.path.dirname(current_dir)               # points to AI-IncidentManagement/
    csv_file_path = os.path.join(project_root, "data", "raw", "incidents.csv")
    
    # 1. Local Validation
    df_incidents = validate_and_inspect_data(csv_file_path)
    
    # 2. Azure ML Connection via config.json & Data Asset Registration
    print("\n--- Connecting to Azure ML via config.json ---")
    try:
        credential = DefaultAzureCredential()
        
        # Ensure config.json is also looked up from project root
        config_path = os.path.join(project_root, "config.json")
        if os.path.exists(config_path):
            ml_client = MLClient.from_config(credential=credential, path=config_path)
        else:
            ml_client = MLClient.from_config(credential=credential)
        
        print(f"Connected successfully to Azure ML Workspace: {ml_client.workspace_name}")
        
        # Register dataset
        register_azure_ml_data_asset(ml_client, csv_file_path)
        
    except Exception as e:
        print(f"\n[Note on Azure Connection]: Could not connect via config.json: {e}")
        print("Ensure 'config.json' downloaded from your Azure ML workspace is placed in the project root.")