import os
import sys

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.models.inference import IncidentTriageEngine

if __name__ == "__main__":
    print("--- Testing Local Inference Engine ---")
    engine = IncidentTriageEngine(artifacts_dir=os.path.join(project_root, "outputs"))
    
    # Test sample incident
    sample_short = "Database connection pool exhausted on primary postgres instance"
    sample_desc = "Users are receiving 504 Gateway Timeouts when attempting to log in. Connection count reached max limit of 500."
    
    print(f"\nTest Input:\n- Short Desc: {sample_short}\n- Desc: {sample_desc}\n")
    
    result = engine.predict(sample_short, sample_desc)
    
    print("Prediction Result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    
    print("\nInference engine verified successfully and ready for FastAPI containerization!")