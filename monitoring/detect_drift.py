import os
import json
import pandas as pd

def run_native_drift_analysis():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(current_dir, "inference_logs.jsonl")
    
    if not os.path.exists(log_file):
        print("❌ No inference logs found yet. Make some prediction requests to the API first!")
        return

    # 1. Load current production inference logs
    current_data = pd.read_json(log_file, lines=True)
    print(f"Loaded {len(current_data)} production inference records for analysis.")

    # 2. Define Reference Baseline (Training distribution summary)
    reference_data = pd.DataFrame({
        "short_description": [
            "Database connection timeout on primary node",
            "API gateway 504 gateway timeout",
            "Disk space exhaustion on logging volume",
            "Memory leak detected in worker service",
            "SSL certificate expiration warning"
        ],
        "predicted_category": ["Database", "Network", "Infrastructure", "Application", "Security"],
        "category_confidence": [0.85, 0.90, 0.78, 0.82, 0.95]
    })

    # 3. Perform Statistical Drift Comparison
    print("\n--- Phase 2: Native Data & Model Drift Report ---")
    
    # Category Distribution Shift
    print("\n[1] Predicted Category Distribution Shift:")
    ref_cat_dist = reference_data["predicted_category"].value_counts(normalize=True) * 100
    curr_cat_dist = current_data["predicted_category"].value_counts(normalize=True) * 100
    
    comparison_df = pd.DataFrame({"Baseline (%)": ref_cat_dist, "Production (%)": curr_cat_dist}).fillna(0)
    print(comparison_df.to_string())

    # Confidence Degradation Check
    print("\n[2] Model Confidence Metrics:")
    ref_mean_conf = reference_data["category_confidence"].mean()
    curr_mean_conf = current_data["category_confidence"].mean()
    print(f"Baseline Mean Confidence: {ref_mean_conf:.4f}")
    print(f"Production Mean Confidence: {curr_mean_conf:.4f}")
    
    if curr_mean_conf < (ref_mean_conf - 0.15):
        print("⚠️ WARNING: Significant model confidence drop detected! Potential concept drift.")
    else:
        print("✅ Model confidence levels are stable within acceptable thresholds.")

    # 4. Export JSON Summary Report
    report_path = os.path.join(current_dir, "drift_summary_report.json")
    report_data = {
        "total_production_samples": len(current_data),
        "baseline_mean_confidence": float(ref_mean_conf),
        "production_mean_confidence": float(curr_mean_conf),
        "category_distribution_production": curr_cat_dist.to_dict()
    }
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=4)
    print(f"\nDrift summary report successfully saved to: {report_path}")

if __name__ == "__main__":
    run_native_drift_analysis()