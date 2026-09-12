import ijson
import csv
import numpy as np
import os

# Works both as a package import (`python -m data_pipeline.engineer_features`, and
# when imported by the backend) and as a direct script (`python data_pipeline/engineer_features.py`),
# where only the script's own directory is on sys.path.
try:
    from data_pipeline.feature_adapter import compute_features_for_record
except ModuleNotFoundError:
    from feature_adapter import compute_features_for_record

def calculate_12_features(input_json_path, output_csv_path):
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    print(f"Streaming raw data from {input_json_path} to prevent MemoryError...")
    
    # Define the exact columns we want in our output CSV
    fieldnames = [
        "msme_id", "company_name", "average_monthly_balance", "inflow_to_outflow_ratio",
        "bounce_mandate_failure_rate", "balance_depletion_speed_days", "gstr1_vs_gstr3b_variance",
        "late_gst_filing_frequency", "b2b_vs_b2c_revenue_ratio", "monthly_revenue_growth_rate",
        "active_employee_count_trend", "epfo_wage_consistency_score", "upi_transaction_velocity_daily",
        "night_transaction_ratio", "is_defaulter", "risk_band"
    ]
    
    # Open the JSON file in binary mode (required by ijson) and the CSV in write mode
    with open(input_json_path, "rb") as infile, open(output_csv_path, "w", newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # ijson streams the array items one by one without loading the whole file
        items = ijson.items(infile, 'item')
        
        for idx, msme in enumerate(items):
            # Deterministic feature math lives in the shared adapter so request-time
            # scoring (evaluate-raw endpoint) and this batch generator can never drift.
            row = compute_features_for_record(msme)

            # TRAINING-DATA REALISM ONLY: re-inject the stochastic jitter on the four
            # "soft" compliance/operational features. This deliberately blurs the label
            # so the model can't memorise clean synthetic signals. It must NOT run at
            # scoring time — hence it lives here in the generator, not in the adapter.
            row["late_gst_filing_frequency"] = round(
                float(np.clip(row["late_gst_filing_frequency"] + np.random.normal(0, 0.08), 0, 1)), 3
            )
            row["monthly_revenue_growth_rate"] = round(
                float(row["monthly_revenue_growth_rate"] + np.random.normal(0, 0.04)), 3
            )
            row["active_employee_count_trend"] = round(
                float(row["active_employee_count_trend"] + np.random.normal(0, 0.03)), 3
            )
            row["epfo_wage_consistency_score"] = round(
                float(np.clip(row["epfo_wage_consistency_score"] + np.random.normal(0, 8), 0, 100)), 1
            )

            writer.writerow(row)

            if (idx + 1) % 10000 == 0:
                print(f"Processed {idx + 1} features safely...")
            
    print(f"✅ Successfully extracted all features into {output_csv_path} with zero memory bloat!")

if __name__ == "__main__":
    calculate_12_features("data/raw/raw_msme_data.json", "data/processed/msme_final_engineered_features.csv")