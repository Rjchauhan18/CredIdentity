import json
import pandas as pd
import numpy as np
from datetime import datetime

def calculate_12_features(input_json_path, output_csv_path):
    with open(input_json_path, "r") as f:
        data = json.load(f)
        
    features_list = []
    
    for msme in data:
        # Load sub-dictionaries
        aa = msme["account_aggregator"]
        gst = msme["gst_data"]
        epfo = msme["epfo_data"]
        
        # --- 1. Cash Flow Pillar ---
        balances = [day["closing_balance"] for day in aa["daily_balances"]]
        avg_monthly_balance = np.mean(balances) if balances else 0
        
        inflows = sum(t["amount"] for t in aa["transactions"] if t["type"] == "INFLOW" and not t["is_bounce_event"])
        outflows = sum(t["amount"] for t in aa["transactions"] if t["type"] == "OUTFLOW" and not t["is_bounce_event"])
        inflow_to_outflow_ratio = inflows / outflows if outflows > 0 else 1.0
        
        total_outflow_attempts = sum(1 for t in aa["transactions"] if t["type"] == "OUTFLOW")
        total_bounces = sum(1 for t in aa["transactions"] if t["type"] == "OUTFLOW" and t["is_bounce_event"])
        bounce_rate = total_bounces / total_outflow_attempts if total_outflow_attempts > 0 else 0
        
        # Balance Depletion Speed (Proxy: Days of cash runway based on avg daily outflow)
        daily_outflow = outflows / len(aa["daily_balances"]) if aa["daily_balances"] else 1
        depletion_speed_days = avg_monthly_balance / daily_outflow if daily_outflow > 0 else 999
        
        # --- 2. Compliance Pillar ---
        variance_list = [abs(f["gstr1_reported_revenue"] - f["gstr3b_computed_revenue"]) for f in gst["filings"]]
        gstr_variance = np.mean(variance_list) if variance_list else 0
        
        late_filings = sum(1 for f in gst["filings"] if f["actual_filing_date"] > f["due_date"])
        late_gst_frequency = late_filings / len(gst["filings"]) if gst["filings"] else 0
        
        # --- 3. Commercial Revenue Pillar ---
        b2b_ratio = gst["business_nature_b2b_ratio"]
        b2b_vs_b2c = b2b_ratio / (1 - b2b_ratio) if b2b_ratio < 1 else 99.0
        
        # Sort filings by date to calculate growth
        sorted_gst = sorted(gst["filings"], key=lambda x: x["tax_period"])
        if len(sorted_gst) >= 2:
            old_rev = sorted_gst[0]["gstr1_reported_revenue"]
            new_rev = sorted_gst[-1]["gstr1_reported_revenue"]
            rev_growth = (new_rev - old_rev) / old_rev if old_rev > 0 else 0
        else:
            rev_growth = 0
            
        # --- 4. Operational Stability Pillar ---
        sorted_epfo = sorted(epfo["historical_records"], key=lambda x: x["wage_month"])
        if len(sorted_epfo) >= 2:
            old_emp = sorted_epfo[0]["active_employee_count"]
            new_emp = sorted_epfo[-1]["active_employee_count"]
            emp_trend = (new_emp - old_emp) / old_emp if old_emp > 0 else 0
        else:
            emp_trend = 0
            
        on_time_epfo = sum(1 for e in epfo["historical_records"] if e["payment_date"] <= e["due_date"])
        epfo_consistency = (on_time_epfo / len(epfo["historical_records"])) * 100 if epfo["historical_records"] else 0
        
        # --- 5. Risk & Fraud Anomalies ---
        upi_txns = [t for t in aa["transactions"] if t["mode"] == "UPI"]
        upi_velocity = len(upi_txns) / (len(aa["daily_balances"]) / 30) if aa["daily_balances"] else 0
        
        night_txns = sum(1 for t in aa["transactions"] if 22 <= datetime.strptime(t["timestamp"], "%Y-%m-%dT%H:%M:%SZ").hour or datetime.strptime(t["timestamp"], "%Y-%m-%dT%H:%M:%SZ").hour <= 5)
        night_ratio = night_txns / len(aa["transactions"]) if aa["transactions"] else 0
        
        # Compile Row
        features_list.append({
            "msme_id": msme["msme_id"],
            "company_name": msme["company_name"],
            "average_monthly_balance": round(avg_monthly_balance, 2),
            "inflow_to_outflow_ratio": round(inflow_to_outflow_ratio, 3),
            "bounce_mandate_failure_rate": round(bounce_rate, 3),
            "balance_depletion_speed_days": round(depletion_speed_days, 1),
            "gstr1_vs_gstr3b_variance": round(gstr_variance, 2),
            "late_gst_filing_frequency": round(late_gst_frequency, 3),
            "b2b_vs_b2c_revenue_ratio": round(b2b_vs_b2c, 2),
            "monthly_revenue_growth_rate": round(rev_growth, 3),
            "active_employee_count_trend": round(emp_trend, 3),
            "epfo_wage_consistency_score": round(epfo_consistency, 1),
            "upi_transaction_velocity_daily": round(upi_velocity, 1),
            "night_transaction_ratio": round(night_ratio, 3)
        })
        
    # Output to ML-ready format
    df = pd.DataFrame(features_list)
    df.to_csv(output_csv_path, index=False)
    print(f"✅ Extracted 12 engineering features for {len(df)} MSMEs into {output_csv_path}")
    print(df.head())

if __name__ == "__main__":
    calculate_12_features("data/raw/raw_msme_data.json", "data/processed/msme_engineered_features.csv")