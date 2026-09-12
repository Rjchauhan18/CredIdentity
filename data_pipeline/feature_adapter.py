"""Single-record feature adapter.

Maps ONE MSME's raw Account Aggregator / GST / EPFO / UPI payload into the 12
engineered features the model scores on. This is the same math as the batch
pipeline in `engineer_features.py`, extracted into a pure, reusable function so
it can be called at request time (see the /api/v1/evaluate-raw endpoint) and
unit-tested against known records.

IMPORTANT — noise omission:
The batch script adds `np.random.normal(...)` jitter to four features
(late_gst_filing_frequency, monthly_revenue_growth_rate, active_employee_count_trend,
epfo_wage_consistency_score). That noise is a *training-data realism device* used
when synthesising the 250k dataset — it deliberately blurs the label so the model
can't memorise clean signals. It is NOT part of scoring: a real applicant's features
should be computed deterministically. This adapter therefore reproduces the batch
math EXACTLY except for those four noise injections, which are intentionally omitted.
"""

from typing import Any, Dict

import numpy as np


def compute_features_for_record(msme: Dict[str, Any]) -> Dict[str, Any]:
    """Compute the 12 engineered features for a single raw MSME record.

    Args:
        msme: one raw MSME object with `account_aggregator`, `gst_data`, and
            `epfo_data` sub-objects (same shape as items in the raw JSON array).

    Returns:
        A dict with `msme_id`, `company_name`, the 12 engineered features, and the
        derived `is_defaulter` / `risk_band` labels — matching the batch CSV columns.

    Raises:
        KeyError: if a required sub-object or field is missing. Callers exposing this
            over HTTP should catch and translate into a 4xx.
    """
    aa = msme["account_aggregator"]
    gst = msme["gst_data"]
    epfo = msme["epfo_data"]

    # ---- Labels (derived, display/eval only — never fed to the model) ----
    is_defaulter = msme.get("is_defaulter", 0)
    if is_defaulter == 1:
        risk_band = "C"
    else:
        bounces = sum(1 for t in aa["transactions"] if t.get("is_bounce_event", False))
        risk_band = "B" if bounces >= 3 else "A"

    # ---- 1. Cash Flow Pillar ----
    balances = [day["closing_balance"] for day in aa["daily_balances"]]
    avg_monthly_balance = np.mean(balances) if balances else 0

    inflows = sum(
        t["amount"] for t in aa["transactions"]
        if t["type"] == "INFLOW" and not t.get("is_bounce_event", False)
    )
    outflows = sum(
        t["amount"] for t in aa["transactions"]
        if t["type"] == "OUTFLOW" and not t.get("is_bounce_event", False)
    )
    inflow_to_outflow_ratio = inflows / outflows if outflows > 0 else 1.0

    total_outflow_attempts = sum(1 for t in aa["transactions"] if t["type"] == "OUTFLOW")
    total_bounces = sum(
        1 for t in aa["transactions"]
        if t["type"] == "OUTFLOW" and t.get("is_bounce_event", False)
    )
    bounce_rate = total_bounces / total_outflow_attempts if total_outflow_attempts > 0 else 0

    daily_outflow = outflows / len(aa["daily_balances"]) if aa["daily_balances"] else 1
    depletion_speed_days = avg_monthly_balance / daily_outflow if daily_outflow > 0 else 999

    # ---- 2. Compliance Pillar ----
    variance_list = [
        abs(f["gstr1_reported_revenue"] - f["gstr3b_computed_revenue"]) for f in gst["filings"]
    ]
    gstr_variance = np.mean(variance_list) if variance_list else 0

    late_filings = sum(1 for f in gst["filings"] if f["actual_filing_date"] > f["due_date"])
    # NOTE: batch script adds np.random.normal(0, 0.08) here — omitted for scoring (see module docstring).
    late_gst_frequency = late_filings / len(gst["filings"]) if gst["filings"] else 0
    late_gst_frequency = float(np.clip(late_gst_frequency, 0, 1))

    # ---- 3. Commercial Revenue Pillar ----
    b2b_ratio = gst["business_nature_b2b_ratio"]
    b2b_vs_b2c = b2b_ratio / (1 - b2b_ratio) if b2b_ratio < 1 else 99.0

    sorted_gst = sorted(gst["filings"], key=lambda x: x["tax_period"])
    if len(sorted_gst) >= 2:
        old_rev = float(sorted_gst[0]["gstr1_reported_revenue"])
        new_rev = float(sorted_gst[-1]["gstr1_reported_revenue"])
        rev_growth = (new_rev - old_rev) / old_rev if old_rev > 0 else 0
    else:
        rev_growth = 0
    # NOTE: batch script adds np.random.normal(0, 0.04) here — omitted for scoring.

    # ---- 4. Operational Stability Pillar ----
    sorted_epfo = sorted(epfo["historical_records"], key=lambda x: x["wage_month"])
    if len(sorted_epfo) >= 2:
        old_emp = sorted_epfo[0]["active_employee_count"]
        new_emp = sorted_epfo[-1]["active_employee_count"]
        emp_trend = (new_emp - old_emp) / old_emp if old_emp > 0 else 0
    else:
        emp_trend = 0
    # NOTE: batch script adds np.random.normal(0, 0.03) here — omitted for scoring.

    on_time_epfo = sum(
        1 for e in epfo["historical_records"] if e["payment_date"] <= e["due_date"]
    )
    epfo_consistency = (on_time_epfo / len(epfo["historical_records"])) * 100 if epfo["historical_records"] else 0
    # NOTE: batch script adds np.random.normal(0, 8) here — omitted for scoring.
    epfo_consistency = float(np.clip(epfo_consistency, 0, 100))

    # ---- 5. Risk & Fraud Anomalies ----
    upi_txns = [t for t in aa["transactions"] if t["mode"] == "UPI"]
    upi_velocity = len(upi_txns) / (len(aa["daily_balances"]) / 30) if aa["daily_balances"] else 0

    night_txns = sum(
        1 for t in aa["transactions"]
        if (int(t["timestamp"][11:13]) >= 22) or (int(t["timestamp"][11:13]) <= 5)
    )
    night_ratio = night_txns / len(aa["transactions"]) if aa["transactions"] else 0

    return {
        "msme_id": msme["msme_id"],
        "company_name": msme["company_name"],
        "average_monthly_balance": round(float(avg_monthly_balance), 2),
        "inflow_to_outflow_ratio": round(float(inflow_to_outflow_ratio), 3),
        "bounce_mandate_failure_rate": round(float(bounce_rate), 3),
        "balance_depletion_speed_days": round(float(depletion_speed_days), 1),
        "gstr1_vs_gstr3b_variance": round(float(gstr_variance), 2),
        "late_gst_filing_frequency": round(float(late_gst_frequency), 3),
        "b2b_vs_b2c_revenue_ratio": round(float(b2b_vs_b2c), 2),
        "monthly_revenue_growth_rate": round(float(rev_growth), 3),
        "active_employee_count_trend": round(float(emp_trend), 3),
        "epfo_wage_consistency_score": round(float(epfo_consistency), 1),
        "upi_transaction_velocity_daily": round(float(upi_velocity), 1),
        "night_transaction_ratio": round(float(night_ratio), 3),
        "is_defaulter": is_defaulter,
        "risk_band": risk_band,
    }
