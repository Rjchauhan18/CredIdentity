"""Unit tests for the single-record feature adapter.

Verifies the adapter reproduces the deterministic feature math and derives labels
correctly. The batch pipeline adds Gaussian noise to four features for training-data
realism; the adapter intentionally omits that, so we assert on the noise-free values.
"""

import math

import pytest

from data_pipeline.feature_adapter import compute_features_for_record


def _sample_record():
    """A minimal but complete raw MSME payload exercising every feature branch."""
    return {
        "msme_id": "TEST001",
        "company_name": "Test Traders",
        "is_defaulter": 0,
        "account_aggregator": {
            "daily_balances": [
                {"closing_balance": 100000},
                {"closing_balance": 120000},
                {"closing_balance": 80000},
            ],
            "transactions": [
                {"type": "INFLOW", "amount": 50000, "mode": "UPI", "timestamp": "2024-01-01T10:00:00", "is_bounce_event": False},
                {"type": "INFLOW", "amount": 30000, "mode": "NEFT", "timestamp": "2024-01-02T14:00:00", "is_bounce_event": False},
                {"type": "OUTFLOW", "amount": 20000, "mode": "UPI", "timestamp": "2024-01-03T23:00:00", "is_bounce_event": False},
                {"type": "OUTFLOW", "amount": 10000, "mode": "UPI", "timestamp": "2024-01-04T02:00:00", "is_bounce_event": True},
            ],
        },
        "gst_data": {
            "business_nature_b2b_ratio": 0.6,
            "filings": [
                {"tax_period": "2024-01", "gstr1_reported_revenue": 100000, "gstr3b_computed_revenue": 98000,
                 "actual_filing_date": "2024-02-15", "due_date": "2024-02-20"},
                {"tax_period": "2024-02", "gstr1_reported_revenue": 120000, "gstr3b_computed_revenue": 121000,
                 "actual_filing_date": "2024-03-25", "due_date": "2024-03-20"},
            ],
        },
        "epfo_data": {
            "historical_records": [
                {"wage_month": "2024-01", "active_employee_count": 10, "payment_date": "2024-02-10", "due_date": "2024-02-15"},
                {"wage_month": "2024-02", "active_employee_count": 12, "payment_date": "2024-03-20", "due_date": "2024-03-15"},
            ],
        },
    }


def test_returns_all_expected_columns():
    out = compute_features_for_record(_sample_record())
    expected = {
        "msme_id", "company_name", "average_monthly_balance", "inflow_to_outflow_ratio",
        "bounce_mandate_failure_rate", "balance_depletion_speed_days", "gstr1_vs_gstr3b_variance",
        "late_gst_filing_frequency", "b2b_vs_b2c_revenue_ratio", "monthly_revenue_growth_rate",
        "active_employee_count_trend", "epfo_wage_consistency_score", "upi_transaction_velocity_daily",
        "night_transaction_ratio", "is_defaulter", "risk_band",
    }
    assert set(out.keys()) == expected


def test_cash_flow_math():
    out = compute_features_for_record(_sample_record())
    # avg of 100000,120000,80000 = 100000
    assert out["average_monthly_balance"] == 100000.0
    # inflows 80000 / outflows 20000 (bounced 10000 excluded) = 4.0
    assert out["inflow_to_outflow_ratio"] == 4.0
    # 1 bounce / 2 outflow attempts = 0.5
    assert out["bounce_mandate_failure_rate"] == 0.5


def test_label_derivation_non_defaulter_low_bounce():
    out = compute_features_for_record(_sample_record())
    assert out["is_defaulter"] == 0
    # only 1 bounce event -> band A
    assert out["risk_band"] == "A"


def test_defaulter_forces_band_c():
    rec = _sample_record()
    rec["is_defaulter"] = 1
    out = compute_features_for_record(rec)
    assert out["risk_band"] == "C"


def test_late_gst_and_variance():
    out = compute_features_for_record(_sample_record())
    # 1 of 2 filings late -> 0.5 (no noise in adapter)
    assert out["late_gst_filing_frequency"] == 0.5
    # mean(|100000-98000|, |120000-121000|) = mean(2000,1000)=1500
    assert out["gstr1_vs_gstr3b_variance"] == 1500.0


def test_night_ratio_and_upi_velocity():
    out = compute_features_for_record(_sample_record())
    # 2 of 4 txns at night (23:00 and 02:00) = 0.5
    assert out["night_transaction_ratio"] == 0.5
    # 3 UPI txns over (3 daily_balances/30) = 3 / 0.1 = 30.0
    assert out["upi_transaction_velocity_daily"] == 30.0


def test_missing_source_raises_keyerror():
    rec = _sample_record()
    del rec["gst_data"]
    with pytest.raises(KeyError):
        compute_features_for_record(rec)
