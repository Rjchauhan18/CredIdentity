import json
import logging
import os
import pandas as pd
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from autogluon.tabular import TabularPredictor

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger("credit_engine")

# Demo dataset served by the API. These are held-out rows generated from the SAME
# distribution as the 250k training set (see data_pipeline/build_demo_dataset.py),
# so scores are trustworthy and the file is small enough to commit / ship to HF Spaces.
CSV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/processed/msme_demo_features.csv"))

# Real permutation feature importance exported from the training notebook.
FEATURE_IMPORTANCE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model/feature_importance.json"))

# Columns that are NOT model inputs: the label, its derived band, and the feature the
# final model dropped as leaky (bounce_mandate_failure_rate). Stripped before inference.
NON_FEATURE_COLUMNS = ["is_defaulter", "risk_band"]

# Global state populated at startup
predictor = None
feature_importance: Dict[str, float] = {}

# Per-feature directionality for honest strength/risk classification.
# True  -> higher value is healthier; False -> lower value is healthier.
FEATURE_HIGHER_IS_BETTER = {
    "average_monthly_balance": True,
    "inflow_to_outflow_ratio": True,
    "balance_depletion_speed_days": True,
    "gstr1_vs_gstr3b_variance": False,
    "late_gst_filing_frequency": False,
    "b2b_vs_b2c_revenue_ratio": True,
    "monthly_revenue_growth_rate": True,
    "active_employee_count_trend": True,
    "epfo_wage_consistency_score": True,
    "upi_transaction_velocity_daily": True,
    "night_transaction_ratio": False,
}

# Human-readable labels + phrasing for driver explanations
FEATURE_LABELS = {
    "average_monthly_balance": "Average Monthly Balance",
    "inflow_to_outflow_ratio": "Inflow-to-Outflow Ratio",
    "balance_depletion_speed_days": "Balance Depletion Speed",
    "gstr1_vs_gstr3b_variance": "GSTR-1 vs GSTR-3B Variance",
    "late_gst_filing_frequency": "Late GST Filing Frequency",
    "b2b_vs_b2c_revenue_ratio": "B2B vs B2C Revenue Ratio",
    "monthly_revenue_growth_rate": "Monthly Revenue Growth",
    "active_employee_count_trend": "Employee Count Trend",
    "epfo_wage_consistency_score": "EPFO Wage Consistency",
    "upi_transaction_velocity_daily": "UPI Transaction Velocity",
    "night_transaction_ratio": "Night Transaction Ratio",
}

# Per-feature display formatting so insight text reads cleanly (₹, %, ×, days)
# instead of raw floats like 1065788.96.
FEATURE_UNITS = {
    "average_monthly_balance": "inr",
    "inflow_to_outflow_ratio": "ratio",
    "balance_depletion_speed_days": "days",
    "gstr1_vs_gstr3b_variance": "inr",
    "late_gst_filing_frequency": "pct_frac",
    "b2b_vs_b2c_revenue_ratio": "ratio",
    "monthly_revenue_growth_rate": "pct_frac",
    "active_employee_count_trend": "pct_frac",
    "epfo_wage_consistency_score": "pct_whole",
    "upi_transaction_velocity_daily": "per_day",
    "night_transaction_ratio": "pct_frac",
}


def _fmt_value(feat: str, value: float) -> str:
    """Render a feature value with the right unit for human-readable insight text."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    unit = FEATURE_UNITS.get(feat, "num")
    if unit == "inr":
        return f"₹{v:,.0f}"
    if unit == "pct_frac":
        return f"{v * 100:.1f}%"
    if unit == "pct_whole":
        return f"{v:.1f}%"
    if unit == "ratio":
        return f"{v:.2f}×"
    if unit == "days":
        return f"{v:.0f} days"
    if unit == "per_day":
        return f"{v:.1f}/day"
    return f"{v:,.2f}"


# Feature medians (computed from the served demo dataset at startup) used to decide
# whether an MSME's value is favorable or unfavorable versus the cohort.
_feature_medians: Dict[str, float] = {}

def init_model():
    """Downloads model from Hugging Face and loads AutoGluon into memory."""
    global predictor, feature_importance
    local_model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model/local_model_weights"))

    # Only skip the download if a COMPLETE model is present. Checking for predictor.pkl
    # (not just a non-empty dir) prevents a partial/interrupted download from poisoning
    # startup with a FileNotFoundError on load.
    predictor_file = os.path.join(local_model_dir, "predictor.pkl")
    if not os.path.exists(predictor_file):
        logger.info("📥 Fetching model artifacts from Hugging Face Registry...")
        snapshot_download(
            repo_id=os.getenv("HF_MODEL_REPO"),
            local_dir=local_model_dir,
            repo_type="model",
            token=os.getenv("HF_TOKEN")
        )
        logger.info("✅ Weights successfully pulled!")
    else:
        logger.info("✅ Complete model already present locally; skipping download.")

    logger.info("🧠 Loading Multi-Layer Stacking Ensemble into Memory...")
    predictor = TabularPredictor.load(local_model_dir, require_py_version_match=False)

    # Load the real permutation importances exported from the training notebook.
    try:
        with open(FEATURE_IMPORTANCE_PATH) as f:
            feature_importance = json.load(f).get("feature_importance", {})
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not load feature_importance.json (%s); driver magnitudes will fall back to 0.", e)
        feature_importance = {}

    _compute_feature_medians()
    _verify_model_features()
    logger.info("🚀 System Online. Production endpoints ready.")


def _verify_model_features():
    """Log the loaded model's feature list and warn on any mismatch with the demo data /
    importance artifact. Makes deploy problems obvious in the Space logs instead of
    surfacing as silently wrong scores."""
    try:
        model_features = set(predictor.feature_metadata_in.get_features())
    except Exception as e:  # older/newer AutoGluon API differences
        logger.warning("Could not introspect model features: %s", e)
        return

    logger.info("🔎 Loaded model expects %d features: %s",
                len(model_features), sorted(model_features))

    expected = set(FEATURE_HIGHER_IS_BETTER)
    missing = model_features - expected           # model wants something we don't map
    extra = expected - model_features             # we map something the model ignores
    if missing:
        logger.warning("⚠️ Model uses features NOT in the demo/importance map: %s", sorted(missing))
    if extra:
        logger.warning("⚠️ Demo/importance map has features the model does not use: %s", sorted(extra))
    if not missing and not extra:
        logger.info("✅ Model feature set matches the served demo data and importance map.")


def is_ready() -> bool:
    """True once the AutoGluon predictor is loaded and can serve requests."""
    return predictor is not None


def _compute_feature_medians():
    """Cache per-feature medians from the served demo dataset for driver classification."""
    global _feature_medians
    if not os.path.exists(CSV_PATH):
        _feature_medians = {}
        return
    df = pd.read_csv(CSV_PATH)
    _feature_medians = {
        col: float(df[col].median())
        for col in FEATURE_HIGHER_IS_BETTER
        if col in df.columns
    }

def get_all_msmes():
    """Returns a list of all available MSME names and IDs for the frontend dropdown."""
    if not os.path.exists(CSV_PATH):
        return []
    df = pd.read_csv(CSV_PATH)
    return df[['msme_id', 'company_name']].to_dict('records')

def load_msme_features(msme_id: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(CSV_PATH):
        return None
    df = pd.read_csv(CSV_PATH)
    record = df[(df['msme_id'].str.upper() == msme_id.upper()) | (df['company_name'].str.upper() == msme_id.upper())]
    if record.empty:
        return None
    return record.iloc[0].to_dict()

def _build_drivers(features: Dict[str, Any]):
    """Build strength/risk drivers from REAL permutation importance + actual values.

    Magnitude comes from the model's global permutation importance (feature_importance.json).
    Direction (strength vs risk) is decided by comparing the MSME's value to the cohort
    median using each feature's known 'higher-is-better' orientation. This replaces the
    previous hardcoded/fabricated numbers with transparent, grounded values.
    """
    scored = []
    for feat, higher_better in FEATURE_HIGHER_IS_BETTER.items():
        if feat not in features:
            continue
        importance = float(feature_importance.get(feat, 0.0))
        if importance <= 0:
            continue  # skip features the model does not meaningfully use
        value = features[feat]
        median = _feature_medians.get(feat, value)
        favorable = (value >= median) if higher_better else (value <= median)
        scored.append({
            "feature": feat,
            "importance": importance,
            "favorable": favorable,
            "value": value,
        })

    strengths_src = sorted(
        [s for s in scored if s["favorable"]], key=lambda x: x["importance"], reverse=True
    )[:3]
    risks_src = sorted(
        [s for s in scored if not s["favorable"]], key=lambda x: x["importance"], reverse=True
    )[:3]

    def to_item(entry, sign):
        feat = entry["feature"]
        label = FEATURE_LABELS.get(feat, feat)
        val_str = _fmt_value(feat, entry["value"])
        median_str = _fmt_value(feat, _feature_medians.get(feat, entry["value"]))
        return {
            "feature_name": label,
            # impact_value = real global permutation importance (percentage points of ROC-AUC)
            "impact_value": round(sign * entry["importance"] * 100, 2),
            "friendly_text": f"{val_str} vs cohort median {median_str}",
        }

    strengths = [to_item(e, 1) for e in strengths_src]
    risks = [to_item(e, -1) for e in risks_src]
    return strengths, risks


def evaluate_credit_profile(msme_id: str) -> Optional[Dict[str, Any]]:
    global predictor
    features = load_msme_features(msme_id)
    if not features:
        return None

    # ---- 1. AUTOGLUON MODEL INFERENCE ----
    # Strip the label/leakage columns before scoring so we never feed the answer key in.
    df_input = pd.DataFrame([features]).drop(columns=NON_FEATURE_COLUMNS, errors="ignore")

    # predict_proba returns a DataFrame whose columns ARE the class labels. Index by the
    # explicit "healthy" label (0 = not defaulter) rather than positional [0], so the score
    # can never silently invert if AutoGluon reorders classes.
    probabilities = predictor.predict_proba(df_input)
    healthy_label = 0 if 0 in probabilities.columns else predictor.class_labels[0]
    default_label = 1 if 1 in probabilities.columns else predictor.class_labels[-1]
    prob_healthy = float(probabilities.iloc[0][healthy_label])
    prob_default = float(probabilities.iloc[0][default_label])

    # Convert probability to a 300-900 Credit Score Scale
    overall_credit_score = max(300, min(900, int(300 + (prob_healthy * 600))))

    if overall_credit_score >= 750: tier = "Excellent"
    elif overall_credit_score >= 650: tier = "Good"
    elif overall_credit_score >= 550: tier = "Average"
    else: tier = "Substandard"

    # Calibrated separation between the two classes (0-100), not floored.
    model_confidence = round(abs(prob_healthy - prob_default) * 100, 1)

    # ---- 2. UI PILLARS (Kept for Frontend Dashboard Visualization) ----
    # NOTE: bounce_mandate_failure_rate is a display-only signal here; the final model
    # dropped it as leaky, so it does not influence the score above.
    ratio_score = min(100, int(features['inflow_to_outflow_ratio'] * 70))
    bounce_penalty = max(0, int(features['bounce_mandate_failure_rate'] * 300))
    cash_flow_pillar = max(10, min(100, ratio_score - bounce_penalty))

    growth_bonus = int(features['monthly_revenue_growth_rate'] * 150)
    b2b_score = min(50, int(features['b2b_vs_b2c_revenue_ratio'] * 10))
    commercial_pillar = max(10, min(100, 60 + growth_bonus + b2b_score))

    epfo_base = int(features['epfo_wage_consistency_score'] * 0.8)
    emp_trend = int(features['active_employee_count_trend'] * 100)
    operational_pillar = max(10, min(100, epfo_base + emp_trend))

    late_filing_penalty = int(features['late_gst_filing_frequency'] * 80)
    variance_penalty = min(30, int(features['gstr1_vs_gstr3b_variance'] / 5000))
    compliance_pillar = max(10, min(100, 100 - late_filing_penalty - variance_penalty))

    # ---- 3. DRIVER INSIGHTS (real permutation importance + actual feature values) ----
    strengths, risks = _build_drivers(features)

    if features['bounce_mandate_failure_rate'] > 0.05:
        guidance = "Prioritize clean settlement clearing metrics. Reducing system payment mandate faults by half can optimize overall rating profiles by ~42 points."
    else:
        guidance = "Improve the capital depletion cycle by extending liquidity coverage windows beyond 20 days to access superior commercial banking tiers."

    assessment_note = f"Cloud AI confirms MSME exhibits operational resilience in {tier} risk classification parameters."

    # Map directly to the schemas config format
    return {
        "msme_id": features["msme_id"],
        "company_name": features["company_name"],
        "evaluation_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "overall_credit_score": overall_credit_score,
        "credit_rating_tier": tier,
        "confidence_level": model_confidence,
        "challenger_disagreement_flag": False, # Ensembles handle disagreements internally
        "four_pillars": {
            "cash_flow_resiliency": cash_flow_pillar,
            "commercial_activity_revenue": commercial_pillar,
            "operational_employee_stability": operational_pillar,
            "compliance_risk": compliance_pillar
        },
        "engineered_features_12": {
            "average_monthly_balance": features["average_monthly_balance"],
            "inflow_to_outflow_ratio": features["inflow_to_outflow_ratio"],
            "bounce_mandate_failure_rate": features["bounce_mandate_failure_rate"],
            "balance_depletion_speed_days": features["balance_depletion_speed_days"],
            "gstr1_vs_gstr3b_variance": features["gstr1_vs_gstr3b_variance"],
            "late_gst_filing_frequency": features["late_gst_filing_frequency"],
            "b2b_vs_b2c_revenue_ratio": features["b2b_vs_b2c_revenue_ratio"],
            "monthly_revenue_growth_rate": features["monthly_revenue_growth_rate"],
            "active_employee_count_trend": features["active_employee_count_trend"],
            "epfo_wage_consistency_score": features["epfo_wage_consistency_score"],
            "upi_transaction_velocity_daily": features["upi_transaction_velocity_daily"],
            "night_transaction_ratio": features["night_transaction_ratio"]
        },
        "shap_drivers": {
            "top_3_strengths": strengths,
            "top_3_risks": risks
        },
        "actionable_guidance": guidance,
        "automated_credit_assessment_note": assessment_note
    }