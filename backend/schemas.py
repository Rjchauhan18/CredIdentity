from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Dict, List

class ShapDriverItem(BaseModel):
    feature_name: str
    impact_value: float
    friendly_text: str

class CounterfactualPath(BaseModel):
    """One 'path to a better score' lever, projected by real model re-scoring."""
    feature: str
    label: str
    current_value: str
    target_value: str
    projected_score_gain: int
    projected_score: int
    crosses_tier: bool

class ShapDrivers(BaseModel):
    top_3_strengths: List[ShapDriverItem]
    top_3_risks: List[ShapDriverItem]

class FourPillars(BaseModel):
    cash_flow_resiliency: int
    commercial_activity_revenue: int
    operational_employee_stability: int
    compliance_risk: int

class EngineeredFeatures(BaseModel):
    average_monthly_balance: float
    inflow_to_outflow_ratio: float
    bounce_mandate_failure_rate: float
    balance_depletion_speed_days: float
    gstr1_vs_gstr3b_variance: float
    late_gst_filing_frequency: float
    b2b_vs_b2c_revenue_ratio: float
    monthly_revenue_growth_rate: float
    active_employee_count_trend: float
    epfo_wage_consistency_score: float
    upi_transaction_velocity_daily: float
    night_transaction_ratio: float

class MSMEEvaluationResponse(BaseModel):
    # This configuration lets FastAPI use the alias name when outputting JSON
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    msme_id: str
    company_name: str
    evaluation_timestamp: str
    overall_credit_score: int = Field(..., ge=300, le=900)
    credit_rating_tier: str
    confidence_level: float
    challenger_disagreement_flag: bool
    four_pillars: FourPillars
    
    # Valid Python identifier mapping to the exact JSON key string
    engineered_features_12: EngineeredFeatures = Field(..., alias="12_engineered_features")
    
    shap_drivers: ShapDrivers
    actionable_guidance: str
    automated_credit_assessment_note: str
    counterfactual_paths: List[CounterfactualPath] = Field(default_factory=list)


class RawMSMEProfile(BaseModel):
    """Raw digital-footprint payload accepted by /api/v1/evaluate-raw.

    Loosely typed on purpose: the real AA/GST/EPFO schemas are deeply nested and the
    feature adapter validates the fields it needs. We enforce presence of the three
    top-level sources and identity fields here, and cap size at the route layer.
    """
    msme_id: str
    company_name: str
    account_aggregator: Dict[str, Any]
    gst_data: Dict[str, Any]
    epfo_data: Dict[str, Any]
    is_defaulter: int = 0