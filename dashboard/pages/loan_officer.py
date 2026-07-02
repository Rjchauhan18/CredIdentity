import streamlit as st
import pandas as pd
from utils.visualizations import create_shap_waterfall
from utils.api_client import get_msme_data

data = get_msme_data()

st.title("🛡️ Underwriting Desk & Risk Matrix")
st.caption("Credit Auditor Workspace — Advanced Explanations & Parameter Tracking")
st.divider()

if data['challenger_disagreement_flag']:
    st.warning("⚠️ **Challenger Model Disagreement Alert:** Primary Model (TabNet) and Baseline (XGBoost) variance exceeds 15%. Forwarding case to Senior Credit Committee for manual audit.")

col1, col2 = st.columns([1.5, 1])

with col1:
    # FIX: Replaced use_container_width=True with width="stretch"
    st.plotly_chart(create_shap_waterfall(data['shap_drivers']['top_3_strengths'], data['shap_drivers']['top_3_risks']), width="stretch")
    
    st.subheader("Automated Credit Assessment Note")
    # FIX: Swapped st.help() out for st.info() to avoid printing python's internal string docs
    st.info(data['automated_credit_assessment_note'])

with col2:
    st.metric("Pipeline Data Confidence Level", f"{data['confidence_level']}%")
    
    st.subheader("Pillar Sub-scores")
    for pillar, val in data['four_pillars'].items():
        st.progress(val / 100, text=f"{pillar.replace('_', ' ').title()}: {val}/100")

st.subheader("Calculated Core Technical Feature Analytics (12 Operational Signals)")
features = data['12_engineered_features']

feat_df = pd.DataFrame(list(features.items()), columns=["Feature Metric ID", "Evaluated Value"])
feat_df["Feature Metric ID"] = feat_df["Feature Metric ID"].str.replace("_", " ").str.title()
st.dataframe(feat_df, width="stretch", hide_index=True)