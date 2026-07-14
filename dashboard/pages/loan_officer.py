import streamlit as st
import pandas as pd
from utils.api_client import get_msme_data, get_msme_dropdown_options
from utils.visualizations import create_driver_chart

# Set page layout to wide for maximum data real estate
st.set_page_config(layout="wide")

st.title("🛡️ Institutional Underwriting Desk & Risk Matrix")
st.caption("Credit Auditor Workspace — Advanced Machine Learning Explainability Registry")

# Fetch live companies from the backend
options = get_msme_dropdown_options()

if "selected_company" not in st.session_state:
    st.session_state.selected_company = options[0] if options else "No MSME records found. Ensure backend API is running."

selected_option = st.selectbox(
    "🔍 Select Company to Audit:", 
    options=options,
    index=options.index(st.session_state.selected_company) if st.session_state.selected_company in options else 0
)

st.session_state.selected_company = selected_option
st.divider()

if selected_option and "No MSME records found" not in selected_option:
    # Extract structural ID from selection string
    search_id = selected_option.split("(")[-1].replace(")", "").strip()
    
    # Request live data contract from FastAPI
    data = get_msme_data(search_id)

    if not data or "detail" in data:
        st.error("⚠️ Could not locate live MSME profile. Verify backend logs.")
    else:
        # ---------------------------------------------------------
        # HEADER & WARNING SYSTEMS
        # ---------------------------------------------------------
        if data.get('challenger_disagreement_flag'):
            st.warning("⚠️ **Challenger Model Disagreement Alert:** Primary Model and Baseline variance exceeds acceptable thresholds. Manual override required.")

        # ---------------------------------------------------------
        # HERO METRICS SECTION
        # ---------------------------------------------------------
        score = data.get('overall_credit_score', 300)
        tier = data.get('credit_rating_tier', 'Unknown')
        confidence = data.get('confidence_level', 0.0)
        timestamp = data.get('evaluation_timestamp', 'N/A')

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Overall Credit Score", f"{score} / 900")
        with col_m2:
            st.metric("Credit Rating Tier", tier)
        with col_m3:
            st.metric("Model Prediction Confidence", f"{confidence}%")
        with col_m4:
            st.metric("Evaluation Node Timestamp", timestamp.split("T")[0])

        st.divider()

        # ---------------------------------------------------------
        # AI ASSESSMENTS & ACTIONABLE GUIDANCE
        # ---------------------------------------------------------
        st.subheader("🤖 Automated AI Risk Assessment & Strategy")
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.markdown("**Automated Credit Assessment Note:**")
            st.info(data.get('automated_credit_assessment_note', 'No note available.'))
            
        with col_g2:
            st.markdown("**Actionable Optimization Guidance:**")
            st.success(data.get('actionable_guidance', 'No guidance generated.'))

        st.divider()

        # ---------------------------------------------------------
        # EXPLAINABILITY DRILLDOWN (STRENGTHS VS RISKS)
        # ---------------------------------------------------------
        st.subheader("📊 Feature Attribution Insights")
        st.caption("Drivers ranked by the model's real permutation feature importance (ROC-AUC points); direction set by each value vs. the cohort median.")

        shap_drivers = data.get('shap_drivers', {})
        strengths = shap_drivers.get('top_3_strengths', [])
        risks = shap_drivers.get('top_3_risks', [])

        if strengths or risks:
            st.plotly_chart(create_driver_chart(strengths, risks), width="stretch")

        col_str, col_rsk = st.columns(2)

        with col_str:
            st.markdown("### ✅ Top 3 Structural Strengths")
            if strengths:
                for idx, item in enumerate(strengths):
                    with st.container(border=True):
                        st.markdown(f"**{idx+1}. {item.get('feature_name')}**")
                        st.markdown(f"*{item.get('friendly_text')}*")
                        st.caption(f"Global importance: `+{item.get('impact_value')}` pts")
            else:
                st.info("No distinct model strengths computed.")

        with col_rsk:
            st.markdown("### ❌ Top 3 Operational Risks")
            if risks:
                for idx, item in enumerate(risks):
                    with st.container(border=True):
                        st.markdown(f"**{idx+1}. {item.get('feature_name')}**")
                        st.markdown(f"*{item.get('friendly_text')}*")
                        st.caption(f"Global importance: `{item.get('impact_value')}` pts")
            else:
                st.info("No distinct model risk parameters flag detected.")

        st.divider()

        # ---------------------------------------------------------
        # CORE 4 PILLARS & RAW 12 FEATURE METRICS
        # ---------------------------------------------------------
        col_p1, col_p2 = st.columns([1, 1.2])

        with col_p1:
            st.subheader("🏛️ Balanced Core Pillar Sub-scores")
            st.caption("Aggregated risk vectors mapping the operational framework.")
            four_pillars = data.get('four_pillars', {})
            if four_pillars:
                for pillar, val in four_pillars.items():
                    st.progress(val / 100, text=f"**{pillar.replace('_', ' ').title()}**: {val}/100")
            else:
                st.warning("Pillar sub-scores missing from response data payload.")

        with col_p2:
            st.subheader("🧮 12 Core Engineered Features Matrix")
            st.caption("Actual raw telemetry points received from corporate GST, Banking, and EPFO layers.")
            
            # Extract safely using the string-key format to protect against frontend syntax errors
            features = data.get("12_engineered_features", {})

            if features:
                feat_df = pd.DataFrame(list(features.items()), columns=["Feature Metric ID", "Evaluated Operational Value"])
                feat_df["Feature Metric ID"] = feat_df["Feature Metric ID"].str.replace("_", " ").str.title()
                st.dataframe(feat_df, hide_index=True,width='stretch')
            else:
                st.warning("Engineered signals missing from data stream parameters.")