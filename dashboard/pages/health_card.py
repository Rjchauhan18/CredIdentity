import streamlit as st
from utils.visualizations import create_radar_chart
from utils.loan_matcher import get_top_matches
from utils.api_client import get_msme_data

data = get_msme_data()

st.title("🏢 MSME Financial Health Card")
st.caption(f"Verified Digital Footprint Identity Profile for **{data['company_name']}**")
st.divider()

col1, col2 = st.columns([1, 1.8])

with col1:
    st.subheader("Unified Health Score")
    score = data['overall_credit_score']
    
    if score >= 750: color = "#66BB6A"
    elif score >= 600: color = "#FFA726"
    else: color = "#EF5350"
        
    st.markdown(f"<h1 style='text-align: center; color: {color}; font-size: 72px; margin-bottom: 0px;'>{score}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; font-size: 20px;'>Rating Tier: <b>{data['credit_rating_tier']}</b></p>", unsafe_allow_html=True)
    
    # FIX: Replaced use_container_width=True with width="stretch"
    st.plotly_chart(create_radar_chart(data['four_pillars']), width="stretch")

with col2:
    st.subheader("SHAP Performance Insights")
    
    st.success("**Top Strengths (Score Drivers):**\n" + "\n".join([f"- {s['friendly_text']}" for s in data['shap_drivers']['top_3_strengths']]))
    st.error("**Risk Flags (Areas of Distress):**\n" + "\n".join([f"- {r['friendly_text']}" for r in data['shap_drivers']['top_3_risks']]))
    
    st.info(f"💡 **Actionable Optimization Guidance:** {data['actionable_guidance']}")
    
    st.subheader("OCEN 4.0 Pre-Matched Loan Products")
    matches = get_top_matches(data['overall_credit_score'])
    
    for m in matches:
        with st.container(border=True):
            c1, c2 = st.columns([2, 1])
            c1.markdown(f"🏦 **{m['lender_name']}** — *{m['loan_type']}*")
            c1.caption(f"Tenure: {m['tenure_months']} Mos | Product ID: {m['product_id']}")
            c2.markdown(f"**{m['interest_rate_pa']}% p.a.**")
            c2.markdown(f"Limit: **₹{m['max_amount_inr']:,.0f}**")