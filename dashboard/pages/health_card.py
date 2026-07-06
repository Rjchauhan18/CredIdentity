import streamlit as st
from utils.visualizations import create_radar_chart
from utils.loan_matcher import get_top_matches
from utils.api_client import get_msme_data, get_msme_dropdown_options

st.title("🏢 MSME Financial Health Card")

options = get_msme_dropdown_options()

if options:

    # --- ADDED: Set up the app's memory ---
    if "selected_company" not in st.session_state:
        st.session_state.selected_company = options[0]

    # Create the dropdown, and tell it to look at the memory for its default value
    selected_option = st.selectbox(
        "🔍 Search and Select a Company:", 
        options=options,
        index=options.index(st.session_state.selected_company) if st.session_state.selected_company in options else 0
    )

    # Save whatever the user just clicked back into memory!
    st.session_state.selected_company = selected_option

    # Extract the ID and fetch the data
    if selected_option:
        search_id = selected_option.split("(")[-1].replace(")", "")
        data = get_msme_data(search_id)
        
        # 5. Error handling
        if not data or "detail" in data:
            st.error(f"⚠️ Could not locate MSME profile.")
        else:
            # 6. Render Dashboard
            st.subheader(f"Verified Digital Footprint Identity Profile for **{data['company_name']}**")
            st.divider()

            col1, col2 = st.columns([1, 1.8])

            with col1:
                st.markdown("### Unified Health Score")
                score = data['overall_credit_score']
                
                if score >= 750: color = "#66BB6A"
                elif score >= 600: color = "#FFA726"
                else: color = "#EF5350"
                    
                st.markdown(f"<h1 style='text-align: center; color: {color}; font-size: 72px; margin-bottom: 0px;'>{score}</h1>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; font-size: 20px;'>Rating Tier: <b>{data['credit_rating_tier']}</b></p>", unsafe_allow_html=True)
                
                st.plotly_chart(create_radar_chart(data['four_pillars']), width="stretch")

            with col2:
                st.markdown("### SHAP Performance Insights")
                
                st.success("**Top Strengths (Score Drivers):**\n" + "\n".join([f"- {s['friendly_text']}" for s in data['shap_drivers']['top_3_strengths']]))
                st.error("**Risk Flags (Areas of Distress):**\n" + "\n".join([f"- {r['friendly_text']}" for r in data['shap_drivers']['top_3_risks']]))
                
                st.info(f"💡 **Actionable Optimization Guidance:** {data['actionable_guidance']}")
                
                st.markdown("### OCEN 4.0 Pre-Matched Loan Products")
                matches = get_top_matches(data['overall_credit_score'])
                
                for m in matches:
                    with st.container(border=True):
                        c1, c2 = st.columns([2, 1])
                        c1.markdown(f"🏦 **{m['lender_name']}** — *{m['loan_type']}*")
                        c1.caption(f"Tenure: {m['tenure_months']} Mos | Product ID: {m['product_id']}")
                        c2.markdown(f"**{m['interest_rate_pa']}% p.a.**")
                        c2.markdown(f"Limit: **₹{m['max_amount_inr']:,.0f}**")

else:
    st.error(f"🔌 System Alert: {options.error['message']}")
    st.info("The backend scoring engine might be waking up. Please refresh in a few moments.")