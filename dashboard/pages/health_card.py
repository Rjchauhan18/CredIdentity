import streamlit as st
from utils.visualizations import create_radar_chart, create_driver_chart, create_score_gauge
from utils.loan_matcher import get_top_matches
from utils.api_client import get_msme_data, get_msme_dropdown_options, evaluate_raw_profile

st.title("🏢 MSME Financial Health Card")
st.caption("Alternative-data credit identity for New-to-Credit micro-enterprises")

# --- Point 4: score a raw digital-footprint profile on the fly ---
with st.expander("📤 Score a raw profile (AA / GST / EPFO JSON)"):
    st.caption(
        "Paste a raw MSME payload (account_aggregator, gst_data, epfo_data). "
        "The backend derives the 12 features and scores it with the live model."
    )
    raw_text = st.text_area("Raw profile JSON", height=180, key="raw_profile_json")
    if st.button("Score raw profile"):
        import json as _json
        try:
            parsed = _json.loads(raw_text)
        except _json.JSONDecodeError as exc:
            st.error(f"Invalid JSON: {exc}")
        else:
            result, err = evaluate_raw_profile(parsed)
            if err:
                st.error(err)
            else:
                st.success(
                    f"Scored **{result['company_name']}** → "
                    f"**{result['overall_credit_score']}/900** ({result['credit_rating_tier']})"
                )
                st.json(result, expanded=False)

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
            score = data['overall_credit_score']
            tier = data['credit_rating_tier']

            st.markdown(f"#### {data['company_name']}")
            st.caption(f"Verified digital-footprint identity · {search_id}")
            st.divider()

            col1, col2 = st.columns([1, 1.6], gap="large")

            with col1:
                # Credit-score gauge (replaces the bare floating number)
                st.plotly_chart(create_score_gauge(score, tier), width="stretch")
                st.markdown(
                    f"<p style='text-align:center; color:#888; margin-top:-12px;'>"
                    f"Model confidence: <b>{data['confidence_level']}%</b></p>",
                    unsafe_allow_html=True,
                )

                st.markdown("##### Four-Pillar Breakdown")
                pillars = data['four_pillars']
                pillar_labels = {
                    'cash_flow_resiliency': '💧 Cash Flow',
                    'commercial_activity_revenue': '📈 Commercial',
                    'operational_employee_stability': '👥 Operational',
                    'compliance_risk': '🛡️ Compliance',
                }
                pcols = st.columns(2)
                for i, (key, label) in enumerate(pillar_labels.items()):
                    with pcols[i % 2]:
                        st.metric(label, f"{pillars[key]}/100")

                st.plotly_chart(create_radar_chart(pillars), width="stretch")

            with col2:
                strengths = data['shap_drivers']['top_3_strengths']
                risks = data['shap_drivers']['top_3_risks']

                st.markdown("##### 🔍 Model-Driven Insights")
                st.caption("Ranked by the model's real permutation feature importance (ROC-AUC points).")

                st.plotly_chart(create_driver_chart(strengths, risks), width="stretch")

                icol1, icol2 = st.columns(2)
                with icol1:
                    st.markdown("**✅ Strengths**")
                    for s in strengths:
                        st.markdown(
                            f"<div style='padding:8px 10px;margin-bottom:6px;border-left:3px solid #22C55E;"
                            f"background:rgba(34,197,94,0.08);border-radius:4px;'>"
                            f"<b>{s['feature_name']}</b> <span style='color:#22C55E;'>+{s['impact_value']:.1f} pts</span>"
                            f"<br><span style='color:#94A3B8;font-size:13px;'>{s['friendly_text']}</span></div>",
                            unsafe_allow_html=True,
                        )
                    if not strengths:
                        st.caption("No favorable drivers stood out.")
                with icol2:
                    st.markdown("**⚠️ Risk Flags**")
                    for r in risks:
                        st.markdown(
                            f"<div style='padding:8px 10px;margin-bottom:6px;border-left:3px solid #EF4444;"
                            f"background:rgba(239,68,68,0.08);border-radius:4px;'>"
                            f"<b>{r['feature_name']}</b> <span style='color:#EF4444;'>{r['impact_value']:.1f} pts</span>"
                            f"<br><span style='color:#94A3B8;font-size:13px;'>{r['friendly_text']}</span></div>",
                            unsafe_allow_html=True,
                        )
                    if not risks:
                        st.caption("No distress signals detected.")

                st.info(f"💡 **Actionable Guidance:** {data['actionable_guidance']}")

                # --- Point 5: counterfactual "path to a better score" ---
                paths = data.get('counterfactual_paths', [])
                if paths:
                    st.markdown("##### 🧭 Path to a Better Score")
                    st.caption("Each lever is re-scored on the real model — projected gains, not guesses.")
                    for p in paths:
                        tier_badge = " · unlocks a higher tier" if p.get('crosses_tier') else ""
                        st.markdown(
                            f"<div style='padding:8px 10px;margin-bottom:6px;border-left:3px solid #1E88E5;"
                            f"background:rgba(30,136,229,0.08);border-radius:4px;'>"
                            f"<b>{p['label']}</b> <span style='color:#1E88E5;'>+{p['projected_score_gain']} pts → {p['projected_score']}</span>{tier_badge}"
                            f"<br><span style='color:#94A3B8;font-size:13px;'>Move from {p['current_value']} toward {p['target_value']}.</span></div>",
                            unsafe_allow_html=True,
                        )

            st.divider()
            st.markdown("##### 🏦 OCEN 4.0 Pre-Matched Loan Products")
            st.caption("Products this profile currently qualifies for, ranked by rate.")

            matches = get_top_matches(score)
            if matches:
                mcols = st.columns(len(matches))
                for mcol, m in zip(mcols, matches):
                    with mcol:
                        with st.container(border=True):
                            st.markdown(f"**{m['lender_name']}**")
                            st.caption(m['loan_type'])
                            st.markdown(f"<span style='font-size:26px; font-weight:700; color:#1E88E5;'>{m['interest_rate_pa']}%</span> <span style='color:#888;'>p.a.</span>", unsafe_allow_html=True)
                            st.markdown(f"Limit: **₹{m['max_amount_inr']:,.0f}**")
                            st.caption(f"Tenure {m['tenure_months']} mo · {m['product_id']}")
            else:
                st.warning("No pre-matched products at this score. Improving the risk flags above unlocks eligibility.")

else:
    st.error(f"🔌 System Alert: {options.error['message']}")
    st.info("The backend scoring engine might be waking up. Please refresh in a few moments.")