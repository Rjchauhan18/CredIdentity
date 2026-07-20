import streamlit as st
import pandas as pd
import os

from utils.api_client import get_validation_report

st.title("📈 Executive Portfolio Overview")
st.caption("Aggregated risk metrics across the served New-to-Credit (NTC) MSME cohort")
st.divider()

# Read the same demo cohort the API scores, so portfolio aggregates stay consistent
# with the individual health cards.
csv_path = "data/processed/msme_demo_features.csv"

if os.path.exists(csv_path):
    # Load the real generated data
    df = pd.read_csv(csv_path)
    
    col1, col2, col3 = st.columns(3)
    
    # -----------------------------------------------------------------
    # METRIC 1: TOTAL MSMEs EVALUATED
    # -----------------------------------------------------------------
    critical_risks = len(df[(df['bounce_mandate_failure_rate'] > 0.1) | (df['average_monthly_balance'] < 0)])
    healthy_count = len(df) - critical_risks
    
    col1.metric(
        label="Total MSMEs Evaluated", 
        value=f"{len(df):,}"
    )
    
    # -----------------------------------------------------------------
    # METRIC 2: AVG MONTHLY BALANCE
    # -----------------------------------------------------------------
    avg_balance = df['average_monthly_balance'].mean()
    avg_growth = df['monthly_revenue_growth_rate'].mean() * 100
    
    col2.metric(
        label="Avg Monthly Balance", 
        value=f"₹{avg_balance:,.0f}"
    )
    
    # -----------------------------------------------------------------
    # METRIC 3: SYSTEMIC BOUNCE RATE
    # -----------------------------------------------------------------
    avg_bounce_rate = df['bounce_mandate_failure_rate'].mean() * 100
    success_rate = 100 - avg_bounce_rate
    
    col3.metric(
        label="Systemic Bounce Rate", 
        value=f"{avg_bounce_rate:.1f}%"
    )

    st.subheader("Macro Liquidity Distribution")
    st.caption("Distribution of Balance Depletion Speed (Days of Runway)")
    
    # Clip extreme outliers for a cleaner chart
    runway_data = df['balance_depletion_speed_days'].clip(0, 60)
    
    # Convert Pandas Math Intervals to clean UI Strings
    binned_data = runway_data.value_counts(bins=10).sort_index()
    binned_data.index = [f"{int(interval.left)} - {int(interval.right)} days" for interval in binned_data.index]
    
    # Plot the newly formatted string labels
    st.bar_chart(binned_data, color="#1E88E5", height=300)

    st.subheader("Systemic Weakness Diagnostics")
    
    high_variance_count = len(df[df['gstr1_vs_gstr3b_variance'] > 50000])
    high_bounce_count = len(df[df['bounce_mandate_failure_rate'] > 0.1])
    
    if high_variance_count > high_bounce_count:
        st.warning(f"📊 **Trend Analysis:** GST Compliance registers as the highest systemic bottleneck. {high_variance_count} MSMEs flagged for high GSTR1 vs GSTR3B variance.")
    else:
        st.error(f"📊 **Trend Analysis:** Cash Flow Resiliency is under stress. {high_bounce_count} MSMEs flagged for mandate bounce rates exceeding 10%.")
else:
    st.info("Demo cohort not found. Please run `python -m data_pipeline.build_demo_dataset` to populate portfolio metrics.")

# ---------------------------------------------------------
# BUSINESS-OUTCOME BACKTEST (Point 3) — approval vs default tradeoff
# ---------------------------------------------------------
st.divider()
st.subheader("💼 Approval Policy Simulator")
st.caption("How the score cutoff trades approval volume against defaults, measured on a held-out labelled slice.")

report = get_validation_report()
backtest = report.get("backtest", []) if report else []

if not backtest:
    st.info("Backtest data not available yet. Generate it with `python -m scripts.build_validation_report`.")
else:
    bt_df = pd.DataFrame(backtest)
    cutoffs = bt_df["cutoff"].tolist()
    chosen = st.select_slider("Approval score cutoff", options=cutoffs, value=cutoffs[len(cutoffs) // 2])
    row = bt_df[bt_df["cutoff"] == chosen].iloc[0]

    b1, b2, b3 = st.columns(3)
    b1.metric("Approval Rate", f"{row['approval_rate'] * 100:.1f}%")
    b2.metric("Default Rate (Approved)", f"{row['bad_rate_among_approved'] * 100:.2f}%")
    b3.metric("Defaults Avoided", f"{int(row['defaults_avoided']):,}")
    st.caption(
        f"At a cutoff of **{chosen}**, the bank approves {row['approval_rate'] * 100:.1f}% of applicants "
        f"with a {row['bad_rate_among_approved'] * 100:.2f}% default rate among those approved."
    )

    st.markdown("**Approval rate vs default rate across all cutoffs**")
    curve = bt_df.set_index("cutoff")[["approval_rate", "bad_rate_among_approved"]]
    st.line_chart(curve, height=280)