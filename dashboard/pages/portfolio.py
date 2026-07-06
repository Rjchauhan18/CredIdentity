import streamlit as st
import pandas as pd
import os

st.title("📈 Executive Portfolio Overview")
st.caption("Aggregated risk metrics across processed New-to-Credit (NTC) MSMEs")
st.divider()

csv_path = "data/processed/msme_engineered_features.csv"

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
    st.info("Pipeline data not found. Please run `python data_pipeline/engineer_features.py` to populate portfolio metrics.")