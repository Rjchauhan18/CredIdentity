import plotly.graph_objects as go
import pandas as pd

def create_radar_chart(pillars):
    categories = [
        'Cash Flow Resiliency', 
        'Commercial Activity', 
        'Operational Stability', 
        'Compliance & Risk'
    ]
    values = [
        pillars['cash_flow_resiliency'], 
        pillars['commercial_activity_revenue'], 
        pillars['operational_employee_stability'], 
        pillars['compliance_risk']
    ]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(30, 136, 229, 0.2)',
        line_color='#1E88E5'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        margin=dict(t=30, b=30, l=30, r=30),
        height=320
    )
    return fig

def create_shap_waterfall(strengths, risks):
    data = []
    for s in strengths:
        data.append({"Feature": s['feature_name'], "Impact": s['impact_value']})
    for r in risks:
        data.append({"Feature": r['feature_name'], "Impact": r['impact_value']})
        
    df = pd.DataFrame(data).sort_values(by="Impact")
    colors = ['#EF5350' if x < 0 else '#66BB6A' for x in df['Impact']]
    
    fig = go.Figure(go.Bar(
        x=df['Impact'],
        y=df['Feature'],
        orientation='h',
        marker_color=colors
    ))
    fig.update_layout(
        title="SHAP Feature Contribution Quantities",
        margin=dict(l=20, r=20, t=40, b=20),
        height=300
    )
    return fig