import plotly.graph_objects as go
import pandas as pd

# Shared palette (kept consistent with the health-card score colors)
GOOD = "#22C55E"
WARN = "#F59E0B"
BAD = "#EF4444"
BRAND = "#3B82F6"
MUTED = "#94A3B8"
TRANSPARENT = "rgba(0,0,0,0)"

# Common layout applied to every chart so they blend into the Streamlit theme
_BASE_LAYOUT = dict(
    paper_bgcolor=TRANSPARENT,
    plot_bgcolor=TRANSPARENT,
    font=dict(family="Inter, Segoe UI, sans-serif", size=13, color="#CBD5E1"),
)


def _score_color(score):
    if score >= 750:
        return GOOD
    if score >= 600:
        return WARN
    return BAD


def create_score_gauge(score, tier):
    """Credit-score gauge (300-900) — a far cleaner hero than a bare number."""
    color = _score_color(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(font=dict(size=46, color=color)),
        title=dict(text=f"<span style='font-size:15px;color:{MUTED}'>{tier}</span>", font=dict(size=15)),
        gauge=dict(
            axis=dict(range=[300, 900], tickwidth=1, tickcolor=MUTED, tickfont=dict(size=10)),
            bar=dict(color=color, thickness=0.28),
            bgcolor=TRANSPARENT,
            borderwidth=0,
            steps=[
                dict(range=[300, 550], color="rgba(239,68,68,0.15)"),
                dict(range=[550, 650], color="rgba(245,158,11,0.15)"),
                dict(range=[650, 750], color="rgba(245,158,11,0.22)"),
                dict(range=[750, 900], color="rgba(34,197,94,0.18)"),
            ],
            threshold=dict(line=dict(color=color, width=3), thickness=0.75, value=score),
        ),
    ))
    fig.update_layout(height=250, margin=dict(t=45, b=10, l=25, r=25), **_BASE_LAYOUT)
    return fig


def create_radar_chart(pillars):
    categories = [
        'Cash Flow<br>Resiliency',
        'Commercial<br>Activity',
        'Operational<br>Stability',
        'Compliance<br>& Risk'
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
        fillcolor='rgba(59, 130, 246, 0.22)',
        line=dict(color=BRAND, width=2),
        marker=dict(size=6, color=BRAND),
        hovertemplate='%{theta}: %{r}/100<extra></extra>'
    ))
    fig.update_layout(
        polar=dict(
            bgcolor=TRANSPARENT,
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9, color=MUTED),
                            gridcolor="rgba(148,163,184,0.2)", linecolor="rgba(148,163,184,0.2)"),
            angularaxis=dict(tickfont=dict(size=11), gridcolor="rgba(148,163,184,0.2)"),
        ),
        margin=dict(t=40, b=40, l=50, r=50),
        height=340,
        showlegend=False,
        **_BASE_LAYOUT
    )
    return fig


def create_driver_chart(strengths, risks):
    """Horizontal bar of the top drivers.

    Magnitudes are the model's REAL global permutation importance (in ROC-AUC points);
    strengths point right (favorable), risks point left (unfavorable).
    """
    data = [{"Feature": s['feature_name'], "Impact": s['impact_value']} for s in strengths]
    data += [{"Feature": r['feature_name'], "Impact": r['impact_value']} for r in risks]

    if not data:
        return go.Figure()

    df = pd.DataFrame(data).sort_values(by="Impact")
    colors = [BAD if x < 0 else GOOD for x in df['Impact']]

    fig = go.Figure(go.Bar(
        x=df['Impact'],
        y=df['Feature'],
        orientation='h',
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:+.2f}" for v in df['Impact']],
        textposition='outside',
        textfont=dict(size=11),
        hovertemplate='%{y}: %{x:+.2f} pts<extra></extra>',
        cliponaxis=False,
    ))
    fig.update_layout(
        title=dict(text="Feature Importance <span style='color:#94A3B8;font-size:12px'>(permutation · ROC-AUC pts)</span>",
                   font=dict(size=14)),
        xaxis=dict(title="favorable (→) vs. unfavorable (←)", zeroline=True,
                   zerolinecolor="rgba(148,163,184,0.4)", gridcolor="rgba(148,163,184,0.12)",
                   title_font=dict(size=11, color=MUTED)),
        yaxis=dict(gridcolor=TRANSPARENT),
        margin=dict(l=10, r=40, t=45, b=35),
        height=320,
        bargap=0.35,
        **_BASE_LAYOUT
    )
    return fig


# Backwards-compatible alias (drivers now come from permutation importance, not SHAP)
create_shap_waterfall = create_driver_chart