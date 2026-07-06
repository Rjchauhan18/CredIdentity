import streamlit as st

st.set_page_config(page_title="IDBI MSME CredIdentity", layout="wide", page_icon="🏦")

page_1 = st.Page("pages/health_card.py", title="MSME Health Card", icon="🏢", default=True)
page_2 = st.Page("pages/loan_officer.py", title="Underwriting Desk", icon="🛡️")
page_3 = st.Page("pages/portfolio.py", title="Executive Portfolio", icon="📈")

pg = st.navigation([page_1, page_2, page_3], position="hidden")

st.sidebar.title("🏦 IDBI CredIdentity")
st.sidebar.caption("AI-Driven Financial Health Card")
st.sidebar.divider()

st.sidebar.page_link(page_1, label="MSME Health Card", icon="🏢")
st.sidebar.page_link(page_2, label="Underwriting Desk", icon="🛡️")
st.sidebar.page_link(page_3, label="Executive Portfolio", icon="📈")

st.sidebar.divider()
pg.run()