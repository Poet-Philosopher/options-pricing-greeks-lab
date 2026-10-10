"""Options Pricing & Greeks Lab - Streamlit entrypoint (multipage router).

Run with:  streamlit run app.py
Pages live in views/ so that navigation titles and URLs are explicit.
"""
import streamlit as st

PAGES = [
    st.Page("views/0_Home.py", title="Home", icon="🏠", default=True),
    st.Page("views/1_BSM_Pricer.py", title="BSM Pricer", icon="🧮", url_path="BSM_Pricer"),
    st.Page("views/2_Greeks_Explorer.py", title="Greeks Explorer", icon="📊", url_path="Greeks_Explorer"),
    st.Page("views/3_Strategy_Builder.py", title="Strategy Builder", icon="🧱", url_path="Strategy_Builder"),
    st.Page("views/4_Implied_Vol.py", title="Implied Volatility", icon="🎯", url_path="Implied_Vol"),
    st.Page("views/5_Monte_Carlo.py", title="Monte Carlo", icon="🎲", url_path="Monte_Carlo"),
    st.Page("views/6_Binomial_Tree.py", title="Binomial Tree", icon="🌳", url_path="Binomial_Tree"),
    st.Page("views/7_Methodology.py", title="Methodology", icon="📚", url_path="Methodology"),
]
st.navigation(PAGES).run()
