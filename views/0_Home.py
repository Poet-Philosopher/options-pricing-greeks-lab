"""Home page (rendered through the router in app.py)."""
import streamlit as st

from app_utils import ui

ui.setup("Options Pricing & Greeks Lab", "📈",
         "A quantitative toolkit for pricing, risk and strategy analysis of options - built on a tested, "
         "pure-Python pricing library.")

st.markdown(
    "Explore the Black-Scholes-Merton model, its Greeks, multi-leg strategies, implied volatility, "
    "Monte Carlo and binomial-tree pricing - with an India (Nifty 50) context. "
    "Use the **sidebar** to open each tool; inputs you set on one page carry over to the next."
)
ui.assumption_note("<b>Assumptions, not data.</b> Every default rate, volatility and price in this app is an "
                   "illustrative, editable assumption. No market data is fetched or implied.")
st.write("")

cards = [
    ("views/1_BSM_Pricer.py", "🧮", "BSM Pricer", "European call/put with dividend yield, five Greeks and a live put-call parity check."),
    ("views/2_Greeks_Explorer.py", "📊", "Greeks Explorer", "Interactive Greek curves vs spot, time to expiry and volatility."),
    ("views/3_Strategy_Builder.py", "🧱", "Strategy Builder", "Multi-leg payoff and P&L today vs expiry; breakevens, max profit/loss."),
    ("views/4_Implied_Vol.py", "🎯", "Implied Volatility", "Newton-Raphson with Brent fallback, edge-case diagnostics and a smile view."),
    ("views/5_Monte_Carlo.py", "🎲", "Monte Carlo", "Antithetic GBM, confidence intervals, convergence, path fans and Asian options."),
    ("views/6_Binomial_Tree.py", "🌳", "Binomial Tree", "CRR American vs European, convergence to BSM and early-exercise premium."),
    ("views/7_Methodology.py", "📚", "Methodology", "Formulas, conventions, algorithms, assumptions and limitations."),
]
cols = st.columns(3)
for i, (path, icon, title, desc) in enumerate(cards):
    with cols[i % 3]:
        with st.container(border=True):
            st.markdown(f"#### {icon} {title}")
            st.caption(desc)
            st.page_link(path, label=f"Open {title}", icon="➡️")

st.write("")
left, right = st.columns(2)
with left:
    st.subheader("Engineering")
    st.markdown(
        "- `optionslab/` - pure-Python package (type hints, docstrings, **no Streamlit imports**)\n"
        "- Vectorised NumPy / SciPy core; `@st.cache_data` on heavy calls\n"
        "- `pytest` suite: textbook values, parity, finite-difference Greeks, IV round-trips, "
        "MC confidence-interval coverage, CRR convergence\n"
        "- Input validation with explicit error messages for every edge case"
    )
with right:
    st.subheader("Conventions")
    st.markdown(
        "- Rates and dividend yield are continuously compounded; time is Act/365\n"
        "- **Vega** per 1 vol point, **Theta** per calendar day, **Rho** per 1 rate point\n"
        "- Black-Scholes-Merton prices **European** options only; the tree handles American exercise\n"
        "- See **Methodology** for formulas and limitations"
    )
ui.footer()
