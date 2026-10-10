"""Black-Scholes-Merton pricer with Greeks and a put-call parity check."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app_utils import compute, theme, ui
from app_utils.text import GREEK_HELP
from optionslab import bs_greeks, bs_price, put_call_parity_gap

ui.setup("BSM Pricer", "🧮", "European call and put under Black-Scholes-Merton with continuous dividend yield.")
m = ui.market_inputs("pricer")
cur = m.preset.currency

call = float(bs_price(m.S, m.K, m.T, m.r, m.q, m.sigma, "call"))
put = float(bs_price(m.S, m.K, m.T, m.r, m.q, m.sigma, "put"))
gc = bs_greeks(m.S, m.K, m.T, m.r, m.q, m.sigma, "call").display_units()
gp = bs_greeks(m.S, m.K, m.T, m.r, m.q, m.sigma, "put").display_units()


def greek_row(g, label: str, price: float, kind: str) -> None:
    intrinsic = max(m.S - m.K, 0.0) if kind == "call" else max(m.K - m.S, 0.0)
    st.markdown(f"##### {label}")
    c = st.columns(6)
    c[0].metric("Price", ui.money(price, cur, 4 if m.S < 1000 else 2),
                help=f"Intrinsic {ui.money(intrinsic, cur)} + time value {ui.money(price - intrinsic, cur)}")
    c[1].metric("Delta", f"{g.delta:.4f}", help=GREEK_HELP["Delta"])
    c[2].metric("Gamma", f"{g.gamma:.5f}", help=GREEK_HELP["Gamma"])
    c[3].metric("Vega (per 1 vol pt)", f"{g.vega:.4f}", help=GREEK_HELP["Vega"])
    c[4].metric("Theta (per day)", f"{g.theta:.4f}", help=GREEK_HELP["Theta"])
    c[5].metric("Rho (per 1 rate pt)", f"{g.rho:.4f}", help=GREEK_HELP["Rho"])


greek_row(gc, "Call", call, "call")
greek_row(gp, "Put", put, "put")

st.subheader("Put-call parity check")
pv_s, pv_k = m.S * np.exp(-m.q * m.T), m.K * np.exp(-m.r * m.T)
gap = float(put_call_parity_gap(call, put, m.S, m.K, m.T, m.r, m.q))
p1, p2, p3, p4 = st.columns(4)
p1.metric("Call - Put", f"{call - put:,.6f}")
p2.metric("S·e^(-qT) - K·e^(-rT)", f"{pv_s - pv_k:,.6f}")
p3.metric("Gap", f"{gap:.2e}")
with p4:
    if abs(gap) < 1e-8 * max(1.0, m.S):
        st.success("Parity holds (|gap| < 1e-8 × S)")
    else:
        st.error("Parity violated - check inputs")
st.latex(r"C - P = S e^{-qT} - K e^{-rT}")

with st.expander("Test parity on your own market quotes"):
    st.caption("Enter a call and put quoted at the same strike and expiry. A gap larger than bid-ask "
               "spreads and carry costs hints at an arbitrage or stale quote.")
    q1, q2 = st.columns(2)
    mc = q1.number_input("Market call price", min_value=0.0, value=round(call, 2), step=0.05, key="par_c")
    mp = q2.number_input("Market put price", min_value=0.0, value=round(put, 2), step=0.05, key="par_p")
    g2 = float(put_call_parity_gap(mc, mp, m.S, m.K, m.T, m.r, m.q))
    st.metric("Parity gap on your quotes", ui.money(g2, cur, 4))

if m.lot_size:
    st.subheader(f"Contract view - lot size {m.lot_size:,}")
    l1, l2, l3, l4 = st.columns(4)
    l1.metric("Call premium per lot", ui.money(call * m.lot_size, cur))
    l2.metric("Put premium per lot", ui.money(put * m.lot_size, cur))
    l3.metric("Notional per lot", ui.money(m.S * m.lot_size, cur, 0))
    l4.metric("Call delta-equivalent exposure", ui.money(gc.delta * m.S * m.lot_size, cur, 0),
              help="Delta × spot × lot size: the index exposure the position behaves like.")

st.subheader("Value vs spot")
grid = np.linspace(0.6 * m.S, 1.4 * m.S, 161)
fig = go.Figure()
for kind, col in (("call", theme.TEAL), ("put", theme.ORANGE)):
    fig.add_scatter(x=grid, y=bs_price(grid, m.K, m.T, m.r, m.q, m.sigma, kind), name=f"{kind.title()} (BSM today)",
                    line=dict(color=col, width=3))
    pay = np.maximum(grid - m.K, 0) if kind == "call" else np.maximum(m.K - grid, 0)
    fig.add_scatter(x=grid, y=pay, name=f"{kind.title()} (payoff at expiry)", line=dict(color=col, width=1.5, dash="dot"))
theme.vref(fig, m.S, "Spot")
theme.style_fig(fig, x=f"Spot {cur}".strip(), y="Option value", height=400)
st.plotly_chart(fig, use_container_width=True)
ui.footer()
