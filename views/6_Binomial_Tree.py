"""Cox-Ross-Rubinstein binomial tree: American vs European, convergence, early exercise."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import optionslab as ol
from app_utils import compute, theme, ui

ui.setup("Binomial Tree (CRR)", "🌳", "Price American and European options on a recombining tree and watch it converge to Black-Scholes-Merton.")
m = ui.market_inputs("tree")
cur = m.preset.currency

sb = st.sidebar
sb.header("Tree")
kind = sb.radio("Option type", ["put", "call"], horizontal=True, key="tree_kind", format_func=str.title)
n = sb.slider("Steps (N)", 10, 1500, 300, 10, key="tree_n", help="More steps = closer to the continuous-time price.")

try:
    eu = ol.crr_tree(m.S, m.K, m.T, m.r, m.q, m.sigma, n, kind, american=False)
    am = ol.crr_tree(m.S, m.K, m.T, m.r, m.q, m.sigma, n, kind, american=True)
except ValueError as exc:
    st.error(str(exc))
    st.stop()
bsm = float(ol.bs_price(m.S, m.K, m.T, m.r, m.q, m.sigma, kind))
eep = am.price - eu.price

a, b, c, d = st.columns(4)
a.metric("European (tree)", ui.money(eu.price, cur, 4), delta=f"{eu.price - bsm:+.5f} vs BSM", delta_color="off")
b.metric("American (tree)", ui.money(am.price, cur, 4))
c.metric("Black-Scholes-Merton", ui.money(bsm, cur, 4), help="European, analytic.")
d.metric("Early-exercise premium", ui.money(eep, cur, 4), help="American minus European price on the same tree.")
if kind == "call" and m.q == 0:
    st.info("With no dividends, early exercise of an American call is never optimal, so its premium is zero.")
elif eep < 1e-9:
    st.info("The early-exercise premium is ~0 here: holding the option dominates exercising for these inputs.")
else:
    st.success(f"Early exercise adds {eep:,.4f} ({eep / am.price:.1%} of the American price): the holder can do better by "
               "exercising before expiry in some states.")
g = ol.bs_greeks(m.S, m.K, m.T, m.r, m.q, m.sigma, kind)
st.caption(f"Tree parameters: u = {eu.u:.5f}, d = {eu.d:.5f}, risk-neutral p = {eu.p:.5f}. "
           f"European tree delta {eu.delta:.4f} (BSM {float(g.delta):.4f}), gamma {eu.gamma:.5f} (BSM {float(g.gamma):.5f}).")

steps = tuple(sorted({s for s in (10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1000) if s <= max(n, 100)} | {n}))
conv = compute.crr_convergence(m.S, m.K, m.T, m.r, m.q, m.sigma, steps, kind)

left, right = st.columns(2)
fig = go.Figure()
fig.add_scatter(x=conv.steps, y=conv.european, name="European (CRR)", mode="lines+markers", line=dict(color=theme.TEAL, width=2.5))
fig.add_scatter(x=conv.steps, y=conv.american, name="American (CRR)", mode="lines+markers", line=dict(color=theme.ORANGE, width=2.5))
theme.href(fig, bsm, "BSM", theme.INDIGO)
theme.style_fig(fig, "Convergence in the number of steps", "Steps N (log scale)", f"Price {cur}".strip(), height=400)
theme.log_axis(fig, "x", [10, 20, 50, 100, 200, 500, 1000])
left.plotly_chart(fig, use_container_width=True)

err = conv.euro_error.abs().clip(lower=1e-12)
fig2 = go.Figure()
fig2.add_scatter(x=conv.steps, y=err, name="|European CRR - BSM|", mode="lines+markers", line=dict(color=theme.TEAL, width=2.5))
fig2.add_scatter(x=conv.steps, y=err.iloc[0] * conv.steps.iloc[0] / conv.steps, name="1/N reference", line=dict(color=theme.SLATE, dash="dot"))
theme.style_fig(fig2, "Pricing error vs BSM (log-log)", "Steps N", "Absolute error", height=400)
theme.log_axis(fig2, "x", [10, 20, 50, 100, 200, 500, 1000])
theme.log_axis(fig2, "y", *theme.decades(err))
right.plotly_chart(fig2, use_container_width=True)
st.caption("The error decays roughly like 1/N but zig-zags: CRR is sensitive to where the strike falls relative to the final layer of nodes.")

st.subheader("Where does early exercise matter?")
spots = tuple(np.round(np.linspace(0.6 * m.K, 1.4 * m.K, 41), 4).tolist())
eep_df = compute.eep_curve(spots, m.K, m.T, m.r, m.q, m.sigma, min(n, 300), kind)
fig3 = make_subplots(rows=1, cols=2, subplot_titles=("Option value vs spot", "Early-exercise premium vs spot"), horizontal_spacing=0.08)
fig3.add_scatter(x=eep_df.spot, y=eep_df.american, name="American", line=dict(color=theme.ORANGE, width=3), row=1, col=1)
fig3.add_scatter(x=eep_df.spot, y=eep_df.european, name="European", line=dict(color=theme.TEAL, width=3), row=1, col=1)
fig3.add_scatter(x=eep_df.spot, y=eep_df.intrinsic, name="Intrinsic", line=dict(color=theme.SLATE, width=1.5, dash="dot"), row=1, col=1)
fig3.add_scatter(x=eep_df.spot, y=eep_df.premium, name="Premium", line=dict(color=theme.INDIGO, width=3), fill="tozeroy",
                 fillcolor="rgba(79,70,229,0.12)", showlegend=False, row=1, col=2)
for col in (1, 2):
    fig3.add_vline(x=m.S, line=dict(color=theme.SLATE, dash="dash", width=1), row=1, col=col)
theme.style_fig(fig3, height=400)
fig3.update_layout(hovermode="x")
st.plotly_chart(fig3, use_container_width=True)
st.caption("Deep in-the-money American puts hit the intrinsic-value line: it is optimal to exercise and earn interest on the strike. "
           f"(Curves use N = {min(n, 300)}.)")
ui.footer()
