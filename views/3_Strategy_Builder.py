"""Multi-leg strategy builder: payoff at expiry vs theoretical value today."""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import optionslab as ol
from app_utils import theme, ui

ui.setup("Strategy Builder", "🧱", "Build multi-leg positions; compare the expiry payoff with the theoretical P&L today.")
m = ui.market_inputs("strategy", strike=False)
cur = m.preset.currency
mult = st.sidebar.number_input("Contract multiplier (lot size)", min_value=1, step=1, value=int(m.lot_size or 1),
                               key=f"strategy_mult_{m.preset.name}",
                               help="P&L is scaled by this many units of the underlying (e.g. Nifty lot size).")

# ---------------------------------------------------------------- presets / legs
CUSTOM = "Custom (edit the table)"
top = st.columns([2, 1, 1, 1])
choice = top[0].selectbox("Strategy preset", [CUSTOM, *ol.PRESET_NAMES], index=2, key="sb_choice")
width = top[1].slider("Preset width (% of spot)", 2.0, 15.0, 5.0, 0.5, key="sb_width",
                      help="Distance of the first OTM strikes from spot; iron-condor wings sit at 2x this.")
step = top[2].number_input("Strike step", min_value=0.01, value=float(m.strike_step), step=float(max(m.strike_step / 5, 0.01)),
                           key=f"sb_step_{m.preset.name}", format="%.2f")
reset = top[3].button("Reset legs to preset", disabled=(choice == CUSTOM), use_container_width=True)


def legs_to_df(legs: list[ol.Leg]) -> pd.DataFrame:
    return pd.DataFrame({
        "Type": [l.kind for l in legs],
        "Position": ["long" if l.qty > 0 else "short" for l in legs],
        "Strike": [np.nan if l.kind == "stock" else l.strike for l in legs],
        "Qty": [abs(l.qty) for l in legs],
        "Premium": [np.nan if l.premium is None else l.premium for l in legs],
    })


sig = (choice, width, round(m.S, 2), step)
if "sb_df" not in st.session_state or reset or (choice != CUSTOM and st.session_state.get("sb_sig") != sig):
    if choice != CUSTOM:
        st.session_state["sb_df"] = legs_to_df(ol.preset_legs(choice, m.S, step, width / 100.0))
        st.session_state["sb_ver"] = st.session_state.get("sb_ver", 0) + 1
    elif "sb_df" not in st.session_state:
        st.session_state["sb_df"] = legs_to_df(ol.preset_legs("Straddle", m.S, step))
        st.session_state["sb_ver"] = 0
st.session_state["sb_sig"] = sig

st.markdown("**Legs** - leave *Premium* blank to use the theoretical Black-Scholes-Merton price (editable assumption: "
            "one volatility for all legs); enter a number to use your own quote. Add or delete rows freely.")
edited = st.data_editor(
    st.session_state["sb_df"], key=f"sb_editor_{st.session_state.get('sb_ver', 0)}", num_rows="dynamic",
    hide_index=True, use_container_width=True,
    column_config={
        "Type": st.column_config.SelectboxColumn("Type", options=["call", "put", "stock"], required=True),
        "Position": st.column_config.SelectboxColumn("Position", options=["long", "short"], required=True),
        "Strike": st.column_config.NumberColumn("Strike", min_value=0.01, format="%.2f", help="Ignored for stock legs."),
        "Qty": st.column_config.NumberColumn("Qty", min_value=1, step=1, default=1, help="Units per leg (multiplied by the contract multiplier in P&L)."),
        "Premium": st.column_config.NumberColumn("Premium (blank = theoretical)", min_value=0.0, format="%.2f"),
    },
)

legs: list[ol.Leg] = []
errors: list[str] = []
for i, row in edited.iterrows():
    if pd.isna(row["Type"]) and pd.isna(row["Strike"]):
        continue
    try:
        if pd.isna(row["Type"]) or pd.isna(row["Position"]):
            raise ValueError("choose a Type and Position")
        qty = float(row["Qty"]) if not pd.isna(row["Qty"]) else 1.0
        sign = 1.0 if row["Position"] == "long" else -1.0
        strike = 0.0 if row["Type"] == "stock" else (float(row["Strike"]) if not pd.isna(row["Strike"]) else 0.0)
        prem = None if pd.isna(row["Premium"]) else float(row["Premium"])
        legs.append(ol.Leg(row["Type"], sign * qty, strike, prem))
    except ValueError as exc:
        errors.append(f"Row {i + 1}: {exc}")
if errors:
    for e in errors:
        st.error(e)
    st.stop()
if not legs:
    st.warning("Add at least one leg to see the payoff.")
    st.stop()

# ---------------------------------------------------------------- analytics
summary = ol.analyze(legs, m.S, m.T, m.r, m.q, m.sigma)
cost = summary.entry_cost
strikes = [l.strike for l in legs if l.kind != "stock"]
lo = min([0.6 * m.S] + [0.85 * k for k in strikes])
hi = max([1.4 * m.S] + [1.15 * k for k in strikes])
grid = np.unique(np.concatenate([np.linspace(lo, hi, 500), strikes, list(summary.breakevens)]))
pnl_exp = ol.expiry_pnl(legs, grid, cost) * mult
pnl_today = ol.pnl_at(legs, grid, m.T, m.r, m.q, m.sigma, cost) * mult
elapsed = st.slider("Days elapsed (theoretical P&L after this many days)", 0, int(m.days), int(m.days // 2), key="sb_elapsed",
                    help="Uses Black-Scholes-Merton with the remaining time; spot-vol-rate held at the sidebar values.")

st.subheader("Summary at expiry")


def fmt_extreme(x: float) -> str:
    return "Unlimited" if math.isinf(x) else ui.money(x * mult, cur)


c = st.columns(5)
c[0].metric("Net debit" if cost > 0 else "Net credit", ui.money(abs(cost) * mult, cur),
            help="Cash paid (debit) or received (credit) to open the position, including the multiplier.")
c[1].metric("Max profit", fmt_extreme(summary.max_profit))
c[2].metric("Max loss", fmt_extreme(summary.max_loss))
if not math.isinf(summary.max_profit) and not math.isinf(summary.max_loss) and summary.max_loss < 0:
    c[3].metric("Reward : risk", f"{summary.max_profit / abs(summary.max_loss):.2f} : 1")
else:
    c[3].metric("Reward : risk", "n/a (unbounded)")
c[4].metric("Breakevens", len(summary.breakevens), help="Underlying prices at expiry where P&L is exactly zero (listed below).")
if summary.breakevens:
    st.markdown("**Breakeven at expiry:** " + "  |  ".join(f"**{b:,.2f}** ({b / m.S - 1:+.1%} vs spot)" for b in summary.breakevens))

fig = go.Figure()
fig.add_scatter(x=grid, y=np.maximum(pnl_exp, 0), fill="tozeroy", mode="lines", line=dict(width=0),
                fillcolor="rgba(22,163,74,0.14)", hoverinfo="skip", showlegend=False)
fig.add_scatter(x=grid, y=np.minimum(pnl_exp, 0), fill="tozeroy", mode="lines", line=dict(width=0),
                fillcolor="rgba(220,38,38,0.14)", hoverinfo="skip", showlegend=False)
fig.add_scatter(x=grid, y=pnl_exp, name="P&L at expiry", line=dict(color=theme.ORANGE, width=3))
fig.add_scatter(x=grid, y=pnl_today, name="Theoretical P&L today (T+0)", line=dict(color=theme.TEAL, width=3))
if 0 < elapsed < m.days:
    mid = ol.pnl_at(legs, grid, (m.days - elapsed) / 365.0, m.r, m.q, m.sigma, cost) * mult
    fig.add_scatter(x=grid, y=mid, name=f"Theoretical after {elapsed}d", line=dict(color=theme.INDIGO, width=2, dash="dot"))
if summary.breakevens:
    fig.add_scatter(x=list(summary.breakevens), y=[0] * len(summary.breakevens), mode="markers+text", name="Breakeven",
                    marker=dict(size=10, color="white", line=dict(color=theme.SLATE, width=2)),
                    text=[f"{b:,.1f}" for b in summary.breakevens], textposition="top center", hoverinfo="skip")
theme.vref(fig, m.S, "Spot")
theme.style_fig(fig, x=f"Underlying price at evaluation {cur}".strip(), y=f"P&L {cur}".strip(), height=480)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- details
left, right = st.columns([3, 2])
with left:
    st.markdown("**Resolved legs**")
    rows = []
    for l in legs:
        price = ol.entry_price(l, m.S, m.T, m.r, m.q, m.sigma)
        rows.append({"Leg": f"{'Long' if l.qty > 0 else 'Short'} {abs(l.qty):g} {l.kind}"
                            + ("" if l.kind == "stock" else f" @ {l.strike:,.2f}"),
                     "Entry price": round(price, 4),
                     "Source": "your quote" if l.premium is not None else ("spot" if l.kind == "stock" else "theoretical BSM"),
                     "Cash flow": round(-l.qty * price * mult, 2)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
with right:
    st.markdown("**Position Greeks today** (× multiplier)")
    tot = dict(Delta=0.0, Gamma=0.0, Vega=0.0, Theta=0.0, Rho=0.0)
    for l in legs:
        if l.kind == "stock":
            tot["Delta"] += l.qty
            continue
        g = ol.bs_greeks(m.S, l.strike, m.T, m.r, m.q, m.sigma, l.kind).display_units()
        for k, v in zip(tot, (g.delta, g.gamma, g.vega, g.theta, g.rho)):
            tot[k] += l.qty * float(v)
    st.dataframe(pd.DataFrame({"Greek": list(tot), "Value": [round(v * mult, 4) for v in tot.values()]}),
                 hide_index=True, use_container_width=True)
    st.caption("Vega per 1 vol point, theta per calendar day, rho per 1 rate point.")
ui.footer()
