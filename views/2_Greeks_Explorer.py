"""Interactive Greek curves vs spot, time to expiry and volatility."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app_utils import compute, theme, ui
from app_utils.text import GREEK_HELP, GREEK_UNITS

ui.setup("Greeks Explorer", "📊", "See how each Greek moves with spot, time and volatility.")
m = ui.market_inputs("greeks")
cur = m.preset.currency

c1, c2 = st.columns([1, 2])
kind = c1.radio("Option type", ["call", "put"], horizontal=True, key="gx_kind", format_func=str.title)
greek = c2.radio("Greek", list(GREEK_HELP), horizontal=True, key="gx_greek", help="Choose which sensitivity to plot.")
st.info(f"**{greek}** ({GREEK_UNITS[greek]}): {GREEK_HELP[greek]}")
color = theme.KIND_COLOR[kind]


def curve_fig(x_name: str, xs: np.ndarray, current: float, xlabel: str, scale: float = 1.0) -> go.Figure:
    df = compute.greek_curve(x_name, tuple(xs.tolist()), m.S, m.K, m.T, m.r, m.q, m.sigma, kind)
    fig = go.Figure()
    fig.add_scatter(x=df["x"] * scale, y=df[greek], name=greek, line=dict(color=color, width=3))
    theme.vref(fig, current * scale, "Current")
    return theme.style_fig(fig, x=xlabel, y=f"{greek} ({GREEK_UNITS[greek]})", height=430)


tab_s, tab_t, tab_v, tab_all = st.tabs(["vs Spot", "vs Time to expiry", "vs Volatility", "All Greeks vs spot"])

with tab_s:
    spots = np.linspace(0.5 * m.S, 1.5 * m.S, 201)
    overlay = st.checkbox("Overlay other maturities", value=True, key="gx_overlay",
                          help="Shows how the curve sharpens as expiry approaches (e.g. gamma spikes at-the-money).")
    fig = curve_fig("spot", spots, m.S, f"Spot {cur}".strip())
    if overlay:
        fig.data[0].name = f"{m.days}d (current)"
        palette = [theme.INDIGO, theme.SLATE, theme.ORANGE]
        others = sorted({max(1, round(m.days * f)) for f in (0.25, 0.5, 2.0)} - {m.days})
        for d, col in zip(others, palette):
            df = compute.greek_curve("spot", tuple(spots.tolist()), m.S, m.K, d / 365.0, m.r, m.q, m.sigma, kind)
            fig.add_scatter(x=df["x"], y=df[greek], name=f"{d}d", line=dict(color=col, width=1.8, dash="dot"))
    theme.vref(fig, m.K, "Strike", dash="dot")
    st.plotly_chart(fig, use_container_width=True)

with tab_t:
    max_days = max(60, 2 * m.days)
    days = np.linspace(1, max_days, 200)
    st.plotly_chart(curve_fig("days", days, float(m.days), "Days to expiry"), use_container_width=True)
    st.caption("Read right-to-left to watch the option age toward expiry.")

with tab_v:
    vols = np.linspace(0.05, 0.8, 200)
    st.plotly_chart(curve_fig("vol", vols, m.sigma, "Volatility (%)", scale=100.0), use_container_width=True)

with tab_all:
    spots = np.linspace(0.5 * m.S, 1.5 * m.S, 161)
    names = list(GREEK_HELP)
    fig = make_subplots(rows=2, cols=3, subplot_titles=[f"{n} ({GREEK_UNITS[n]})" for n in names] + ["Price"],
                        horizontal_spacing=0.07, vertical_spacing=0.16)
    for k, col in (("call", theme.TEAL), ("put", theme.ORANGE)):
        df = compute.greek_curve("spot", tuple(spots.tolist()), m.S, m.K, m.T, m.r, m.q, m.sigma, k)
        for i, n in enumerate(names + ["Price"]):
            fig.add_scatter(x=df["x"], y=df[n], name=k.title(), line=dict(color=col, width=2.4),
                            showlegend=(i == 0), row=i // 3 + 1, col=i % 3 + 1, legendgroup=k)
    for i in range(6):
        fig.add_vline(x=m.S, line=dict(color=theme.SLATE, dash="dash", width=1), row=i // 3 + 1, col=i % 3 + 1)
    theme.style_fig(fig, height=620)
    fig.update_layout(hovermode="x")
    st.plotly_chart(fig, use_container_width=True)
ui.footer()
