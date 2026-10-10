"""Implied-volatility solver and volatility-smile view."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import optionslab as ol
from app_utils import theme, ui

ui.setup("Implied Volatility", "🎯", "Back out the volatility that reproduces an option price - and view a smile across strikes.")
m = ui.market_inputs("iv", vol=False)
cur = m.preset.currency
preset_vol = m.preset.vol_pct / 100.0

tab1, tab2 = st.tabs(["Single option", "Volatility smile"])

# ------------------------------------------------------------------ single solve
with tab1:
    c1, c2 = st.columns([1, 2])
    kind = c1.radio("Option type", ["call", "put"], horizontal=True, key="iv_kind", format_func=str.title)
    default_price = round(float(ol.bs_price(m.S, m.K, m.T, m.r, m.q, preset_vol, kind)), 2)
    price = c1.number_input(f"Observed option price {cur}".strip(), min_value=0.0, step=0.05,
                            value=default_price, key=f"iv_price_{m.preset.name}_{kind}", format="%.4f",
                            help="Default is the BSM price at the preset volatility (an assumption) - overwrite with a quote.")
    res = ol.implied_vol(price, m.S, m.K, m.T, m.r, m.q, kind)
    lower, upper = ol.no_arbitrage_bounds(m.S, m.K, m.T, m.r, m.q, kind)
    c1.caption(f"No-arbitrage range for this option: **{lower:,.4f} - {upper:,.4f}**")

    with c2:
        if res.converged:
            a, b, c = st.columns(3)
            a.metric("Implied volatility", f"{res.iv * 100:.4f}%")
            b.metric("Method", {"newton": "Newton-Raphson", "brentq": "Brent (fallback)", "bound": "Lower bound"}[res.method])
            c.metric("Iterations", res.iterations)
            repriced = float(ol.bs_price(m.S, m.K, m.T, m.r, m.q, res.iv, kind))
            st.success(f"{res.message}  Re-priced option = {repriced:,.6f} (error {abs(repriced - price):.2e}).")
            for w in res.warnings:
                st.warning(w)
        else:
            st.error(f"**{res.status.replace('_', ' ').capitalize()}** - {res.message}")

    sig_grid = np.linspace(0.01, 1.5, 300)
    fig = go.Figure()
    fig.add_scatter(x=sig_grid * 100, y=ol.bs_price(m.S, m.K, m.T, m.r, m.q, sig_grid, kind), name="BSM price",
                    line=dict(color=theme.TEAL, width=3))
    theme.href(fig, price, "Observed price", theme.ORANGE)
    if res.converged and res.iv is not None:
        theme.vref(fig, res.iv * 100, "Implied vol", theme.INDIGO, "dot")
        fig.add_scatter(x=[res.iv * 100], y=[price], mode="markers", name="Solution",
                        marker=dict(size=12, color=theme.INDIGO))
    theme.style_fig(fig, "Price as a function of volatility (monotonic, so the root is unique)", "Volatility (%)",
                    f"Option price {cur}".strip(), height=360)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("How the solver handles edge cases"):
        st.markdown(
            "- **Price below discounted intrinsic / above the upper bound** -> reported as an arbitrage or bad quote; no solve attempted.\n"
            "- **Zero or negative price** -> rejected.\n"
            "- **Newton-Raphson** starts from a Brenner-Subrahmanyam seed using analytic vega; if vega vanishes or a step leaves "
            "[0.0001%, 1000%], it **falls back to Brent's bracketing method**.\n"
            "- **Deep ITM/OTM or tiny vega** -> still solved if possible, but flagged as poorly determined.\n"
            "- **IV outside [1e-6, 10]** -> explicit message instead of a silent wrong answer."
        )

# ------------------------------------------------------------------ smile
with tab2:
    st.markdown("Enter or paste option quotes for **one expiry** (the sidebar's days-to-expiry, rate and dividend yield apply). "
                "One line per quote: `strike, price, call|put`.")
    ui.assumption_note("The pre-loaded quotes are <b>synthetic</b>: priced by code from a made-up skew curve so you can "
                       "see the tool work. They are not market data - replace them with your own quotes.")

    def synthetic_text() -> str:
        ks = sorted({ol.round_to_step(m.S * f, m.strike_step) for f in np.arange(0.88, 1.1201, 0.02)})
        df = ol.synthetic_smile_example(m.S, m.T, m.r, m.q, ks, base_vol=preset_vol)
        return "strike, price, type\n" + "\n".join(f"{r.strike:g}, {r.price:.2f}, {r.kind}" for r in df.itertuples())

    sig = (m.preset.name, round(m.S, 2), m.days)
    if st.session_state.get("iv_sig") != sig or "iv_text" not in st.session_state:
        st.session_state["iv_text"] = synthetic_text()
        st.session_state["iv_sig"] = sig
    if st.button("Load synthetic example at current inputs"):
        st.session_state["iv_text"] = synthetic_text()
    mode = st.radio("Input mode", ["Paste / type text", "Edit as table"], horizontal=True, key="iv_mode")
    if mode == "Paste / type text":
        text = st.text_area("Quotes", key="iv_text", height=220)
        quotes, parse_errors = ol.parse_quotes(text)
    else:
        seed_df, parse_errors = ol.parse_quotes(st.session_state["iv_text"])
        quotes = st.data_editor(seed_df, num_rows="dynamic", hide_index=True, use_container_width=True, key=f"iv_editor_{hash(st.session_state['iv_text'])}",
                                column_config={"kind": st.column_config.SelectboxColumn("kind", options=["call", "put"], required=True)})
        quotes = quotes.dropna()
    for e in parse_errors:
        st.warning(e)

    if quotes.empty:
        st.info("Add at least one quote to see the smile.")
    else:
        df = ol.implied_vol_smile(quotes["strike"].tolist(), quotes["price"].tolist(), quotes["kind"].tolist(), m.S, m.T, m.r, m.q)
        df = df.sort_values("strike").reset_index(drop=True)
        xaxis = st.radio("X-axis", ["Strike", "Moneyness (K/S)", "Log-moneyness ln(K/F)"], horizontal=True, key="iv_x")
        fwd = m.S * np.exp((m.r - m.q) * m.T)
        xs = {"Strike": df["strike"], "Moneyness (K/S)": df["moneyness"], "Log-moneyness ln(K/F)": np.log(df["strike"] / fwd)}[xaxis]
        ok = df["iv"].notna()
        fig = go.Figure()
        fig.add_scatter(x=xs[ok], y=df.loc[ok, "iv"] * 100, mode="lines", line=dict(color=theme.SLATE, width=1.5), name="Smile", hoverinfo="skip")
        for k, col in (("call", theme.TEAL), ("put", theme.ORANGE)):
            sel = ok & (df["kind"] == k)
            fig.add_scatter(x=xs[sel], y=df.loc[sel, "iv"] * 100, mode="markers", name=f"{k.title()} quotes",
                            marker=dict(size=10, color=col), customdata=df.loc[sel, ["price"]],
                            hovertemplate="IV %{y:.2f}%<br>price %{customdata[0]:.2f}<extra></extra>")
        if (~ok).any():
            fig.add_scatter(x=xs[~ok], y=[float(np.nanmin(df["iv"] * 100)) if ok.any() else 0.0] * int((~ok).sum()), mode="markers",
                            name="No solution", marker=dict(size=11, color=theme.RED, symbol="x"))
        theme.style_fig(fig, "Implied-volatility smile", xaxis, "Implied volatility (%)", height=420)
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig, use_container_width=True)
        if (~ok).any():
            st.warning(f"{int((~ok).sum())} quote(s) could not be solved - see the Status column.")
        show = df.assign(IV_pct=(df["iv"] * 100).round(4))[["strike", "kind", "price", "IV_pct", "method", "status", "message"]]
        st.dataframe(show, hide_index=True, use_container_width=True)
        st.download_button("Download results (CSV)", show.to_csv(index=False).encode(), "implied_vol_smile.csv", "text/csv")
ui.footer()
