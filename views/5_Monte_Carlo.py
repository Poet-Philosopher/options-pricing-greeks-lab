"""Monte Carlo engine: European pricing, convergence, path fan and Asian options."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import norm

import optionslab as ol
from app_utils import compute, theme, ui

ui.setup("Monte Carlo Engine", "🎲", "Vectorised GBM simulation with antithetic variates, confidence intervals and an Asian option.")
m = ui.market_inputs("mc")
cur = m.preset.currency

sb = st.sidebar
sb.header("Simulation")
kind = sb.radio("Option type", ["call", "put"], horizontal=True, key="mc_kind", format_func=str.title)
n_paths = sb.select_slider("Paths", [1_000, 5_000, 10_000, 50_000, 100_000, 200_000, 500_000, 1_000_000], value=100_000, key="mc_n")
n_steps = sb.slider("Time steps (fan chart & Asian fixings)", 5, 252, 50, key="mc_steps")
seed = int(sb.number_input("Random seed", min_value=0, value=42, step=1, key="mc_seed", help="Fixed seed => reproducible results."))
anti = sb.checkbox("Antithetic variates", value=True, key="mc_anti")
args = (m.S, m.K, m.T, m.r, m.q, m.sigma, kind)
exact = float(ol.bs_price(*args))

tab_e, tab_c, tab_f, tab_a = st.tabs(["European price", "Convergence", "Path fan", "Asian option"])

# ------------------------------------------------------------------ European
with tab_e:
    plain = compute.mc_european(*args, n_paths, False, seed)
    av = compute.mc_european(*args, n_paths, True, seed)
    sel = av if anti else plain
    a, b, c, d = st.columns(4)
    a.metric("Analytic BSM", ui.money(exact, cur, 4))
    b.metric(f"MC estimate ({'antithetic' if anti else 'plain'})", ui.money(sel.price, cur, 4), delta=f"{sel.price - exact:+.4f} vs BSM", delta_color="off")
    c.metric("Standard error", f"{sel.std_error:.5f}", help="Std. deviation of the estimator: sample std / sqrt(n). For antithetic runs n counts pairs.")
    d.metric("95% CI", f"[{sel.ci_low:,.4f}, {sel.ci_high:,.4f}]", help="Estimate +/- 1.96 standard errors.")
    (st.success if sel.contains(exact) else st.warning)(
        f"Analytic BSM price {exact:,.4f} {'lies inside' if sel.contains(exact) else 'is OUTSIDE'} the 95% confidence interval. "
        "(A correct 95% CI misses about 1 time in 20 - try another seed.)")
    vrf = (plain.std_error / av.std_error) ** 2
    st.markdown(f"**Variance reduction from antithetic variates:** SE {plain.std_error:.5f} -> {av.std_error:.5f}, "
                f"i.e. about **{vrf:.1f}x** less variance for the same number of paths.")
    table = pd.DataFrame([
        {"Method": r.method, "Paths": r.n_paths, "Price": r.price, "Std error": r.std_error, "CI low": r.ci_low, "CI high": r.ci_high,
         "Error vs BSM": r.price - exact, "BSM inside CI": r.contains(exact)} for r in (plain, av)])
    st.dataframe(table.style.format({"Price": "{:.4f}", "Std error": "{:.5f}", "CI low": "{:.4f}", "CI high": "{:.4f}", "Error vs BSM": "{:+.4f}"}),
                 hide_index=True, use_container_width=True)
    fig = go.Figure()
    for r, col in ((plain, theme.ORANGE), (av, theme.TEAL)):
        fig.add_scatter(x=[r.price], y=[r.method.title()], mode="markers", marker=dict(size=11, color=col),
                        error_x=dict(type="data", array=[ol.monte_carlo.Z95 * r.std_error], color=col, thickness=2.5, width=8),
                        showlegend=False, hovertemplate="%{x:.4f}<extra></extra>")
    theme.vref(fig, exact, "Analytic BSM", theme.INDIGO)
    theme.style_fig(fig, "Estimates with 95% confidence intervals", f"Option price {cur}".strip(), None, height=280)
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ Convergence
with tab_c:
    df = compute.mc_convergence(*args, n_paths, anti, seed)
    fig = go.Figure()
    fig.add_scatter(x=df.n_paths, y=df.ci_high, line=dict(width=0), hoverinfo="skip", showlegend=False)
    fig.add_scatter(x=df.n_paths, y=df.ci_low, fill="tonexty", fillcolor=theme.BAND, line=dict(width=0), name="95% CI", hoverinfo="skip")
    fig.add_scatter(x=df.n_paths, y=df.price, name="MC estimate", line=dict(color=theme.TEAL, width=2.5))
    theme.href(fig, exact, "Analytic BSM", theme.INDIGO)
    theme.style_fig(fig, "Price estimate vs number of paths", "Paths (log scale)", f"Option price {cur}".strip(), height=420)
    theme.log_axis(fig, "x", [100, 1e3, 1e4, 1e5, 1e6], ["100", "1k", "10k", "100k", "1M"])
    left, right = st.columns(2)
    left.plotly_chart(fig, use_container_width=True)
    ref = df.std_error.iloc[0] * np.sqrt(df.n_paths.iloc[0] / df.n_paths)
    fig2 = go.Figure()
    fig2.add_scatter(x=df.n_paths, y=df.std_error, name="Standard error", line=dict(color=theme.ORANGE, width=2.5))
    fig2.add_scatter(x=df.n_paths, y=ref, name="1/sqrt(n) reference", line=dict(color=theme.SLATE, dash="dot"))
    theme.style_fig(fig2, "Standard error shrinks like 1/sqrt(n)", "Paths (log scale)", "Standard error (log scale)", height=420)
    theme.log_axis(fig2, "x", [100, 1e3, 1e4, 1e5, 1e6], ["100", "1k", "10k", "100k", "1M"])
    theme.log_axis(fig2, "y", *theme.decades(df.std_error))
    right.plotly_chart(fig2, use_container_width=True)
    st.caption("One simulation, evaluated at growing sample sizes. Quadrupling the paths halves the error - "
               "the cost of Monte Carlo accuracy.")

# ------------------------------------------------------------------ Fan chart
with tab_f:
    n_show = 2000
    times, paths = compute.mc_paths(m.S, m.r, m.q, m.sigma, m.T, n_show, n_steps, seed)
    days_axis = times * 365
    qs = [5, 25, 50, 75, 95]
    pct = np.percentile(paths, qs, axis=0)
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.78, 0.22], horizontal_spacing=0.02)
    for row in paths[:25]:
        fig.add_scatter(x=days_axis, y=row, mode="lines", line=dict(color="rgba(100,116,139,0.25)", width=1), hoverinfo="skip", showlegend=False, row=1, col=1)
    fig.add_scatter(x=days_axis, y=pct[4], line=dict(width=0), hoverinfo="skip", showlegend=False, row=1, col=1)
    fig.add_scatter(x=days_axis, y=pct[0], fill="tonexty", fillcolor="rgba(13,148,136,0.14)", line=dict(width=0), name="5-95% band", hoverinfo="skip", row=1, col=1)
    fig.add_scatter(x=days_axis, y=pct[3], line=dict(width=0), hoverinfo="skip", showlegend=False, row=1, col=1)
    fig.add_scatter(x=days_axis, y=pct[1], fill="tonexty", fillcolor="rgba(13,148,136,0.28)", line=dict(width=0), name="25-75% band", hoverinfo="skip", row=1, col=1)
    fig.add_scatter(x=days_axis, y=pct[2], name="Median", line=dict(color=theme.TEAL, width=3), row=1, col=1)
    # analytic lognormal quantiles as validation
    mu = (m.r - m.q - 0.5 * m.sigma**2) * times
    for p_, nm in ((0.05, "5%"), (0.95, "95%")):
        fig.add_scatter(x=days_axis, y=m.S * np.exp(mu + m.sigma * np.sqrt(times) * norm.ppf(p_)), name="Analytic quantiles", legendgroup="an",
                        showlegend=(p_ == 0.05), line=dict(color=theme.INDIGO, width=1.5, dash="dash"), row=1, col=1)
    theme.href(fig, m.K, "Strike", theme.ORANGE, "dot", row=1, col=1)
    fig.add_histogram(y=paths[:, -1], nbinsy=40, marker_color=theme.TEAL, opacity=0.7, showlegend=False, orientation="h", row=1, col=2)
    theme.style_fig(fig, f"Simulated GBM paths ({n_show:,} paths, {n_steps} steps; 25 drawn)", None, f"Underlying {cur}".strip(), height=520)
    fig.update_yaxes(title_text=None, row=1, col=2)
    fig.update_xaxes(title_text="Days from today", row=1, col=1)
    fig.update_xaxes(title_text="Terminal", showticklabels=False, row=1, col=2)
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Dashed indigo lines are the exact lognormal 5%/95% quantiles - the simulated band should hug them.")

# ------------------------------------------------------------------ Asian
with tab_a:
    st.markdown("**Arithmetic-average Asian option** (no closed form): payoff is based on the average of the underlying "
                f"at {n_steps} equally spaced fixings. Compare plain, antithetic and control-variate estimators.")
    runs = {
        "Plain": compute.mc_asian(*args, n_paths, n_steps, False, False, seed),
        "Antithetic": compute.mc_asian(*args, n_paths, n_steps, True, False, seed),
        "Antithetic + control variate": compute.mc_asian(*args, n_paths, n_steps, True, True, seed),
    }
    geo = ol.geometric_asian_price(*args[:6], n_steps, kind)
    best = runs["Antithetic + control variate"]
    a, b, c, d = st.columns(4)
    a.metric("Asian (best estimator)", ui.money(best.price, cur, 4), help="Antithetic + geometric-Asian control variate.")
    b.metric("95% CI", f"[{best.ci_low:,.4f}, {best.ci_high:,.4f}]")
    c.metric("European (BSM)", ui.money(exact, cur, 4), help="Averaging lowers variance, so an Asian option is usually cheaper than its European twin.")
    d.metric("Geometric Asian (closed form)", ui.money(geo, cur, 4), help="Kemna-Vorst-style exact price; used as the control variate.")
    table = pd.DataFrame([{"Estimator": k, "Price": r.price, "Std error": r.std_error, "CI low": r.ci_low, "CI high": r.ci_high,
                           "Variance vs plain": (runs["Plain"].std_error / r.std_error) ** 2} for k, r in runs.items()])
    st.dataframe(table.style.format({"Price": "{:.4f}", "Std error": "{:.5f}", "CI low": "{:.4f}", "CI high": "{:.4f}", "Variance vs plain": "{:.1f}x less"}),
                 hide_index=True, use_container_width=True)
    fig = go.Figure()
    for (k, r), col in zip(runs.items(), (theme.ORANGE, theme.TEAL, theme.INDIGO)):
        fig.add_scatter(x=[r.price], y=[k], mode="markers", marker=dict(size=11, color=col), showlegend=False,
                        error_x=dict(type="data", array=[ol.monte_carlo.Z95 * r.std_error], color=col, thickness=2.5, width=8))
    theme.style_fig(fig, "Asian estimates with 95% CI", f"Option price {cur}".strip(), None, height=260, legend=False)
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig, use_container_width=True)
ui.footer()
