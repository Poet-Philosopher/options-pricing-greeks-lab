"""Cached wrappers around optionslab so expensive calls are not recomputed on every rerun."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import optionslab as ol


@st.cache_data(show_spinner=False)
def greek_curve(x_name: str, xs: tuple[float, ...], S: float, K: float, T: float, r: float, q: float,
                sigma: float, kind: str) -> pd.DataFrame:
    """All five Greeks (display units) + price along a grid of one variable.

    ``x_name`` is ``'spot'``, ``'days'`` (days to expiry) or ``'vol'`` (decimal).
    """
    x = np.asarray(xs, dtype=float)
    args = dict(S=S, K=K, T=T, r=r, q=q, sigma=sigma)
    if x_name == "spot":
        args["S"] = x
    elif x_name == "days":
        args["T"] = x / ol.DAYS_PER_YEAR
    elif x_name == "vol":
        args["sigma"] = x
    else:
        raise ValueError(x_name)
    g = ol.bs_greeks(**args, kind=kind).display_units()
    return pd.DataFrame({"x": x, "Price": ol.bs_price(**args, kind=kind), "Delta": g.delta, "Gamma": g.gamma,
                         "Vega": g.vega, "Theta": g.theta, "Rho": g.rho})


@st.cache_data(show_spinner="Running Monte Carlo...")
def mc_european(S, K, T, r, q, sigma, kind, n_paths, antithetic, seed):
    return ol.mc_european(S, K, T, r, q, sigma, kind, n_paths, antithetic, seed)


@st.cache_data(show_spinner="Building convergence series...")
def mc_convergence(S, K, T, r, q, sigma, kind, n_max, antithetic, seed):
    return ol.convergence_series(S, K, T, r, q, sigma, kind, n_max, antithetic, seed)


@st.cache_data(show_spinner="Simulating paths...")
def mc_paths(S, r, q, sigma, T, n_paths, n_steps, seed):
    return ol.simulate_gbm_paths(S, r, q, sigma, T, n_paths, n_steps, seed)


@st.cache_data(show_spinner="Pricing Asian option...")
def mc_asian(S, K, T, r, q, sigma, kind, n_paths, n_steps, antithetic, control_variate, seed):
    return ol.mc_asian_arithmetic(S, K, T, r, q, sigma, kind, n_paths, n_steps, antithetic, control_variate, seed)


@st.cache_data(show_spinner="Building tree...")
def crr_convergence(S, K, T, r, q, sigma, steps: tuple[int, ...], kind):
    return ol.crr_convergence(S, K, T, r, q, sigma, list(steps), kind)


@st.cache_data(show_spinner=False)
def eep_curve(spots: tuple[float, ...], K, T, r, q, sigma, n_steps, kind) -> pd.DataFrame:
    rows = []
    for s in spots:
        a = ol.crr_price(s, K, T, r, q, sigma, n_steps, kind, True)
        e = ol.crr_price(s, K, T, r, q, sigma, n_steps, kind, False)
        rows.append({"spot": s, "american": a, "european": e, "premium": a - e, "intrinsic": max((s - K) if kind == "call" else (K - s), 0.0)})
    return pd.DataFrame(rows)
