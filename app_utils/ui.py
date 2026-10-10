"""Page scaffolding and the shared sidebar of market inputs.

Inputs persist across pages via ``st.session_state['shared_*']`` so a user can
set up a scenario once and move between pages without re-entering it.
"""
from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from optionslab import GENERIC, PRESETS, MarketPreset, validate_market_inputs
from optionslab.bsm import DAYS_PER_YEAR

from .text import DISCLAIMER, SIDEBAR_NOTE

_CSS = """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px;}
[data-testid="stMetric"] {background: rgba(100,116,139,0.07); border: 1px solid rgba(100,116,139,0.18);
  border-radius: 10px; padding: 0.7rem 0.9rem;}
[data-testid="stMetricLabel"] p {font-size: 0.82rem; opacity: 0.8;}
.lab-hero h1 {margin-bottom: 0.1rem;}
.lab-sub {opacity: 0.75; margin-bottom: 1rem;}
.assump {border-left: 4px solid #EA580C; padding: 0.4rem 0.8rem; background: rgba(234,88,12,0.07);
  border-radius: 4px; font-size: 0.88rem;}
</style>
"""


def setup(title: str, icon: str, subtitle: str | None = None) -> None:
    """Configure the page (must be the first Streamlit call) and render the header."""
    st.set_page_config(page_title=title if "Lab" in title else f"{title} - Options Lab", page_icon=icon, layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(_CSS, unsafe_allow_html=True)
    st.title(f"{icon} {title}")
    if subtitle:
        st.markdown(f"<div class='lab-sub'>{subtitle}</div>", unsafe_allow_html=True)


def footer() -> None:
    """Disclaimer shown at the bottom of every page."""
    st.divider()
    st.caption(DISCLAIMER)


def assumption_note(text: str) -> None:
    """Orange call-out used wherever a default is an assumption."""
    st.markdown(f"<div class='assump'>{text}</div>", unsafe_allow_html=True)


@dataclass(frozen=True)
class MarketInputs:
    """Validated market inputs (rates and vol as decimals, ``T`` in years)."""

    S: float
    K: float
    days: int
    T: float
    r: float
    q: float
    sigma: float
    preset: MarketPreset
    lot_size: int | None
    strike_step: float


def _apply_preset(page: str) -> None:
    p = PRESETS[st.session_state[f"{page}_preset"]]
    st.session_state["shared_preset"] = p.name
    for name, val in (("spot", p.spot), ("strike", p.strike), ("days", p.days_to_expiry), ("rate", p.rate_pct),
                      ("div", p.div_yield_pct), ("vol", p.vol_pct), ("lot", p.lot_size or 1)):
        st.session_state[f"shared_{name}"] = val
        st.session_state[f"{page}_{name}"] = val


def _field(page: str, name: str, default):
    key = f"{page}_{name}"
    if key not in st.session_state:
        st.session_state[key] = st.session_state.get(f"shared_{name}", default)
    return key


def market_inputs(page: str, *, strike: bool = True, vol: bool = True) -> MarketInputs:
    """Render the sidebar market inputs and return validated values.

    Stops the page with a clear error message if any input is invalid.
    """
    sb = st.sidebar
    sb.header("Market inputs")
    names = list(PRESETS)
    pkey = f"{page}_preset"
    if pkey not in st.session_state:
        st.session_state[pkey] = st.session_state.get("shared_preset", GENERIC.name)
    sb.selectbox("Preset", names, key=pkey, on_change=_apply_preset, args=(page,),
                 help="Loads a set of illustrative defaults. Every value below stays editable.")
    preset = PRESETS[st.session_state[pkey]]
    cur = f"{preset.currency} " if preset.currency else ""

    spot_step = max(preset.strike_step / 5, 0.01)
    S = sb.number_input(f"Spot price {cur}".strip(), min_value=0.01, step=spot_step, format="%.2f",
                        key=_field(page, "spot", preset.spot), help="Current price of the underlying (assumption).")
    K = S
    if strike:
        K = sb.number_input(f"Strike {cur}".strip(), min_value=0.01, step=preset.strike_step, format="%.2f",
                            key=_field(page, "strike", preset.strike))
    days = sb.number_input("Days to expiry", min_value=1, max_value=3650, step=1,
                           key=_field(page, "days", preset.days_to_expiry),
                           help="Calendar days; T = days / 365.")
    r = sb.number_input("Risk-free rate (% p.a.)", min_value=-5.0, max_value=50.0, step=0.1, format="%.2f",
                        key=_field(page, "rate", preset.rate_pct), help="Continuously compounded. Assumption.")
    q = sb.number_input("Dividend yield (% p.a.)", min_value=0.0, max_value=50.0, step=0.1, format="%.2f",
                        key=_field(page, "div", preset.div_yield_pct), help="Continuous yield. Assumption.")
    sigma = 0.2
    if vol:
        sigma = sb.number_input("Volatility (% p.a.)", min_value=0.1, max_value=500.0, step=0.5, format="%.2f",
                                key=_field(page, "vol", preset.vol_pct),
                                help="Annualised, constant. Assumption - not a market quote.") / 100.0
    lot = None
    if preset.lot_size:
        lot = int(sb.number_input("Lot size (units per contract)", min_value=1, step=1,
                                  key=_field(page, "lot", preset.lot_size),
                                  help="NSE revises lot sizes periodically - verify the current value on nseindia.com."))
    if preset.currency:
        sb.caption(preset.notes)
    else:
        sb.caption(SIDEBAR_NOTE)
    sb.divider()
    sb.caption("Educational only - not investment advice.")

    for k, v in (("spot", S), ("strike", K), ("days", days), ("rate", r), ("div", q), ("vol", sigma * 100), ("lot", lot or 1)):
        st.session_state[f"shared_{k}"] = v
    st.session_state["shared_preset"] = preset.name

    T, rr, qq = days / DAYS_PER_YEAR, r / 100.0, q / 100.0
    problems = validate_market_inputs(S, K, T, rr, qq, sigma if vol else None)
    if problems:
        for p in problems:
            st.error(p)
        st.stop()
    return MarketInputs(S, K, int(days), T, rr, qq, sigma, preset, lot, preset.strike_step)


def money(x: float, currency: str = "", digits: int = 2) -> str:
    """Format a number with thousands separators and an optional currency symbol."""
    sym = {"INR": "₹"}.get(currency, f"{currency} " if currency else "")
    return f"{sym}{x:,.{digits}f}"
