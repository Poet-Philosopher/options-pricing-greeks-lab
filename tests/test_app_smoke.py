"""Headless smoke tests: every Streamlit page must render without exceptions."""
import glob
import os

import pytest
from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGES = ["app.py"] + sorted(os.path.relpath(p, ROOT) for p in glob.glob(os.path.join(ROOT, "views", "*.py"))
                           if not p.endswith("0_Home.py"))  # Home needs the router (st.page_link) -> covered via app.py


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exceptions(page):
    at = AppTest.from_file(os.path.join(ROOT, page), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


@pytest.mark.parametrize("page", [p for p in PAGES if p.startswith("views") and "Methodology" not in p and "Home" not in p])
def test_nifty_preset_renders(page):
    at = AppTest.from_file(os.path.join(ROOT, page), default_timeout=60).run()
    sel = [s for s in at.sidebar.selectbox if s.label == "Preset"]
    assert sel, "sidebar preset selector missing"
    sel[0].set_value("Nifty 50 (India)").run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def _run(page):
    return AppTest.from_file(os.path.join(ROOT, "views", page), default_timeout=90).run()


def _clean(at):
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("preset", ["Covered call", "Bull call spread", "Straddle", "Strangle", "Iron condor"])
def test_strategy_presets(preset):
    at = _run("3_Strategy_Builder.py")
    at.selectbox(key="sb_choice").set_value(preset).run()
    _clean(at)
    assert not at.error
    assert any("Max profit" in m.label for m in at.metric)


def test_implied_vol_bad_price_shows_message_not_exception():
    at = _run("4_Implied_Vol.py")
    num = [n for n in at.number_input if n.label.startswith("Observed option price")][0]
    num.set_value(0.01).run()
    _clean(at)
    num = [n for n in at.number_input if n.label.startswith("Observed option price")][0]
    num.set_value(500.0).run()
    _clean(at)
    assert any("above upper bound" in e.value.lower() for e in at.error)


def test_monte_carlo_toggles():
    at = _run("5_Monte_Carlo.py")
    at.checkbox(key="mc_anti").uncheck().run()
    _clean(at)
    at.radio(key="mc_kind").set_value("put").run()
    _clean(at)
    assert not at.error


def test_binomial_call_and_extreme_inputs():
    at = _run("6_Binomial_Tree.py")
    at.radio(key="tree_kind").set_value("call").run()
    _clean(at)
    # very low vol with long maturity and many steps => invalid tree probability => friendly error, no crash
    at.slider(key="tree_n").set_value(1500).run()
    for n in at.sidebar.number_input:
        if n.label.startswith("Volatility"):
            n.set_value(0.1)
        if n.label.startswith("Risk-free"):
            n.set_value(40.0)
        if n.label.startswith("Days"):
            n.set_value(3650)
    at.run()
    _clean(at)


def test_greeks_every_greek():
    at = _run("2_Greeks_Explorer.py")
    for g in ["Delta", "Gamma", "Vega", "Theta", "Rho"]:
        at.radio(key="gx_greek").set_value(g).run()
        _clean(at)
    at.radio(key="gx_kind").set_value("put").run()
    _clean(at)


def test_invalid_input_stops_with_error():
    at = _run("1_BSM_Pricer.py")
    for n in at.sidebar.number_input:
        if n.label.startswith("Spot"):
            n.set_value(0.01)
        if n.label.startswith("Days"):
            n.set_value(1)
    at.run()
    _clean(at)
