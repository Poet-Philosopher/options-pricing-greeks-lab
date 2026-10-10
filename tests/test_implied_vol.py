"""Implied volatility: round trips, edge cases, parsing and smile."""
import numpy as np
import pytest

from optionslab import (
    bs_price, implied_vol, implied_vol_smile, no_arbitrage_bounds, parse_quotes, synthetic_smile_example,
)

BASE = dict(S=100.0, T=0.5, r=0.05, q=0.01)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("K", [70, 90, 100, 110, 140])
@pytest.mark.parametrize("sigma", [0.08, 0.2, 0.5, 1.2])
def test_round_trip(K, sigma, kind):
    price = bs_price(BASE["S"], K, BASE["T"], BASE["r"], BASE["q"], sigma, kind)
    res = implied_vol(price, BASE["S"], K, BASE["T"], BASE["r"], BASE["q"], kind)
    if res.status == "ok" and not res.warnings:
        assert res.converged and res.iv == pytest.approx(sigma, abs=2e-5)
    elif res.status == "ok":  # ill-conditioned (tiny vega): IV is flagged, but must still reprice
        reprice = bs_price(BASE["S"], K, BASE["T"], BASE["r"], BASE["q"], res.iv, kind)
        assert reprice == pytest.approx(price, abs=1e-6 * max(1.0, price), rel=1e-6)
    else:  # only acceptable when the price has (numerically) no time value
        assert res.status in {"at_lower_bound", "iv_below_floor"}
        assert price - no_arbitrage_bounds(100, K, BASE["T"], BASE["r"], BASE["q"], kind)[0] < 1e-6


def test_newton_is_default_and_fast():
    p = bs_price(100, 100, 1, 0.05, 0, 0.25, "call")
    res = implied_vol(p, 100, 100, 1, 0.05, 0, "call")
    assert res.method == "newton" and res.iterations <= 10 and res.iv == pytest.approx(0.25, abs=1e-8)


def test_bisection_fallback_is_used_when_newton_cannot_start():
    # Very deep OTM + tiny vega at the seed: Newton bails out, Brent must still find the root.
    sigma = 0.9
    p = bs_price(100, 400, 0.1, 0.02, 0.0, sigma, "call")
    res = implied_vol(p, 100, 400, 0.1, 0.02, 0.0, "call")
    assert res.converged and res.iv == pytest.approx(sigma, abs=1e-4)
    assert res.method in {"newton", "brentq"}


def test_forced_fallback_method_label(monkeypatch):
    import importlib
    m = importlib.import_module("optionslab.implied_vol")
    monkeypatch.setattr(m, "bs_greeks", lambda *a, **k: type("G", (), {"vega": 0.0})())
    p = bs_price(100, 110, 0.5, 0.03, 0.0, 0.3, "call")
    res = m.implied_vol(p, 100, 110, 0.5, 0.03, 0.0, "call")
    assert res.method == "brentq" and res.iv == pytest.approx(0.3, abs=1e-6)


def test_price_below_intrinsic():
    res = implied_vol(5.0, 120, 100, 0.5, 0.05, 0.0, "call")  # intrinsic ~ 22.5
    assert not res.converged and res.iv is None and res.status == "below_intrinsic"
    assert "intrinsic" in res.message


def test_price_above_upper_bound():
    res = implied_vol(101, 100, 100, 0.5, 0.05, 0.0, "call")
    assert res.status == "above_upper_bound" and not res.converged


@pytest.mark.parametrize("price", [0.0, -1.0])
def test_non_positive_price(price):
    assert implied_vol(price, 100, 100, 1, 0.05, 0, "call").status == "non_positive_price"


def test_invalid_inputs():
    assert implied_vol(5, 100, 100, 0.0, 0.05, 0, "call").status == "invalid_input"
    assert implied_vol(float("nan"), 100, 100, 1, 0.05, 0, "call").status == "invalid_input"


def test_price_equal_to_intrinsic_reports_zero_time_value():
    lower, _ = no_arbitrage_bounds(120, 100, 0.5, 0.05, 0.0, "call")
    res = implied_vol(lower, 120, 100, 0.5, 0.05, 0.0, "call")
    assert res.status in {"at_lower_bound", "iv_below_floor"} and res.iv in (None, 1e-6)


def test_deep_itm_warns_about_low_vega():
    p = bs_price(100, 40, 0.1, 0.03, 0.0, 0.2, "call") + 1e-6
    res = implied_vol(p, 100, 40, 0.1, 0.03, 0.0, "call")
    assert res.status == "ok" and any("Vega is tiny" in w for w in res.warnings)


def test_tiny_otm_price_is_not_accepted_at_the_seed_vol():
    # Regression: an absolute tolerance used to 'converge' instantly at the seed for ~0 prices.
    p = bs_price(100, 70, 0.5, 0.05, 0.01, 0.12, "put")
    res = implied_vol(p, 100, 70, 0.5, 0.05, 0.01, "put")
    assert res.status == "ok" and res.iv == pytest.approx(0.12, abs=1e-3)


def test_parse_quotes_variants():
    text = "strike,price,type\n95, 8.1, call\n100\t5.2\t C\n105;3.1;put\n# comment\n\n110 1.9"
    df, errs = parse_quotes(text)
    assert errs == []
    assert list(df["strike"]) == [95, 100, 105, 110]
    assert list(df["kind"]) == ["call", "call", "put", "call"]


def test_parse_quotes_reports_bad_lines():
    df, errs = parse_quotes("100, 5\nabc, def\n105, -3\n110, 2, swap")
    assert len(df) == 1 and len(errs) == 3


def test_smile_recovers_synthetic_vol_curve():
    S, T, r, q = 100.0, 0.25, 0.05, 0.0
    strikes = [85, 90, 95, 100, 105, 110, 115]
    quotes = synthetic_smile_example(S, T, r, q, strikes, base_vol=0.2, skew=-0.3, curvature=1.0)
    df = implied_vol_smile(quotes.strike, quotes.price, quotes.kind, S, T, r, q)
    assert (df.status == "ok").all()
    # Prices are rounded to 2dp, so IVs are close but not exact; the skew shape must be recovered.
    assert df.iv.iloc[0] > df.iv.iloc[3]
    assert np.all(np.isfinite(df.iv))


def test_smile_marks_failures_as_nan():
    df = implied_vol_smile([100, 100], [8.0, 0.5], ["call", "call"], 150, 0.5, 0.05, 0.0)
    assert df.iv.isna().all() and (df.status == "below_intrinsic").all()
