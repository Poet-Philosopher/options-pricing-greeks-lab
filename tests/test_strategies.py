"""Strategy payoff analytics, breakevens and presets."""
import math

import numpy as np
import pytest

from optionslab import (
    PRESET_NAMES, Leg, analyze, bs_price, entry_cost, expiry_pnl, pnl_at, preset_legs, validate_market_inputs,
)

M = dict(spot=100.0, T=0.25, r=0.05, q=0.0, sigma=0.2)


def test_long_straddle_breakevens_and_extremes():
    legs = [Leg("call", 1, 100), Leg("put", 1, 100)]
    s = analyze(legs, **M)
    prem = bs_price(100, 100, 0.25, 0.05, 0.0, 0.2, "call") + bs_price(100, 100, 0.25, 0.05, 0.0, 0.2, "put")
    assert s.entry_cost == pytest.approx(prem)
    assert s.breakevens == pytest.approx((100 - prem, 100 + prem))
    assert s.max_loss == pytest.approx(-prem)
    assert s.unlimited_profit and math.isinf(s.max_profit)


def test_bull_call_spread():
    legs = [Leg("call", 1, 100, 6.0), Leg("call", -1, 110, 2.0)]
    s = analyze(legs, **M)
    assert s.entry_cost == pytest.approx(4.0)
    assert s.max_profit == pytest.approx(6.0)     # width 10 - debit 4
    assert s.max_loss == pytest.approx(-4.0)
    assert s.breakevens == pytest.approx((104.0,))
    assert not s.unlimited_profit and not s.unlimited_loss


def test_short_iron_condor_bounded_both_ways():
    legs = preset_legs("Iron condor", 100, 5)
    s = analyze(legs, **M)
    assert s.entry_cost < 0                       # net credit
    assert s.max_profit == pytest.approx(-s.entry_cost)
    assert s.max_loss == pytest.approx(-(5 + s.entry_cost))  # wing width 5 minus credit
    assert len(s.breakevens) == 2 and s.breakevens[0] < 100 < s.breakevens[1]


def test_covered_call():
    legs = [Leg("stock", 1), Leg("call", -1, 105, 2.0)]
    s = analyze(legs, **M)
    assert not s.unlimited_profit and not s.unlimited_loss
    assert s.max_profit == pytest.approx(105 - 100 + 2.0)
    assert s.max_loss == pytest.approx(-(100 - 2.0))
    assert s.breakevens == pytest.approx((98.0,))


def test_naked_short_call_has_unlimited_loss_and_single_breakeven():
    s = analyze([Leg("call", -1, 100, 4.0)], **M)
    assert s.unlimited_loss and math.isinf(s.max_loss) and s.max_loss < 0
    assert s.max_profit == pytest.approx(4.0)
    assert s.breakevens == pytest.approx((104.0,))


def test_long_put_loss_and_profit():
    s = analyze([Leg("put", 1, 100, 3.0)], **M)
    assert s.max_loss == pytest.approx(-3.0) and s.max_profit == pytest.approx(97.0)
    assert s.breakevens == pytest.approx((97.0,))


def test_breakevens_have_zero_pnl_on_grid():
    for name in PRESET_NAMES:
        legs = preset_legs(name, 100, 5)
        s = analyze(legs, **M)
        for be in s.breakevens:
            assert abs(expiry_pnl(legs, np.array([be]), s.entry_cost)[0]) < 1e-8


def test_max_profit_matches_dense_grid():
    legs = preset_legs("Bull call spread", 100, 5)
    s = analyze(legs, **M)
    grid = np.linspace(0, 400, 40001)
    assert s.max_profit == pytest.approx(expiry_pnl(legs, grid, s.entry_cost).max(), abs=1e-6)


def test_pnl_today_is_zero_at_inception_with_theoretical_premiums():
    legs = preset_legs("Iron condor", 100, 5)
    cost = entry_cost(legs, **M)
    today = pnl_at(legs, np.array([100.0]), M["T"], M["r"], M["q"], M["sigma"], cost)
    assert today[0] == pytest.approx(0.0, abs=1e-12)


def test_value_converges_to_expiry_payoff_as_time_runs_out():
    legs = preset_legs("Strangle", 100, 5)
    cost = entry_cost(legs, **M)
    grid = np.linspace(60, 140, 81)
    near = pnl_at(legs, grid, 1e-9, M["r"], M["q"], M["sigma"], cost)
    assert np.max(np.abs(near - expiry_pnl(legs, grid, cost))) < 1e-3  # residual = time value at the kinks


def test_leg_validation():
    with pytest.raises(ValueError):
        Leg("call", 0, 100)
    with pytest.raises(ValueError):
        Leg("call", 1, -5)
    with pytest.raises(ValueError):
        Leg("future", 1, 5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        preset_legs("Butterfly", 100, 5)


def test_validate_market_inputs():
    assert validate_market_inputs(100, 100, 1, 0.05, 0.0, 0.2) == []
    assert any("Spot" in m for m in validate_market_inputs(-1, 100, 1, 0.05, 0.0, 0.2))
    assert any("Volatility" in m for m in validate_market_inputs(100, 100, 1, 0.05, 0.0, 0.0))
    assert any("finite" in m for m in validate_market_inputs(float("nan"), 100, 1, 0.05, 0.0, 0.2))
