"""Black-Scholes-Merton: textbook values, parity, Greeks vs finite differences."""
import math

import numpy as np
import pytest
from scipy.stats import norm

from optionslab import bs_greeks, bs_price, put_call_parity_gap

# (S, K, T, r, q, sigma)
PARAMS = [
    (100, 100, 1.0, 0.05, 0.00, 0.20),
    (100, 110, 0.5, 0.03, 0.02, 0.35),
    (100, 85, 2.0, 0.06, 0.01, 0.15),
    (25000, 25500, 30 / 365, 0.065, 0.013, 0.14),
    (42, 40, 0.5, 0.10, 0.00, 0.20),
]


def reference_price(S, K, T, r, q, sigma, kind):
    """Independent implementation using scipy.stats.norm (cross-check)."""
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "call":
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)


class TestTextbookValues:
    def test_hull_atm_call_and_put(self):
        # Standard benchmark: S=K=100, T=1, r=5%, sigma=20%, q=0
        assert bs_price(100, 100, 1, 0.05, 0, 0.20, "call") == pytest.approx(10.4506, abs=5e-5)
        assert bs_price(100, 100, 1, 0.05, 0, 0.20, "put") == pytest.approx(5.5735, abs=5e-5)

    def test_hull_example_call_put(self):
        # Hull, Options Futures & Other Derivatives: S=42,K=40,r=10%,sigma=20%,T=0.5 -> c=4.76, p=0.81
        assert bs_price(42, 40, 0.5, 0.10, 0, 0.20, "call") == pytest.approx(4.76, abs=0.005)
        assert bs_price(42, 40, 0.5, 0.10, 0, 0.20, "put") == pytest.approx(0.81, abs=0.005)

    def test_hull_greeks(self):
        # Hull: S=49, K=50, r=5%, sigma=20%, T=20 weeks -> delta .522, gamma .066, vega 12.1, theta -4.31, rho 8.91
        g = bs_greeks(49, 50, 20 / 52, 0.05, 0, 0.20, "call")
        assert g.delta == pytest.approx(0.522, abs=0.001)
        assert g.gamma == pytest.approx(0.066, abs=0.001)
        assert g.vega == pytest.approx(12.1, abs=0.05)
        assert g.theta == pytest.approx(-4.31, abs=0.02)
        assert g.rho == pytest.approx(8.91, abs=0.02)

    @pytest.mark.parametrize("kind", ["call", "put"])
    @pytest.mark.parametrize("p", PARAMS)
    def test_matches_independent_reference(self, p, kind):
        assert bs_price(*p, kind) == pytest.approx(reference_price(*p, kind), rel=1e-12)


class TestParityAndBounds:
    @pytest.mark.parametrize("p", PARAMS)
    def test_put_call_parity(self, p):
        S, K, T, r, q, sigma = p
        c, pu = bs_price(*p, "call"), bs_price(*p, "put")
        assert abs(put_call_parity_gap(c, pu, S, K, T, r, q)) < 1e-9 * max(1, S)

    def test_vectorised_parity(self):
        S = np.linspace(50, 150, 41)
        c, p = bs_price(S, 100, 0.5, 0.04, 0.01, 0.3, "call"), bs_price(S, 100, 0.5, 0.04, 0.01, 0.3, "put")
        assert np.max(np.abs(put_call_parity_gap(c, p, S, 100, 0.5, 0.04, 0.01))) < 1e-10

    @pytest.mark.parametrize("p", PARAMS)
    def test_no_arbitrage_bounds(self, p):
        S, K, T, r, q, _ = p
        c = bs_price(*p, "call")
        assert max(S * math.exp(-q * T) - K * math.exp(-r * T), 0) - 1e-12 <= c <= S * math.exp(-q * T) + 1e-12

    def test_expiry_and_zero_vol_limits(self):
        assert bs_price(110, 100, 0.0, 0.05, 0.0, 0.2, "call") == pytest.approx(10.0)
        assert bs_price(90, 100, 0.0, 0.05, 0.0, 0.2, "put") == pytest.approx(10.0)
        assert bs_price(100, 90, 1.0, 0.0, 0.0, 1e-14, "call") == pytest.approx(10.0)

    def test_invalid_kind(self):
        with pytest.raises(ValueError):
            bs_price(100, 100, 1, 0.05, 0, 0.2, "straddle")  # type: ignore[arg-type]


class TestGreeksFiniteDifference:
    """Analytic Greeks must match central finite differences of the price."""

    @pytest.mark.parametrize("kind", ["call", "put"])
    @pytest.mark.parametrize("p", PARAMS)
    def test_all_greeks(self, p, kind):
        S, K, T, r, q, sigma = p
        g = bs_greeks(*p, kind)
        f = lambda S=S, T=T, r=r, sigma=sigma: bs_price(S, K, T, r, q, sigma, kind)  # noqa: E731
        hS, hv, hr, hT = S * 1e-4, 1e-5, 1e-6, 1e-5
        delta = (f(S=S + hS) - f(S=S - hS)) / (2 * hS)
        gamma = (f(S=S + hS) - 2 * f() + f(S=S - hS)) / hS**2
        vega = (f(sigma=sigma + hv) - f(sigma=sigma - hv)) / (2 * hv)
        theta = -(f(T=T + hT) - f(T=T - hT)) / (2 * hT)  # theta = dV/dt = -dV/dT
        rho = (f(r=r + hr) - f(r=r - hr)) / (2 * hr)
        assert g.delta == pytest.approx(delta, rel=1e-6, abs=1e-8)
        assert g.gamma == pytest.approx(gamma, rel=1e-4, abs=1e-8)
        assert g.vega == pytest.approx(vega, rel=1e-6, abs=1e-7)
        assert g.theta == pytest.approx(theta, rel=1e-5, abs=1e-6)
        assert g.rho == pytest.approx(rho, rel=1e-5, abs=1e-6)

    def test_call_put_greek_relationships(self):
        p = (100, 105, 0.75, 0.04, 0.02, 0.25)
        gc, gp = bs_greeks(*p, "call"), bs_greeks(*p, "put")
        S, K, T, r, q, _ = p
        assert gc.delta - gp.delta == pytest.approx(math.exp(-q * T))
        assert gc.gamma == pytest.approx(gp.gamma)
        assert gc.vega == pytest.approx(gp.vega)
        assert gc.rho - gp.rho == pytest.approx(K * T * math.exp(-r * T))

    def test_display_units(self):
        g = bs_greeks(100, 100, 1, 0.05, 0, 0.2, "call")
        d = g.display_units()
        assert d.vega == pytest.approx(g.vega / 100)
        assert d.theta == pytest.approx(g.theta / 365)
        assert d.rho == pytest.approx(g.rho / 100)

    def test_deterministic_limit_greeks_are_finite(self):
        g = bs_greeks(110, 100, 0.0, 0.05, 0.0, 0.2, "call")
        assert g.delta == 1.0 and g.gamma == 0 and g.vega == 0
