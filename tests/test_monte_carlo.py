"""Monte Carlo: agreement with BSM, CI coverage, variance reduction, Asian options."""
import numpy as np
import pytest

from optionslab import (
    bs_price, convergence_series, geometric_asian_price, mc_asian_arithmetic, mc_european, simulate_gbm_paths,
)

ARGS = dict(S=100.0, K=105.0, T=1.0, r=0.05, q=0.02, sigma=0.25)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("antithetic", [True, False])
def test_mc_within_ci_of_bsm(kind, antithetic):
    res = mc_european(**ARGS, kind=kind, n_paths=400_000, antithetic=antithetic, seed=7)
    exact = bs_price(**ARGS, kind=kind)
    assert abs(res.price - exact) < 4 * res.std_error  # robust 4-SE band
    assert res.ci_low < res.price < res.ci_high


def test_ci_coverage_is_close_to_nominal_95():
    exact = bs_price(**ARGS, kind="call")
    hits = sum(mc_european(**ARGS, kind="call", n_paths=4_000, seed=s).contains(exact) for s in range(300))
    assert 0.90 <= hits / 300 <= 0.99


def test_antithetic_reduces_standard_error():
    plain = mc_european(**ARGS, kind="call", n_paths=200_000, antithetic=False, seed=1)
    anti = mc_european(**ARGS, kind="call", n_paths=200_000, antithetic=True, seed=1)
    assert anti.std_error < plain.std_error


def test_reproducible_with_seed():
    a = mc_european(**ARGS, kind="put", n_paths=10_000, seed=123)
    b = mc_european(**ARGS, kind="put", n_paths=10_000, seed=123)
    assert a.price == b.price


def test_convergence_series_shape_and_trend():
    df = convergence_series(**ARGS, kind="call", n_max=100_000, seed=3)
    assert df.n_paths.is_monotonic_increasing and len(df) > 20
    assert df.std_error.iloc[-1] < df.std_error.iloc[0] / 10
    assert abs(df.price.iloc[-1] - bs_price(**ARGS, kind="call")) < 4 * df.std_error.iloc[-1]


def test_paths_shape_start_and_martingale_mean():
    t, paths = simulate_gbm_paths(100, 0.05, 0.02, 0.2, 1.0, n_paths=40_000, n_steps=20, seed=5, antithetic=True)
    assert paths.shape == (40_000, 21) and t[-1] == 1.0 and np.all(paths[:, 0] == 100)
    # E[S_T] = S exp((r-q)T)
    assert paths[:, -1].mean() == pytest.approx(100 * np.exp(0.03), rel=0.01)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        mc_european(100, 100, -1, 0.05, 0, 0.2)
    with pytest.raises(ValueError):
        mc_european(100, 100, 1, 0.05, 0, 0.2, n_paths=1)


class TestAsian:
    def test_geometric_asian_with_one_fixing_equals_european(self):
        for kind in ("call", "put"):
            assert geometric_asian_price(100, 100, 1, 0.05, 0.01, 0.2, 1, kind) == pytest.approx(
                bs_price(100, 100, 1, 0.05, 0.01, 0.2, kind), rel=1e-12)

    def test_control_variate_is_consistent_and_tighter(self):
        kw = dict(S=100, K=100, T=1.0, r=0.05, q=0.0, sigma=0.2, kind="call", n_paths=100_000, n_steps=50, seed=11)
        plain = mc_asian_arithmetic(**kw, antithetic=True, control_variate=False)
        cv = mc_asian_arithmetic(**kw, antithetic=True, control_variate=True)
        assert cv.std_error < plain.std_error / 5
        assert abs(cv.price - plain.price) < 4 * plain.std_error

    def test_asian_call_cheaper_than_european_and_above_geometric(self):
        kw = dict(S=100, K=100, T=1.0, r=0.05, q=0.0, sigma=0.3)
        asian = mc_asian_arithmetic(**kw, kind="call", n_paths=200_000, n_steps=52, control_variate=True, seed=2)
        assert asian.price < bs_price(**kw, kind="call")
        assert asian.price > geometric_asian_price(100, 100, 1.0, 0.05, 0.0, 0.3, 52, "call")  # AM-GM

    def test_asian_with_one_step_matches_european(self):
        res = mc_asian_arithmetic(100, 100, 1, 0.05, 0.0, 0.2, "put", n_paths=400_000, n_steps=1, seed=9)
        assert abs(res.price - bs_price(100, 100, 1, 0.05, 0.0, 0.2, "put")) < 4 * res.std_error
