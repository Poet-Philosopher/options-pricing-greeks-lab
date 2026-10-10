"""CRR binomial tree: convergence to BSM, American vs European behaviour."""
import pytest

from optionslab import bs_price, crr_convergence, crr_price, crr_tree, early_exercise_premium

P = dict(S=100.0, K=100.0, T=1.0, r=0.05, q=0.02, sigma=0.2)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_european_crr_converges_to_bsm(kind):
    exact = bs_price(**P, kind=kind)
    errs = [abs(crr_price(**P, n_steps=n, kind=kind) - exact) for n in (25, 100, 400, 1600)]
    assert errs[-1] < 0.005
    assert errs[-1] < errs[0]


def test_convergence_dataframe():
    df = crr_convergence(**P, steps=[50, 200, 800], kind="put")
    assert list(df.columns) == ["steps", "european", "american", "bsm", "euro_error", "premium"]
    assert abs(df.euro_error.iloc[-1]) < abs(df.euro_error.iloc[0]) + 1e-12
    assert (df.premium >= -1e-12).all()


def test_american_call_without_dividends_equals_european():
    kw = dict(P, q=0.0)
    assert crr_price(**kw, n_steps=300, kind="call", american=True) == pytest.approx(
        crr_price(**kw, n_steps=300, kind="call"), abs=1e-9)


def test_american_put_has_positive_early_exercise_premium():
    prem = early_exercise_premium(S=100, K=100, T=1, r=0.08, q=0.0, sigma=0.2, n_steps=400, kind="put")
    assert prem > 0.1


def test_american_put_dominates_intrinsic_and_european():
    for S in (60, 80, 100, 120):
        a = crr_price(S, 100, 1, 0.06, 0, 0.3, 200, "put", True)
        e = crr_price(S, 100, 1, 0.06, 0, 0.3, 200, "put", False)
        assert a >= e - 1e-12 and a >= max(100 - S, 0) - 1e-12


def test_deep_itm_american_put_equals_intrinsic():
    assert crr_price(30, 100, 1, 0.08, 0, 0.2, 200, "put", True) == pytest.approx(70.0, abs=1e-9)


def test_hull_five_step_american_put():
    # Hull Example: S=K=50, r=10%, sigma=40%, T=5/12, 5 steps -> 4.49
    assert crr_price(50, 50, 5 / 12, 0.10, 0.0, 0.40, 5, "put", True) == pytest.approx(4.49, abs=0.01)


def test_tree_delta_gamma_match_bsm():
    from optionslab import bs_greeks
    t = crr_tree(**P, n_steps=800, kind="call")
    g = bs_greeks(**P, kind="call")
    assert t.delta == pytest.approx(g.delta, abs=2e-3)
    assert t.gamma == pytest.approx(g.gamma, abs=5e-4)


def test_invalid_probability_raises_helpful_error():
    with pytest.raises(ValueError, match="outside"):
        crr_price(100, 100, 5.0, 0.30, 0.0, 0.01, 10, "call")


def test_requires_two_steps():
    with pytest.raises(ValueError):
        crr_price(100, 100, 1, 0.05, 0, 0.2, 1, "call")
