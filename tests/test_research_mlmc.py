"""Tests for MLMC research module."""

from mc_pricer import HestonParams, black_scholes_call
from research.mlmc import compare_mlmc_heston_vs_mc, compare_mlmc_vs_mc, mlmc_price_gbm_option


def test_mlmc_price_reasonable_vs_black_scholes():
    res = mlmc_price_gbm_option(
        s0=100.0,
        strike=100.0,
        r=0.03,
        sigma=0.2,
        maturity=1.0,
        is_call=True,
        payoff_style="european",
        max_level=4,
        min_level_samples=1200,
        seed=123,
    )
    bs = black_scholes_call(100.0, 100.0, 0.03, 0.2, 1.0)
    assert abs(res.price - bs) < 1.2
    assert res.std_error > 0


def test_mlmc_vs_mc_comparison_payload():
    out = compare_mlmc_vs_mc(seed=321)
    assert "mlmc" in out and "mc" in out
    assert out["mlmc"]["runtime_seconds"] > 0
    assert out["mc"]["runtime_seconds"] > 0
    assert out["abs_price_gap"] >= 0


def test_heston_mlmc_vs_mc_comparison_payload():
    out = compare_mlmc_heston_vs_mc(
        heston=HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.4, rho=-0.6),
        seed=456,
    )
    assert "mlmc" in out and "mc" in out
    assert out["mlmc"]["runtime_seconds"] > 0
    assert out["mc"]["runtime_seconds"] > 0
    assert out["abs_price_gap"] >= 0
