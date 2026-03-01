"""Tests for full rough-Heston calibration engine."""

from calibration import MarketOption, heston_call_price
from research.rough_full_calibration import calibrate_rough_heston_full


def test_calibrate_rough_heston_full_runs():
    spot = 100.0
    rate = 0.03
    true = dict(v0=0.04, kappa=2.1, theta=0.04, xi=0.34, rho=-0.6)
    options = []
    for maturity in [0.5, 1.0]:
        for strike in [90.0, 100.0, 110.0]:
            price = heston_call_price(spot, strike, rate, maturity, **true)
            options.append(MarketOption(strike=strike, maturity=maturity, market_price=price, option_type="call"))

    out = calibrate_rough_heston_full(
        options,
        spot=spot,
        rate=rate,
        use_iv=False,
        max_iter=50,
        num_paths=260,
        num_steps=14,
        n_global_samples=5,
        local_iterations=3,
        seed=14,
    )
    assert out.rmse >= 0.0
    assert "hurst" in out.best_params
    assert len(out.objective_trace) >= 1
