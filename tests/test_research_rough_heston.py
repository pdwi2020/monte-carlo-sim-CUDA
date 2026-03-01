"""Tests for rough-Heston research hooks."""

import pytest

from calibration import MarketOption, SCIPY_AVAILABLE, heston_call_price
from research.rough_heston import (
    RoughHestonParams,
    calibrate_rough_heston_hook,
    rough_heston_price_option,
)


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy required for calibration tests")
def test_rough_heston_calibration_hook_runs():
    spot = 100.0
    rate = 0.03
    true = dict(v0=0.04, kappa=2.2, theta=0.04, xi=0.35, rho=-0.6)

    options = []
    for maturity in [0.5, 1.0]:
        for strike in [90.0, 100.0, 110.0]:
            price = heston_call_price(spot, strike, rate, maturity, **true)
            options.append(
                MarketOption(
                    strike=strike,
                    maturity=maturity,
                    market_price=price,
                    option_type="call",
                )
            )

    out = calibrate_rough_heston_hook(
        options,
        spot=spot,
        rate=rate,
        use_iv=False,
        max_iter=60,
        hurst_grid=[0.2, 0.5],
        xi_scales=[0.9, 1.1],
        num_paths=250,
        num_steps=18,
        seed=11,
    )

    assert out.rough_rmse >= 0.0
    assert "hurst" in out.best_params
    assert len(out.grid_results) == 4


def test_rough_heston_price_option_runs():
    params = RoughHestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.4, rho=-0.7, hurst=0.2)
    out = rough_heston_price_option(
        s0=100.0,
        strike=100.0,
        r=0.03,
        maturity=1.0,
        params=params,
        num_paths=500,
        num_steps=24,
        seed=9,
    )
    assert out.price > 0.0
    assert out.std_error >= 0.0
