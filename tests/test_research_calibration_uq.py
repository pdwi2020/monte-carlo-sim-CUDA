"""Tests for calibration bootstrap uncertainty module."""

import pytest

from calibration import MarketOption, SCIPY_AVAILABLE, heston_call_price, implied_volatility
from research.calibration_uq import bayesian_heston_calibration, bootstrap_heston_calibration


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy required for calibration tests")
def test_bootstrap_heston_calibration_runs():
    spot = 100.0
    rate = 0.03
    params = dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.35, rho=-0.6)

    options = []
    for maturity in [0.5, 1.0]:
        for strike in [90.0, 100.0, 110.0]:
            price = heston_call_price(spot, strike, rate, maturity, **params)
            iv = implied_volatility(price, spot, strike, rate, maturity, option_type="call")
            options.append(
                MarketOption(
                    strike=strike,
                    maturity=maturity,
                    market_price=price,
                    market_iv=iv,
                    option_type="call",
                )
            )

    result = bootstrap_heston_calibration(
        options,
        spot=spot,
        rate=rate,
        use_iv=True,
        n_bootstrap=4,
        max_iter=60,
        seed=123,
    )

    assert result.successful_runs > 0
    for key in ["v0", "kappa", "theta", "xi", "rho"]:
        assert key in result.parameter_uq


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy required for calibration tests")
def test_bayesian_heston_calibration_runs():
    spot = 100.0
    rate = 0.03
    params = dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.35, rho=-0.6)

    options = []
    for maturity in [0.5, 1.0]:
        for strike in [90.0, 100.0, 110.0]:
            price = heston_call_price(spot, strike, rate, maturity, **params)
            iv = implied_volatility(price, spot, strike, rate, maturity, option_type="call")
            options.append(
                MarketOption(
                    strike=strike,
                    maturity=maturity,
                    market_price=price,
                    market_iv=iv,
                    option_type="call",
                )
            )

    result = bayesian_heston_calibration(
        options,
        spot=spot,
        rate=rate,
        use_iv=True,
        max_iter=60,
        n_samples=60,
        burn_in=40,
        proposal_scale=0.12,
        seed=321,
    )

    assert 0.0 <= result.acceptance_rate <= 1.0
    for key in ["v0", "kappa", "theta", "xi", "rho"]:
        assert key in result.parameter_uq
