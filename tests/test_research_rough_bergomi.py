"""Tests for rough-Bergomi baseline calibration."""

import pytest

from calibration import SCIPY_AVAILABLE
from research.real_data_calibration import generate_synthetic_option_chain, quotes_to_market_options
from research.rough_bergomi import calibrate_rough_bergomi_hook, rough_bergomi_to_dict


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy required for calibration tests")
def test_rough_bergomi_hook_runs():
    quotes = generate_synthetic_option_chain(seed=65)
    opts = quotes_to_market_options(quotes)
    out = calibrate_rough_bergomi_hook(
        opts,
        spot=100.0,
        rate=0.03,
        use_iv=False,
        max_iter=40,
        hurst_grid=[0.12, 0.3, 0.5],
        eta_grid=[0.25, 0.5],
    )
    assert out.rough_bergomi_rmse >= 0.0
    payload = rough_bergomi_to_dict(out)
    assert "best_params" in payload
