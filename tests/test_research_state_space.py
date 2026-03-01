"""Tests for sequential/state-space calibration module."""

import pytest

from calibration import SCIPY_AVAILABLE
from research.historical_backtest import generate_synthetic_market_option_timeseries
from research.state_space import run_heston_state_space_filter, state_space_to_dict


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy required for calibration tests")
def test_state_space_filter_runs():
    panels = generate_synthetic_market_option_timeseries(num_dates=6, step_days=7, seed=66)
    out = run_heston_state_space_filter(
        panels,
        spot=100.0,
        rate=0.03,
        use_iv=False,
        max_iter=30,
        process_noise=0.03,
        measurement_noise=0.08,
    )
    assert out.num_dates == 6
    assert out.mean_panel_rmse >= 0.0
    payload = state_space_to_dict(out)
    assert len(payload["estimates"]) == 6
