"""Tests for rolling out-of-sample forecasting diagnostics."""

import pytest

from calibration import SCIPY_AVAILABLE
from research.forecasting import rolling_forecast_to_dict, run_rolling_oos_forecast_study
from research.historical_backtest import generate_synthetic_market_option_timeseries


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy required for calibration tests")
def test_rolling_oos_forecast_runs():
    panels = generate_synthetic_market_option_timeseries(num_dates=5, seed=51)
    out = run_rolling_oos_forecast_study(
        panels,
        spot=100.0,
        rate=0.03,
        use_iv=False,
        heston_max_iter=40,
        rough_max_iter=30,
        rough_num_paths=100,
        rough_num_steps=10,
        max_transitions=3,
        seed=52,
    )
    assert out.num_transitions == 3
    assert out.best_model in out.models
    assert len(out.losses_by_model["heston"]) == 3
    assert len(out.losses_by_model["rough_heston"]) == 3
    assert len(out.losses_by_model["naive_last_surface"]) == 3

    payload = rolling_forecast_to_dict(out)
    assert payload["best_model"] in payload["models"]
    assert len(payload["target_dates"]) == 3
