"""Tests for leakage-free walk-forward protocol."""

import pytest

from calibration import SCIPY_AVAILABLE
from research.historical_backtest import generate_synthetic_market_option_timeseries
from research.walkforward import build_walkforward_windows, run_leakage_free_walkforward, walkforward_to_dict


def test_build_walkforward_windows_shapes():
    dates = [f"2024-01-{i:02d}" for i in range(1, 21)]
    windows = build_walkforward_windows(dates, train_size=8, validate_size=2, test_size=2, step_size=2)
    assert len(windows) >= 3
    w0 = windows[0]
    assert len(w0.train_dates) == 8
    assert len(w0.validate_dates) == 2
    assert len(w0.test_dates) == 2


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy required for calibration tests")
def test_run_leakage_free_walkforward_runs():
    panels = generate_synthetic_market_option_timeseries(num_dates=10, step_days=7, seed=62)
    out = run_leakage_free_walkforward(
        panels,
        spot=100.0,
        rate=0.03,
        train_size=6,
        validate_size=2,
        test_size=2,
        step_size=2,
        max_windows=2,
        heston_max_iter=30,
        rough_max_iter=24,
        rough_num_paths=80,
        rough_num_steps=8,
        seed=63,
    )
    assert out.num_windows >= 1
    assert out.best_model_by_aggregate_rmse in {"heston", "rough_heston", "naive_last_surface"}
    payload = walkforward_to_dict(out)
    assert len(payload["windows"]) == out.num_windows
