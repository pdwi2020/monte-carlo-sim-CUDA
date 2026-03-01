"""Tests for historical date-sliced rough-Heston backtesting."""

from research.historical_backtest import (
    generate_synthetic_market_option_timeseries,
    run_historical_rough_heston_backtest,
)


def test_historical_backtest_runs_and_splits_dates():
    panels = generate_synthetic_market_option_timeseries(
        start_date="2024-01-05",
        num_dates=7,
        step_days=7,
        seed=101,
    )
    out = run_historical_rough_heston_backtest(
        panels,
        spot=100.0,
        rate=0.03,
        use_iv=False,
        train_fraction=0.57,
        validate_fraction=0.2,
        max_iter=60,
        num_paths=250,
        num_steps=18,
        seed=102,
    )

    assert len(out.train_dates) >= 1
    assert len(out.validate_dates) >= 1
    assert len(out.test_dates) >= 1
    assert out.train_rmse >= 0.0
    assert out.validate_mean_rmse >= 0.0
    assert out.test_mean_rmse >= 0.0
    assert "hurst" in out.reference_rough_params
    assert out.parameter_drift.num_transitions >= 1
