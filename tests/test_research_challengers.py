"""Tests for challenger baseline study (SABR + SSVI)."""

from research.challengers import challenger_to_dict, run_challenger_baseline_study
from research.historical_backtest import generate_synthetic_market_option_timeseries


def test_challenger_baseline_study_basic():
    panels = generate_synthetic_market_option_timeseries(
        start_date="2020-01-03",
        num_dates=8,
        step_days=14,
        seed=99,
    )
    out = run_challenger_baseline_study(
        panels,
        spot=100.0,
        rate=0.03,
        use_iv=True,
        max_transitions=4,
    )
    assert out.num_transitions >= 2
    assert set(out.models) == {"sabr_hagan_surface", "ssvi_carry_surface"}
    assert out.best_model in out.models
    payload = challenger_to_dict(out)
    assert "mean_rmse_by_model" in payload
    assert len(payload["observations"]) >= 4
