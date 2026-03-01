"""Tests for crisis/subperiod empirical analysis."""

from research.crisis import crisis_to_dict, run_crisis_subperiod_study
from research.historical_backtest import generate_synthetic_market_option_timeseries


def test_crisis_subperiod_study_basic():
    panels = generate_synthetic_market_option_timeseries(
        start_date="2020-01-03",
        num_dates=20,
        step_days=14,
        seed=42,
    )
    observations = []
    for i in range(1, len(panels)):
        td = panels[i].quote_date
        observations.extend(
            [
                {"model": "heston", "target_date": td, "rmse": 0.11 + 0.002 * i},
                {"model": "rough_heston", "target_date": td, "rmse": 0.10 + 0.002 * i},
                {"model": "naive_last_surface", "target_date": td, "rmse": 0.15 + 0.003 * i},
            ]
        )

    out = run_crisis_subperiod_study(panels, forecast_observations=observations, spot=100.0)
    assert out.num_dates == len(panels)
    assert out.num_observations == len(observations)
    assert "full_sample" in out.best_model_by_episode
    assert out.stress_episode_id is not None
    payload = crisis_to_dict(out)
    assert "episode_performance" in payload
    assert len(payload["episodes"]) >= 2
