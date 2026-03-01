"""Tests for regime-aware diagnostics."""

from research.historical_backtest import generate_synthetic_market_option_timeseries
from research.regime import regime_diagnostics_to_dict, run_regime_diagnostics


def test_run_regime_diagnostics_basic():
    panels = generate_synthetic_market_option_timeseries(
        start_date="2024-01-05",
        num_dates=7,
        step_days=7,
        spot=100.0,
        rate=0.03,
        seed=77,
    )

    observations = []
    for i in range(1, len(panels)):
        target = panels[i].quote_date
        observations.extend(
            [
                {"model": "heston", "target_date": target, "rmse": 0.09 + 0.002 * i},
                {"model": "rough_heston", "target_date": target, "rmse": 0.08 + 0.002 * i},
                {"model": "naive_last_surface", "target_date": target, "rmse": 0.11 + 0.003 * i},
            ]
        )

    out = run_regime_diagnostics(panels, forecast_observations=observations, spot=100.0)
    assert out.num_dates == len(panels)
    assert out.num_regimes >= 2
    assert 0.0 <= out.persistence_ratio <= 1.0
    assert len(out.features) == len(panels)
    assert out.high_vol_regime in {x.regime for x in out.features}
    assert out.high_vol_best_model in {"heston", "rough_heston", "naive_last_surface"}


def test_regime_diagnostics_to_dict_payload():
    panels = generate_synthetic_market_option_timeseries(
        start_date="2024-01-05",
        num_dates=5,
        step_days=7,
        seed=123,
    )
    observations = [{"model": "heston", "target_date": panels[i].quote_date, "rmse": 0.1} for i in range(1, 5)]
    out = run_regime_diagnostics(panels, forecast_observations=observations, spot=100.0)
    payload = regime_diagnostics_to_dict(out)
    assert "features" in payload
    assert "transition_probabilities" in payload
    assert "best_model_by_regime" in payload
