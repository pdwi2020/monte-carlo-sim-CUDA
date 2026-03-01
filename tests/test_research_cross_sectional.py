"""Tests for cross-sectional rough-Heston studies."""

from research.cross_sectional import cross_sectional_study_to_dict, run_cross_sectional_rough_heston_study


def test_cross_sectional_study_runs():
    out = run_cross_sectional_rough_heston_study(
        symbols=["AAA", "BBB", "CCC"],
        num_dates=6,
        step_days=7,
        spot=100.0,
        rate=0.03,
        max_iter=45,
        num_paths=130,
        num_steps=12,
        seed=17,
    )
    assert len(out.asset_results) == 3
    assert out.best_symbol_by_test_rmse in {"AAA", "BBB", "CCC"}
    assert out.worst_symbol_by_test_rmse in {"AAA", "BBB", "CCC"}
    assert len(out.ranking_by_test_rmse) == 3

    payload = cross_sectional_study_to_dict(out)
    assert len(payload["asset_results"]) == 3
    assert payload["mean_test_rmse"] >= 0.0
