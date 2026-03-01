"""Tests for ablation-study builder."""

from research.ablation import ablation_to_dict, run_ablation_study_from_results


def test_ablation_study_from_results_basic():
    results = {
        "forecasting_oos": {
            "losses_by_model": {
                "heston": [0.14, 0.15, 0.16, 0.17, 0.15],
                "rough_heston": [0.12, 0.13, 0.14, 0.15, 0.14],
                "naive_last_surface": [0.20, 0.19, 0.21, 0.22, 0.20],
            }
        },
        "state_space_filter": {
            "estimates": [
                {"quote_date": "2024-01-01", "panel_rmse": 0.11, "raw_panel_rmse": 0.13},
                {"quote_date": "2024-01-08", "panel_rmse": 0.10, "raw_panel_rmse": 0.12},
                {"quote_date": "2024-01-15", "panel_rmse": 0.12, "raw_panel_rmse": 0.14},
            ]
        },
        "microstructure_hedging": {
            "frictionless": {"cvar95_loss": 1.0},
            "microstructure_stressed": {"cvar95_loss": 1.2},
        },
        "execution_aware_hedging": {
            "frictionless": {"misspecified_heston": {"cvar95_loss": 1.05}},
            "execution_stressed": {"misspecified_heston": {"cvar95_loss": 1.3}},
        },
    }
    out = run_ablation_study_from_results(results, n_bootstrap=100, seed=7)
    assert out.num_scenarios >= 4
    assert out.top_positive_impact_scenario is not None
    payload = ablation_to_dict(out)
    assert "scenarios" in payload
    assert len(payload["scenarios"]) == out.num_scenarios
