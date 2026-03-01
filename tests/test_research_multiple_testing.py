"""Tests for global multiple-testing correction layer."""

from research.multiple_testing import multiple_testing_to_dict, run_global_multiple_testing


def test_global_multiple_testing_collects_and_adjusts():
    results = {
        "statistical_validation": {
            "diebold_mariano_efficiency": {"p_value": 0.04},
            "spa_vs_naive_forecast": {"p_value": 0.03},
            "white_reality_check_vs_naive": {"p_value": 0.06},
        },
        "crisis_subperiod_study": {"dm_tests": [{"episode_id": "covid_crash_2020", "p_value": 0.02}]},
        "ablation_study": {"scenarios": [{"scenario_id": "remove_x", "p_value": 0.01}]},
    }
    claims = [
        {"claim": {"claim_id": "C1"}, "evaluation": {"claim_id": "C1", "p_value": 0.04}},
        {"claim": {"claim_id": "C2"}, "evaluation": {"claim_id": "C2", "p_value": 0.20}},
    ]
    out = run_global_multiple_testing(results=results, claims=claims, alpha=0.05)
    assert out.num_tests >= 6
    assert out.num_reject_bh >= out.num_reject_holm
    payload = multiple_testing_to_dict(out)
    assert len(payload["records"]) == out.num_tests
