"""Tests for publication asset generation."""

from pathlib import Path

from research.reporting import generate_publication_assets


def test_generate_publication_assets_writes_tables(tmp_path: Path):
    results = {
        "mlmc": {"avg_speedup": 1.5},
        "heston_mlmc": {"avg_speedup": 2.0},
        "rough_heston": {"improvement_ratio": 0.01},
        "forecasting_oos": {
            "best_model": "rough_heston",
            "worst_model": "naive_last_surface",
            "target_dates": ["2024-02-01", "2024-02-08"],
            "mean_rmse_by_model": {"heston": 0.19, "rough_heston": 0.17, "naive_last_surface": 0.27},
            "std_rmse_by_model": {"heston": 0.02, "rough_heston": 0.01, "naive_last_surface": 0.03},
            "losses_by_model": {
                "heston": [0.18, 0.20],
                "rough_heston": [0.16, 0.18],
                "naive_last_surface": [0.25, 0.29],
            },
        },
        "forecasting_oos_multi_year": {
            "best_model": "rough_heston",
            "worst_model": "naive_last_surface",
            "target_dates": ["2020-03-01", "2020-03-08"],
            "mean_rmse_by_model": {"heston": 0.21, "rough_heston": 0.19, "naive_last_surface": 0.3},
            "std_rmse_by_model": {"heston": 0.01, "rough_heston": 0.01, "naive_last_surface": 0.02},
            "losses_by_model": {
                "heston": [0.20, 0.22],
                "rough_heston": [0.18, 0.20],
                "naive_last_surface": [0.29, 0.31],
            },
        },
        "challenger_baselines": {
            "best_model": "ssvi_carry_surface",
            "worst_model": "sabr_hagan_surface",
            "mean_rmse_by_model": {"sabr_hagan_surface": 0.22, "ssvi_carry_surface": 0.2},
            "std_rmse_by_model": {"sabr_hagan_surface": 0.02, "ssvi_carry_surface": 0.015},
        },
        "historical_backtest": {
            "validate_mean_rmse": 0.2,
            "test_mean_rmse": 0.25,
            "validate_evaluations": [{"quote_date": "2024-01-01", "rmse": 0.2, "num_options": 10}],
            "test_evaluations": [{"quote_date": "2024-02-01", "rmse": 0.25, "num_options": 10}],
            "date_calibrations": [{"quote_date": "2024-01-01", "rmse": 0.2, "hurst": 0.3}],
        },
        "state_space_filter": {
            "estimates": [
                {"quote_date": "2024-01-01", "panel_rmse": 0.2, "raw_panel_rmse": 0.22},
            ]
        },
        "crisis_subperiod_study": {
            "stress_episode_id": "high_stress_quantile",
            "stress_best_model": "rough_heston",
            "episode_performance": [
                {"episode_id": "full_sample", "model": "heston", "mean_rmse": 0.2, "std_rmse": 0.01, "count": 10},
                {
                    "episode_id": "high_stress_quantile",
                    "model": "rough_heston",
                    "mean_rmse": 0.18,
                    "std_rmse": 0.01,
                    "count": 4,
                },
            ],
            "dm_tests": [
                {
                    "episode_id": "high_stress_quantile",
                    "best_model": "rough_heston",
                    "benchmark_model": "naive_last_surface",
                    "statistic": -2.0,
                    "p_value": 0.04,
                    "n": 8,
                }
            ],
        },
        "ablation_study": {
            "top_positive_impact_scenario": "remove_rough_model_class",
            "mean_effect_across_scenarios": 0.03,
            "scenarios": [
                {
                    "scenario_id": "remove_rough_model_class",
                    "metric": "oos_forecast_rmse",
                    "objective": "lower_better",
                    "baseline_mean": 0.18,
                    "ablated_mean": 0.22,
                    "effect_mean": 0.04,
                    "effect_ci_low": 0.02,
                    "effect_ci_high": 0.06,
                    "p_value": 0.02,
                    "n": 10,
                    "interpretation": "positive means ablation worsens",
                }
            ],
        },
        "cuda_tuning": {
            "status": "ok",
            "best_speedup_over_baseline": 1.15,
            "candidates": [
                {"variant": "mixed_precision", "threads_per_block": 128, "num_streams": 2, "runtime_seconds": 0.01, "price": 5.0}
            ],
        },
        "structural_break_diagnostics": {
            "strongest_break_series": "forecast::heston",
            "strongest_break_p_value": 0.03,
            "entries": [
                {
                    "series_id": "forecast::heston",
                    "n": 12,
                    "break_index": 6,
                    "break_date": "2020-06-01",
                    "pre_mean": 0.12,
                    "post_mean": 0.16,
                    "mean_shift": 0.04,
                    "t_stat": 2.4,
                    "p_value": 0.03,
                    "cusum_max_abs": 1.2,
                }
            ],
        },
        "global_multiple_testing": {
            "num_tests": 8,
            "num_reject_holm": 2,
            "num_reject_bh": 3,
            "records": [
                {
                    "test_id": "claim::C1",
                    "family": "claims",
                    "raw_p_value": 0.01,
                    "holm_adjusted_p_value": 0.02,
                    "holm_reject": True,
                    "bh_adjusted_p_value": 0.02,
                    "bh_reject": True,
                }
            ],
        },
        "hedging_robustness": {
            "cvar95_loss_ratio_misspecified_over_well_specified": 1.8,
            "cvar95_loss_ratio_delta_vega_over_delta_only_misspecified": 0.9,
            "transaction_cost_frontier_heston": [
                {"transaction_cost": 0.0, "delta_only_cvar95_loss": 1.0, "delta_vega_cvar95_loss": 0.8}
            ],
            "rebalance_stability_heston": [
                {"rebalance_every_steps": 1, "delta_only_cvar95_loss": 1.0, "delta_vega_cvar95_loss": 0.8}
            ],
        },
        "statistical_validation": {
            "diebold_mariano_efficiency": {"statistic": 2.0, "p_value": 0.04},
            "spa_vs_naive_forecast": {
                "statistic": 2.8,
                "p_value": 0.01,
                "benchmark_model": "naive_last_surface",
                "best_model": "rough_heston",
            },
            "white_reality_check_vs_naive": {
                "statistic": 0.09,
                "p_value": 0.03,
                "benchmark_model": "naive_last_surface",
                "best_model": "rough_heston",
            },
            "model_confidence_set": {"surviving_models": ["rough_heston", "heston"]},
        },
    }
    claims = [
        {
            "claim": {"claim_id": "C1", "title": "demo", "metric": "x", "minimum_effect": 0.1},
            "evaluation": {"observed_effect": 0.2, "passed": True, "p_value": 0.01, "confidence_interval": {"low": 0.1, "high": 0.3}},
        }
    ]

    out = generate_publication_assets(output_dir=tmp_path / "pub", results=results, claims=claims)
    assert "claims_table_csv" in out
    assert Path(out["claims_table_csv"]).exists()
    assert "claims_table_tex" in out
    assert Path(out["claims_table_tex"]).exists()
    assert "metrics_table_csv" in out
    assert Path(out["metrics_table_csv"]).exists()
    assert "metrics_table_tex" in out
    assert Path(out["metrics_table_tex"]).exists()
    assert "historical_eval_csv" in out
    assert Path(out["historical_eval_csv"]).exists()
    assert "forecast_leaderboard_csv" in out
    assert Path(out["forecast_leaderboard_csv"]).exists()
    assert "challenger_leaderboard_csv" in out
    assert Path(out["challenger_leaderboard_csv"]).exists()
    assert "crisis_episode_performance_csv" in out
    assert Path(out["crisis_episode_performance_csv"]).exists()
    assert "ablation_study_csv" in out
    assert Path(out["ablation_study_csv"]).exists()
    assert "cuda_tuning_candidates_csv" in out
    assert Path(out["cuda_tuning_candidates_csv"]).exists()
    assert "structural_breaks_csv" in out
    assert Path(out["structural_breaks_csv"]).exists()
    assert "global_multiple_testing_csv" in out
    assert Path(out["global_multiple_testing_csv"]).exists()
    assert "econometrics_table_csv" in out
    assert Path(out["econometrics_table_csv"]).exists()
