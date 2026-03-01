"""Integration test for research pipeline artifacts."""

from pathlib import Path

from mc_pricer import (
    DiscretizationScheme,
    MarketData,
    PayoffType,
    VarianceReduction,
)
from research.benchmark import BenchmarkCase, BenchmarkConfig
from research.pipeline import run_research_pipeline


def test_run_research_pipeline_writes_artifacts(tmp_path: Path):
    case = BenchmarkCase(
        case_id="gbm_eur_call_atm",
        name="GBM European ATM Call",
        model="gbm",
        payoff_type=PayoffType.EUROPEAN_CALL,
        market=MarketData(S0=100.0, r=0.03, q=0.0),
        strike=100.0,
        maturity=1.0,
        sigma=0.2,
    )
    cfg_baseline = BenchmarkConfig(
        label="baseline_mc",
        num_paths=1500,
        num_steps=30,
        variance_reduction=VarianceReduction.NONE,
        scheme=DiscretizationScheme.EULER,
    )
    cfg_candidate = BenchmarkConfig(
        label="vr_mc",
        num_paths=1500,
        num_steps=30,
        variance_reduction=VarianceReduction.ANTITHETIC,
        scheme=DiscretizationScheme.QE,
    )

    out = run_research_pipeline(
        output_dir=str(tmp_path / "research_out"),
        seed=7,
        quick=True,
        benchmark_repeats=2,
        mlmc_num_runs=2,
        bootstrap_runs=2,
        diagnostic_num_paths=1200,
        benchmark_cases=[case],
        benchmark_configs=[cfg_baseline, cfg_candidate],
    )

    paths = out["artifact_paths"]
    for key in ["manifest", "results", "claims", "summary"]:
        assert key in paths
        assert Path(paths[key]).exists()

    assert len(out["claims"]) == 3
    assert "performance" in out["results"]
    assert "heston_mlmc" in out["results"]
    assert "rough_heston" in out["results"]
    assert "rough_heston_full" in out["results"]
    assert "rough_bergomi" in out["results"]
    assert "identifiability" in out["results"]
    assert "svi_cleaning" in out["results"]
    assert "historical_backtest" in out["results"]
    assert "state_space_filter" in out["results"]
    assert "walkforward_leakage_free" in out["results"]
    assert "multi_year_dataset" in out["results"]
    assert "cross_sectional_study" in out["results"]
    assert "forecasting_oos" in out["results"]
    assert "challenger_baselines" in out["results"]
    assert "forecasting_oos_multi_year" in out["results"]
    assert "crisis_subperiod_study" in out["results"]
    assert "structural_break_diagnostics" in out["results"]
    assert "ablation_study" in out["results"]
    assert "global_multiple_testing" in out["results"]
    assert "regime_diagnostics" in out["results"]
    assert "real_data_calibration" in out["results"]
    assert "hedging_robustness" in out["results"]
    assert "execution_aware_hedging" in out["results"]
    assert "microstructure_hedging" in out["results"]
    assert "portfolio_overlay" in out["results"]
    assert "statistical_validation" in out["results"]
    assert "spa_vs_naive_forecast" in out["results"]["statistical_validation"]
    assert "model_confidence_set" in out["results"]["statistical_validation"]
    assert "hpc_scaling" in out["results"]
    assert "cuda_tuning" in out["results"]
    assert "market_data" in out["results"]
    assert "bayesian" in out["results"]["calibration_uq"]
    assert "publication_assets" in out["artifact_paths"]
    assert "experiment_registry" in out["artifact_paths"]
    assert "paper_package" in out["artifact_paths"]
    assert "results_chapter" in out["artifact_paths"]
    assert "traceability_package" in out["artifact_paths"]
    assert "reproducibility_hash_bundle" in out["artifact_paths"]
    assert out["artifact_paths"]["reproducibility_verification"]["valid"] is True
