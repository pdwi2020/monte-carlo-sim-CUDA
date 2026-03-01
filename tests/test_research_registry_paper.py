"""Tests for experiment registry and paper package generation."""

from pathlib import Path

from research.experiment_registry import write_experiment_registry
from research.paper_pack import generate_paper_package
from research.repro import collect_manifest


def test_registry_and_paper_outputs(tmp_path: Path):
    manifest = collect_manifest(seed=69, cwd=tmp_path)
    results = {
        "mlmc": {"avg_speedup": 1.2},
        "heston_mlmc": {"avg_speedup": 1.1},
        "historical_backtest": {"test_mean_rmse": 0.2},
        "forecasting_oos": {"mean_rmse_by_model": {"heston": 0.1, "naive_last_surface": 0.2}, "best_model": "heston"},
        "portfolio_overlay": {"portfolio_cvar95_loss": 1.0},
        "statistical_validation": {
            "spa_vs_naive_forecast": {"p_value": 0.03},
            "white_reality_check_vs_naive": {"p_value": 0.04},
            "model_confidence_set": {"surviving_models": ["heston"]},
            "diebold_mariano_efficiency": {"statistic": 2.1, "p_value": 0.02},
        },
        "performance": {"summary": {"max_speedup": 2.0}},
        "hpc_scaling": {"max_single_gpu_speedup": 1.0},
        "market_data": {"source": "synthetic", "status": "fallback"},
        "walkforward_leakage_free": {"best_model_by_aggregate_rmse": "heston"},
        "state_space_filter": {"mean_panel_rmse": 0.2},
    }
    reg = write_experiment_registry(output_dir=tmp_path / "registry", manifest=manifest, results=results, tags={"mode": "test"})
    assert Path(reg["metrics_csv"]).exists()

    paper = generate_paper_package(
        output_dir=tmp_path / "paper",
        results=results,
        claims=[],
        artifact_paths={"publication_assets": "x", "publication_index": "y"},
    )
    assert Path(paper["manuscript_md"]).exists()
    assert Path(paper["manuscript_tex"]).exists()
