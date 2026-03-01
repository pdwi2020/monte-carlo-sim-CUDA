"""Tests for paper-ready results chapter generator."""

from pathlib import Path

from research.results_chapter import generate_results_chapter


def test_generate_results_chapter(tmp_path: Path):
    results = {
        "forecasting_oos": {"best_model": "heston"},
        "challenger_baselines": {"best_model": "ssvi_carry_surface"},
        "crisis_subperiod_study": {"stress_episode_id": "covid_crash_2020", "stress_best_model": "heston"},
        "ablation_study": {"top_positive_impact_scenario": "remove_x"},
        "structural_break_diagnostics": {"strongest_break_series": "forecast::heston", "strongest_break_p_value": 0.03},
        "global_multiple_testing": {"num_tests": 5, "num_reject_holm": 2, "num_reject_bh": 3},
        "cuda_tuning": {"status": "not_available"},
    }
    claims = [{"claim": {"claim_id": "C1", "title": "demo"}, "evaluation": {"passed": True, "p_value": 0.01}}]
    artifacts = {"publication_files": {}, "reproducibility_hash_bundle": "x", "reproducibility_verification": {"valid": True}}
    out = generate_results_chapter(
        output_dir=tmp_path / "chapter",
        results=results,
        claims=claims,
        artifact_paths=artifacts,
    )
    assert Path(out["results_chapter_md"]).exists()
    assert Path(out["results_chapter_tex"]).exists()
