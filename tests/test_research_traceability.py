"""Tests for claim-to-code traceability package generation."""

from pathlib import Path

from research.traceability import generate_traceability_package


def test_generate_traceability_package(tmp_path: Path):
    claims = [
        {"claim": {"claim_id": "C1", "title": "efficiency", "metric": "m"}, "evaluation": {"passed": True, "p_value": 0.02}},
        {"claim": {"claim_id": "C9", "title": "other", "metric": "x"}, "evaluation": {"passed": False, "p_value": 0.2}},
    ]
    out = generate_traceability_package(
        output_dir=tmp_path / "trace",
        claims=claims,
        results={},
        artifact_paths={"results": "a", "claims": "b"},
    )
    assert Path(out["claim_code_traceability_csv"]).exists()
    assert Path(out["defense_brief_md"]).exists()
    assert Path(out["interview_qna_md"]).exists()
