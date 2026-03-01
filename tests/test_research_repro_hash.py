"""Tests for reproducibility hash bundle utilities."""

from pathlib import Path

from research.repro import (
    collect_manifest,
    verify_reproducibility_hash_bundle,
    write_json,
    write_reproducibility_hash_bundle,
)


def test_repro_hash_bundle_roundtrip(tmp_path: Path):
    manifest = collect_manifest(seed=11, cwd=tmp_path)
    write_json(tmp_path / "results.json", {"x": 1, "y": [1, 2, 3]})
    write_json(tmp_path / "claims.json", [{"a": 1}])
    bundle = write_reproducibility_hash_bundle(output_dir=tmp_path, manifest=manifest, seed=11)
    check = verify_reproducibility_hash_bundle(bundle)
    assert check["valid"] is True


def test_repro_hash_bundle_detects_change(tmp_path: Path):
    manifest = collect_manifest(seed=22, cwd=tmp_path)
    write_json(tmp_path / "results.json", {"x": 1})
    bundle = write_reproducibility_hash_bundle(output_dir=tmp_path, manifest=manifest, seed=22)
    (tmp_path / "results.json").write_text('{"x": 2}\n', encoding="utf-8")
    check = verify_reproducibility_hash_bundle(bundle)
    assert check["valid"] is False
    assert check["num_mismatched_files"] >= 1
