"""Reproducibility and artifact helpers for research experiments."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import scipy

    SCIPY_VERSION = scipy.__version__
except Exception:  # pragma: no cover
    SCIPY_VERSION = "unavailable"


@dataclass
class RunManifest:
    """Metadata required for reproducible experiment artifacts."""

    timestamp_utc: str
    python_version: str
    platform: str
    numpy_version: str
    scipy_version: str
    seed: int
    git_commit: Optional[str]


def _current_git_commit(cwd: Optional[Path] = None) -> Optional[str]:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd is not None else None,
            stderr=subprocess.DEVNULL,
        )
        return output.decode("utf-8").strip()
    except Exception:
        return None


def collect_manifest(seed: int, cwd: Optional[Path] = None) -> RunManifest:
    """Collect runtime metadata for exact reproducibility."""

    return RunManifest(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        python_version=sys.version,
        platform=platform.platform(),
        numpy_version=np.__version__,
        scipy_version=SCIPY_VERSION,
        seed=seed,
        git_commit=_current_git_commit(cwd=cwd),
    )


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def write_json(path: Path, payload: Any) -> None:
    """Write pretty-printed deterministic JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, indent=2, sort_keys=True)
        f.write("\n")


def write_markdown_summary(path: Path, lines: list[str]) -> None:
    """Write markdown summary lines."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_research_artifacts(
    *,
    output_dir: Path,
    manifest: RunManifest,
    results: Dict[str, Any],
    claims: Any,
) -> Dict[str, str]:
    """Write all artifact files and return path map."""

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    results_path = output_dir / "results.json"
    claims_path = output_dir / "claims.json"
    summary_path = output_dir / "summary.md"

    write_json(manifest_path, manifest)
    write_json(results_path, results)
    write_json(claims_path, claims)

    summary_lines = [
        "# Research Pipeline Summary",
        "",
        f"- Timestamp (UTC): `{manifest.timestamp_utc}`",
        f"- Python: `{manifest.python_version.split()[0]}`",
        f"- NumPy: `{manifest.numpy_version}`",
        f"- SciPy: `{manifest.scipy_version}`",
        f"- Seed: `{manifest.seed}`",
        f"- Git commit: `{manifest.git_commit}`",
        "",
        "## Outputs",
        f"- `manifest.json`",
        f"- `results.json`",
        f"- `claims.json`",
    ]
    write_markdown_summary(summary_path, summary_lines)

    return {
        "manifest": str(manifest_path),
        "results": str(results_path),
        "claims": str(claims_path),
        "summary": str(summary_path),
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _deterministic_probe_hash(seed: int) -> str:
    rng = np.random.default_rng(seed)
    probe = rng.standard_normal(1024).astype(np.float64).tobytes()
    return hashlib.sha256(probe).hexdigest()


def write_reproducibility_hash_bundle(
    *,
    output_dir: Path,
    manifest: RunManifest,
    seed: int,
) -> str:
    """Write a SHA256 bundle for all run artifacts and deterministic probe."""

    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / "reproducibility_hashes.json"

    file_hashes: Dict[str, str] = {}
    for p in sorted(output_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(output_dir).as_posix()
        if rel == "reproducibility_hashes.json":
            continue
        file_hashes[rel] = _sha256_file(p)

    payload = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": manifest.git_commit,
        "seed": int(seed),
        "deterministic_probe_sha256": _deterministic_probe_hash(seed),
        "num_files_hashed": len(file_hashes),
        "file_sha256": file_hashes,
    }
    write_json(bundle_path, payload)
    return str(bundle_path)


def verify_reproducibility_hash_bundle(bundle_path: str | Path) -> Dict[str, Any]:
    """Verify hash bundle against current artifact files."""

    p = Path(bundle_path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    root = p.parent

    expected = payload.get("file_sha256") or {}
    missing: List[str] = []
    mismatched: List[Dict[str, str]] = []
    for rel, expected_hash in sorted(expected.items()):
        f = root / rel
        if not f.exists():
            missing.append(rel)
            continue
        actual = _sha256_file(f)
        if actual != expected_hash:
            mismatched.append({"path": rel, "expected": str(expected_hash), "actual": actual})

    expected_probe = str(payload.get("deterministic_probe_sha256"))
    actual_probe = _deterministic_probe_hash(int(payload.get("seed", 0)))
    probe_ok = expected_probe == actual_probe

    return {
        "valid": bool(len(missing) == 0 and len(mismatched) == 0 and probe_ok),
        "num_expected_files": int(len(expected)),
        "num_missing_files": int(len(missing)),
        "num_mismatched_files": int(len(mismatched)),
        "missing_files": missing,
        "mismatched_files": mismatched,
        "deterministic_probe_ok": probe_ok,
    }
