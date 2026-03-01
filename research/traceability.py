"""Claim-to-code traceability bundle for interview/defense packages."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List


_DEFAULT_CLAIM_CODE_MAP = {
    "C1": [
        ("research/benchmark.py", "run_benchmark"),
        ("research/claims.py", "evaluate_claim_bundle"),
    ],
    "C2": [
        ("research/mlmc.py", "compare_mlmc_heston_vs_mc"),
        ("research/pipeline.py", "run_research_pipeline"),
    ],
    "C3": [
        ("research/calibration_uq.py", "bootstrap_heston_calibration"),
        ("research/calibration_uq.py", "bayesian_heston_calibration"),
    ],
}


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    return str(path)


def _claim_rows(claims: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in claims or []:
        claim = item.get("claim", {}) if isinstance(item, dict) else {}
        ev = item.get("evaluation", {}) if isinstance(item, dict) else {}
        cid = str(claim.get("claim_id", ""))
        mapping = _DEFAULT_CLAIM_CODE_MAP.get(cid, [])
        if not mapping:
            mapping = [("research/pipeline.py", "run_research_pipeline")]
        for file_path, symbol in mapping:
            rows.append(
                {
                    "claim_id": cid,
                    "claim_title": claim.get("title"),
                    "metric": claim.get("metric"),
                    "minimum_effect": claim.get("minimum_effect"),
                    "observed_effect": ev.get("observed_effect"),
                    "p_value": ev.get("p_value"),
                    "passed": ev.get("passed"),
                    "code_file": file_path,
                    "code_symbol": symbol,
                }
            )
    return rows


def generate_traceability_package(
    *,
    output_dir: Path,
    claims: List[Dict[str, Any]],
    results: Dict[str, Any],
    artifact_paths: Dict[str, Any],
) -> Dict[str, str]:
    """Generate claim-to-code CSV and interview/defense briefing docs."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    claim_rows = _claim_rows(claims)
    csv_path = _write_csv(out / "claim_code_traceability.csv", claim_rows)

    md = out / "defense_brief.md"
    lines: List[str] = []
    lines.append("# Defense Brief")
    lines.append("")
    lines.append("## Key Story")
    lines.append("- End-to-end reproducible derivatives research pipeline with econometric validation.")
    lines.append("- Explicit ablations, crisis subperiod analysis, structural-break checks, and global multiple-testing control.")
    lines.append("")
    lines.append("## Claim-to-Code Map")
    for row in claim_rows:
        lines.append(
            f"- `{row['claim_id']}` -> `{row['code_file']}::{row['code_symbol']}` "
            f"(passed={row['passed']}, p={row['p_value']})"
        )
    lines.append("")
    lines.append("## Artifact Anchors")
    for key in ("results", "claims", "summary", "publication_assets", "reproducibility_hash_bundle"):
        if key in artifact_paths:
            lines.append(f"- `{key}`: `{artifact_paths[key]}`")
    lines.append("")
    lines.append("## Rapid Q&A Prompts")
    lines.append("- Why does model ranking change across crisis vs full sample?")
    lines.append("- How does global multiple-testing alter significance conclusions?")
    lines.append("- What is the strongest structural break and what does it imply?")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    qna = out / "interview_qna.md"
    qna_lines = [
        "# Interview Q&A Pack",
        "",
        "## Methodology",
        "- Explain leakage-free walk-forward setup and why it matters.",
        "- Explain difference between DM, SPA, White Reality Check, and MCS.",
        "",
        "## Robustness",
        "- Explain ablation interpretation and top degradation scenario.",
        "- Explain structural-break test design and bootstrap significance.",
        "",
        "## Engineering",
        "- Explain reproducibility hash bundle and CI verification.",
        "- Explain CUDA tuning strategy and constraints when CUDA module is unavailable.",
    ]
    qna.write_text("\n".join(qna_lines) + "\n", encoding="utf-8")

    return {
        "claim_code_traceability_csv": str(csv_path),
        "defense_brief_md": str(md),
        "interview_qna_md": str(qna),
    }
