"""Generate a paper-ready results chapter from pipeline artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List


def _metric_line(name: str, val: Any) -> str:
    if isinstance(val, float):
        return f"- {name}: {val:.6g}"
    return f"- {name}: {val}"


def _top_claim_lines(claims: Iterable[Dict[str, Any]]) -> List[str]:
    rows = []
    for item in claims or []:
        claim = item.get("claim", {}) if isinstance(item, dict) else {}
        ev = item.get("evaluation", {}) if isinstance(item, dict) else {}
        cid = claim.get("claim_id")
        title = claim.get("title")
        passed = ev.get("passed")
        p = ev.get("p_value")
        rows.append((cid, title, passed, p))
    out = []
    for cid, title, passed, p in rows:
        out.append(f"- `{cid}` {title} | passed={passed} | p={p}")
    return out


def generate_results_chapter(
    *,
    output_dir: Path,
    results: Dict[str, Any],
    claims: List[Dict[str, Any]],
    artifact_paths: Dict[str, Any],
) -> Dict[str, str]:
    """Write paper-ready chapter draft in Markdown + LaTeX."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    md = out / "results_chapter.md"
    tex = out / "results_chapter.tex"

    mtest = results.get("global_multiple_testing", {})
    breaks = results.get("structural_break_diagnostics", {})
    crisis = results.get("crisis_subperiod_study", {})
    ablation = results.get("ablation_study", {})
    forecast = results.get("forecasting_oos", {})
    challengers = results.get("challenger_baselines", {})
    cuda = results.get("cuda_tuning", {})

    lines: List[str] = []
    lines.append("# Results Chapter")
    lines.append("")
    lines.append("## 1. Primary Findings")
    lines.extend(_metric_line(k, v) for k, v in [
        ("Forecast best model", forecast.get("best_model")),
        ("Challenger best model", challengers.get("best_model")),
        ("Crisis stress episode", crisis.get("stress_episode_id")),
        ("Crisis stress best model", crisis.get("stress_best_model")),
        ("Ablation top scenario", ablation.get("top_positive_impact_scenario")),
        ("Structural strongest break series", breaks.get("strongest_break_series")),
        ("Structural strongest break p-value", breaks.get("strongest_break_p_value")),
        ("Global tests (N)", mtest.get("num_tests")),
        ("Global Holm rejections", mtest.get("num_reject_holm")),
        ("Global BH rejections", mtest.get("num_reject_bh")),
        ("CUDA tuning status", cuda.get("status")),
        ("CUDA best speedup", cuda.get("best_speedup_over_baseline")),
    ])
    lines.append("")
    lines.append("## 2. Claims Assessment")
    lines.extend(_top_claim_lines(claims))
    lines.append("")
    lines.append("## 3. Reproducibility")
    lines.append(f"- Repro hash bundle: `{artifact_paths.get('reproducibility_hash_bundle')}`")
    lines.append(f"- Repro verification: `{artifact_paths.get('reproducibility_verification')}`")
    lines.append("")
    lines.append("## 4. Tables/Figures Index")
    pub = artifact_paths.get("publication_files", {})
    if isinstance(pub, dict):
        for k, v in sorted(pub.items()):
            lines.append(f"- `{k}`: `{v}`")

    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    tex_lines = [
        "\\section{Results}",
        "\\subsection{Primary Findings}",
        "\\begin{itemize}",
    ]
    for k, v in [
        ("Forecast best model", forecast.get("best_model")),
        ("Challenger best model", challengers.get("best_model")),
        ("Crisis stress episode", crisis.get("stress_episode_id")),
        ("Ablation top scenario", ablation.get("top_positive_impact_scenario")),
        ("Structural strongest break series", breaks.get("strongest_break_series")),
        ("Global Holm rejections", mtest.get("num_reject_holm")),
        ("Global BH rejections", mtest.get("num_reject_bh")),
    ]:
        tex_lines.append(f"\\item {k}: {v}")
    tex_lines.extend(["\\end{itemize}", "\\subsection{Claims}", "\\begin{verbatim}"])
    tex_lines.extend(_top_claim_lines(claims))
    tex_lines.extend(["\\end{verbatim}"])
    tex.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    return {
        "results_chapter_md": str(md),
        "results_chapter_tex": str(tex),
    }
