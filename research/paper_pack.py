"""Generate manuscript/appendix bundles from research pipeline outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def generate_paper_package(
    *,
    output_dir: str | Path,
    results: Dict[str, Any],
    claims: Any,
    artifact_paths: Dict[str, Any],
) -> Dict[str, str]:
    """Create paper-ready markdown/latex skeleton files."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manuscript_md = out / "manuscript.md"
    appendix_md = out / "appendix.md"
    manuscript_tex = out / "manuscript.tex"

    forecast = results.get("forecasting_oos", {})
    stat = results.get("statistical_validation", {})
    perf = results.get("performance", {}).get("summary", {})
    hpc = results.get("hpc_scaling", {})
    mkt = results.get("market_data", {})

    manuscript_md.write_text(
        "\n".join(
            [
                "# Doctoral Research Manuscript (Draft)",
                "",
                "## Abstract",
                "This study benchmarks advanced Monte Carlo option pricing and calibration pipelines under realistic research diagnostics.",
                "",
                "## Main Contributions",
                "- Rough-model calibration diagnostics with uncertainty quantification.",
                "- Leakage-free walk-forward forecasting and econometric model comparisons.",
                "- Execution and portfolio risk stress diagnostics with publication artifacts.",
                "",
                "## Key Results",
                f"- Forecast best model: `{forecast.get('best_model')}`",
                f"- Forecast mean RMSE map: `{forecast.get('mean_rmse_by_model')}`",
                f"- SPA p-value vs naive: `{(stat.get('spa_vs_naive_forecast') or {}).get('p_value')}`",
                f"- White Reality Check p-value vs naive: `{(stat.get('white_reality_check_vs_naive') or {}).get('p_value')}`",
                f"- MCS survivors: `{(stat.get('model_confidence_set') or {}).get('surviving_models')}`",
                f"- Peak backend speedup: `{perf.get('max_speedup')}`",
                f"- HPC max single-GPU speedup: `{hpc.get('max_single_gpu_speedup')}`",
                f"- Market-data source/status: `{mkt.get('source')}` / `{mkt.get('status')}`",
                "",
                "## Reproducibility",
                f"- Artifact root: `{artifact_paths.get('publication_assets')}`",
                f"- Publication tables: `{artifact_paths.get('publication_index')}`",
                "",
                "## Claims",
                f"- Number of claims evaluated: `{len(claims or [])}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    appendix_md.write_text(
        "\n".join(
            [
                "# Appendix Diagnostics",
                "",
                "## Econometric Validation",
                f"- DM efficiency: `{stat.get('diebold_mariano_efficiency')}`",
                f"- SPA vs naive: `{stat.get('spa_vs_naive_forecast')}`",
                f"- White RC vs naive: `{stat.get('white_reality_check_vs_naive')}`",
                f"- Model confidence set: `{stat.get('model_confidence_set')}`",
                "",
                "## Walk-Forward",
                f"- Leakage-free walk-forward summary: `{results.get('walkforward_leakage_free')}`",
                "",
                "## State-Space",
                f"- Sequential calibration summary: `{results.get('state_space_filter')}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manuscript_tex.write_text(
        "\n".join(
            [
                "\\documentclass[11pt]{article}",
                "\\usepackage[margin=1in]{geometry}",
                "\\begin{document}",
                "\\title{Doctoral Monte Carlo Research Report}",
                "\\author{Automated Pipeline}",
                "\\date{}",
                "\\maketitle",
                "\\section*{Abstract}",
                "Automated manuscript scaffold generated from reproducible pipeline artifacts.",
                "\\section*{Key Metrics}",
                "\\begin{itemize}",
                f"\\item Forecast best model: {forecast.get('best_model')}",
                f"\\item SPA p-value: {_fmt((stat.get('spa_vs_naive_forecast') or {}).get('p_value'))}",
                f"\\item White RC p-value: {_fmt((stat.get('white_reality_check_vs_naive') or {}).get('p_value'))}",
                "\\end{itemize}",
                "\\end{document}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "manuscript_md": str(manuscript_md),
        "appendix_md": str(appendix_md),
        "manuscript_tex": str(manuscript_tex),
    }
