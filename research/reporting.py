"""Publication-style table/figure generation for research artifacts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _write_latex_table(path: Path, rows: Iterable[Dict[str, Any]]) -> str:
    """Write a compact LaTeX tabular from row dicts."""

    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("% empty table\n", encoding="utf-8")
        return str(path)

    cols: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in cols:
                cols.append(k)

    def _fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v)

    lines = []
    lines.append("\\begin{tabular}{" + "l" * len(cols) + "}")
    lines.append("\\hline")
    lines.append(" & ".join(cols) + " \\\\")
    lines.append("\\hline")
    for row in rows:
        lines.append(" & ".join(_fmt(row.get(c, "")) for c in cols) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> str:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["empty"])
        return str(path)

    fieldnames: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return str(path)


def _claims_table(claims: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in claims or []:
        claim = item.get("claim", {})
        ev = item.get("evaluation", {})
        out.append(
            {
                "claim_id": claim.get("claim_id"),
                "title": claim.get("title"),
                "metric": claim.get("metric"),
                "threshold": claim.get("minimum_effect"),
                "observed_effect": ev.get("observed_effect"),
                "passed": ev.get("passed"),
                "p_value": ev.get("p_value"),
                "ci_low": (ev.get("confidence_interval") or {}).get("low"),
                "ci_high": (ev.get("confidence_interval") or {}).get("high"),
            }
        )
    return out


def _summary_table(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    mlmc = results.get("mlmc", {})
    hmlmc = results.get("heston_mlmc", {})
    rough = results.get("rough_heston", {})
    rb = results.get("rough_bergomi", {})
    hist = results.get("historical_backtest", {})
    hedge = results.get("hedging_robustness", {})
    forecast = results.get("forecasting_oos", {})
    challengers = results.get("challenger_baselines", {})
    walk = results.get("walkforward_leakage_free", {})
    micro = results.get("microstructure_hedging", {})
    hpc = results.get("hpc_scaling", {})
    cuda = results.get("cuda_tuning", {})
    crisis = results.get("crisis_subperiod_study", {})
    ablation = results.get("ablation_study", {})
    breaks = results.get("structural_break_diagnostics", {})
    mtest = results.get("global_multiple_testing", {})
    regime = results.get("regime_diagnostics", {})

    rows.append({"metric": "mlmc_avg_speedup", "value": mlmc.get("avg_speedup")})
    rows.append({"metric": "heston_mlmc_avg_speedup", "value": hmlmc.get("avg_speedup")})
    rows.append({"metric": "rough_improvement_ratio", "value": rough.get("improvement_ratio")})
    rows.append({"metric": "rough_bergomi_improvement_ratio", "value": rb.get("improvement_ratio")})
    rows.append({"metric": "historical_validate_rmse", "value": hist.get("validate_mean_rmse")})
    rows.append({"metric": "historical_test_rmse", "value": hist.get("test_mean_rmse")})
    rows.append(
        {
            "metric": "hedge_misspecified_over_well_cvar_ratio",
            "value": hedge.get("cvar95_loss_ratio_misspecified_over_well_specified"),
        }
    )
    rows.append(
        {
            "metric": "hedge_delta_vega_over_delta_only_ratio",
            "value": hedge.get("cvar95_loss_ratio_delta_vega_over_delta_only_misspecified"),
        }
    )
    rows.append({"metric": "forecast_best_model", "value": forecast.get("best_model")})
    rows.append({"metric": "forecast_worst_model", "value": forecast.get("worst_model")})
    rows.append({"metric": "challenger_best_model", "value": challengers.get("best_model")})
    rows.append({"metric": "challenger_worst_model", "value": challengers.get("worst_model")})
    rows.append({"metric": "walkforward_best_model", "value": walk.get("best_model_by_aggregate_rmse")})
    rows.append({"metric": "microstructure_cvar95_degradation", "value": micro.get("cvar95_degradation_ratio")})
    rows.append({"metric": "hpc_max_single_gpu_speedup", "value": hpc.get("max_single_gpu_speedup")})
    rows.append({"metric": "cuda_tuning_status", "value": cuda.get("status")})
    rows.append({"metric": "cuda_tuning_best_speedup", "value": cuda.get("best_speedup_over_baseline")})
    rows.append({"metric": "crisis_stress_episode", "value": crisis.get("stress_episode_id")})
    rows.append({"metric": "crisis_stress_best_model", "value": crisis.get("stress_best_model")})
    rows.append({"metric": "ablation_top_scenario", "value": ablation.get("top_positive_impact_scenario")})
    rows.append({"metric": "ablation_mean_effect", "value": ablation.get("mean_effect_across_scenarios")})
    rows.append({"metric": "structural_break_strongest_series", "value": breaks.get("strongest_break_series")})
    rows.append({"metric": "structural_break_strongest_p_value", "value": breaks.get("strongest_break_p_value")})
    rows.append({"metric": "global_mtesting_num_tests", "value": mtest.get("num_tests")})
    rows.append({"metric": "global_mtesting_reject_holm", "value": mtest.get("num_reject_holm")})
    rows.append({"metric": "global_mtesting_reject_bh", "value": mtest.get("num_reject_bh")})
    rows.append({"metric": "regime_num_regimes", "value": regime.get("num_regimes")})
    rows.append({"metric": "regime_high_vol", "value": regime.get("high_vol_regime")})
    rows.append({"metric": "regime_high_vol_best_model", "value": regime.get("high_vol_best_model")})
    rows.append({"metric": "regime_persistence_ratio", "value": regime.get("persistence_ratio")})
    for model, value in (forecast.get("mean_rmse_by_model") or {}).items():
        rows.append({"metric": f"forecast_mean_rmse_{model}", "value": value})
    for model, value in (challengers.get("mean_rmse_by_model") or {}).items():
        rows.append({"metric": f"challenger_mean_rmse_{model}", "value": value})
    return rows


def _forecast_leaderboard_table(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    forecast = results.get("forecasting_oos", {})
    mean_rmse = forecast.get("mean_rmse_by_model") or {}
    std_rmse = forecast.get("std_rmse_by_model") or {}
    rows: List[Dict[str, Any]] = []
    for model, mean_val in mean_rmse.items():
        rows.append(
            {
                "model": model,
                "mean_rmse": mean_val,
                "std_rmse": std_rmse.get(model),
            }
        )
    rows.sort(key=lambda x: x["mean_rmse"] if x["mean_rmse"] is not None else float("inf"))
    return rows


def _forecast_loss_rows(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    forecast = results.get("forecasting_oos", {})
    target_dates = list(forecast.get("target_dates", []))
    losses_by_model = forecast.get("losses_by_model") or {}
    models = sorted(losses_by_model.keys())
    rows: List[Dict[str, Any]] = []
    n = len(target_dates)
    for i in range(n):
        row: Dict[str, Any] = {"target_date": target_dates[i]}
        for m in models:
            vals = losses_by_model.get(m) or []
            row[m] = vals[i] if i < len(vals) else None
        rows.append(row)
    return rows


def _challenger_leaderboard_rows(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    payload = results.get("challenger_baselines", {})
    mean_rmse = payload.get("mean_rmse_by_model") or {}
    std_rmse = payload.get("std_rmse_by_model") or {}
    rows: List[Dict[str, Any]] = []
    for model, mean_val in mean_rmse.items():
        rows.append({"model": model, "mean_rmse": mean_val, "std_rmse": std_rmse.get(model)})
    rows.sort(key=lambda x: x["mean_rmse"] if x["mean_rmse"] is not None else float("inf"))
    return rows


def _walkforward_rows(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    walk = results.get("walkforward_leakage_free", {})
    return list(walk.get("windows", []))


def _state_space_rows(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    state = results.get("state_space_filter", {})
    return list(state.get("estimates", []))


def _crisis_rows(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    crisis = results.get("crisis_subperiod_study", {})
    return list(crisis.get("episode_performance", []))


def _crisis_dm_rows(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    crisis = results.get("crisis_subperiod_study", {})
    return list(crisis.get("dm_tests", []))


def _ablation_rows(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    ab = results.get("ablation_study", {})
    return list(ab.get("scenarios", []))


def _cuda_tuning_rows(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    cuda = results.get("cuda_tuning", {})
    return list(cuda.get("candidates", []))


def _structural_break_rows(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    payload = results.get("structural_break_diagnostics", {})
    return list(payload.get("entries", []))


def _global_multiple_testing_rows(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    payload = results.get("global_multiple_testing", {})
    return list(payload.get("records", []))


def _regime_feature_rows(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    regime = results.get("regime_diagnostics", {})
    return list(regime.get("features", []))


def _regime_model_rows(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    regime = results.get("regime_diagnostics", {})
    return list(regime.get("model_performance", []))


def _regime_transition_rows(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    regime = results.get("regime_diagnostics", {})
    trans = regime.get("transition_probabilities") or {}
    rows: List[Dict[str, Any]] = []
    for src, row in sorted(trans.items()):
        for dst, prob in sorted((row or {}).items()):
            rows.append({"from_regime": src, "to_regime": dst, "probability": prob})
    return rows


def _econometric_table(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    sv = results.get("statistical_validation", {})

    dm = sv.get("diebold_mariano_efficiency") or {}
    if dm:
        out.append(
            {
                "test": "diebold_mariano_efficiency",
                "statistic": dm.get("statistic"),
                "p_value": dm.get("p_value"),
                "benchmark": "baseline_mc",
                "best_model": "vr_mc",
            }
        )

    spa = sv.get("spa_vs_naive_forecast") or {}
    if spa:
        out.append(
            {
                "test": "spa_vs_naive_forecast",
                "statistic": spa.get("statistic"),
                "p_value": spa.get("p_value"),
                "benchmark": spa.get("benchmark_model"),
                "best_model": spa.get("best_model"),
            }
        )

    rc = sv.get("white_reality_check_vs_naive") or {}
    if rc:
        out.append(
            {
                "test": "white_reality_check_vs_naive",
                "statistic": rc.get("statistic"),
                "p_value": rc.get("p_value"),
                "benchmark": rc.get("benchmark_model"),
                "best_model": rc.get("best_model"),
            }
        )

    mcs = sv.get("model_confidence_set") or {}
    if mcs:
        out.append(
            {
                "test": "model_confidence_set",
                "statistic": None,
                "p_value": None,
                "benchmark": None,
                "best_model": ",".join(mcs.get("surviving_models", [])),
            }
        )
    return out


def _maybe_generate_figures(results: Dict[str, Any], out_dir: Path) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return paths

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    frontier = results.get("hedging_robustness", {}).get("transaction_cost_frontier_heston", [])
    if frontier:
        x = [row["transaction_cost"] for row in frontier]
        y_delta = [row["delta_only_cvar95_loss"] for row in frontier]
        y_dv = [row["delta_vega_cvar95_loss"] for row in frontier]
        plt.figure(figsize=(6.5, 4.0))
        plt.plot(x, y_delta, marker="o", label="Delta only")
        plt.plot(x, y_dv, marker="o", label="Delta-vega")
        plt.xlabel("Transaction Cost")
        plt.ylabel("CVaR95 Loss")
        plt.title("Hedging Risk Frontier")
        plt.grid(alpha=0.25)
        plt.legend()
        p = fig_dir / "hedging_frontier.png"
        plt.tight_layout()
        plt.savefig(p, dpi=220)
        plt.close()
        paths["hedging_frontier_png"] = str(p)

    bsum = results.get("benchmark", {}).get("summary", {})
    labels = []
    vals = []
    if isinstance(bsum, list):
        for row in bsum:
            if not isinstance(row, dict):
                continue
            label = f"{row.get('case_id', 'case')}:{row.get('config_label', 'cfg')}"
            runtime = row.get("mean_runtime_seconds")
            if runtime is not None:
                labels.append(label)
                vals.append(runtime)
    elif isinstance(bsum, dict):
        for case in bsum.values():
            if not isinstance(case, dict):
                continue
            for config_label, item in case.items():
                if isinstance(item, dict) and "avg_runtime_seconds" in item:
                    labels.append(config_label)
                    vals.append(item["avg_runtime_seconds"])

    if labels and vals:
        top = min(10, len(labels))
        plt.figure(figsize=(8.0, 4.0))
        plt.bar(labels[:top], vals[:top])
        plt.ylabel("Avg Runtime (s)")
        plt.title("Benchmark Runtime by Configuration")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        p = fig_dir / "benchmark_runtime.png"
        plt.savefig(p, dpi=220)
        plt.close()
        paths["benchmark_runtime_png"] = str(p)

    hist = results.get("historical_backtest", {})
    if hist:
        pts = list(hist.get("validate_evaluations", [])) + list(hist.get("test_evaluations", []))
        if pts:
            x = [row["quote_date"] for row in pts]
            y = [row["rmse"] for row in pts]
            plt.figure(figsize=(8.0, 3.8))
            plt.plot(x, y, marker="o")
            plt.ylabel("RMSE")
            plt.title("Date-Sliced Out-of-Sample RMSE")
            plt.xticks(rotation=35, ha="right")
            plt.grid(alpha=0.25)
            plt.tight_layout()
            p = fig_dir / "historical_rmse.png"
            plt.savefig(p, dpi=220)
            plt.close()
            paths["historical_rmse_png"] = str(p)

    forecast = results.get("forecasting_oos", {})
    if forecast:
        target_dates = list(forecast.get("target_dates", []))
        losses_by_model = forecast.get("losses_by_model") or {}
        if target_dates and losses_by_model:
            plt.figure(figsize=(8.0, 3.8))
            for model, losses in sorted(losses_by_model.items()):
                if not losses:
                    continue
                n = min(len(target_dates), len(losses))
                plt.plot(target_dates[:n], losses[:n], marker="o", label=model)
            plt.ylabel("RMSE")
            plt.title("Rolling OOS Forecast Error by Model")
            plt.xticks(rotation=35, ha="right")
            plt.grid(alpha=0.25)
            plt.legend()
            plt.tight_layout()
            p = fig_dir / "forecast_rmse_by_model.png"
            plt.savefig(p, dpi=220)
            plt.close()
            paths["forecast_rmse_png"] = str(p)

    return paths


def generate_publication_assets(
    *,
    output_dir: Path,
    results: Dict[str, Any],
    claims: Any,
) -> Dict[str, str]:
    """Generate paper-ready tables/figures from pipeline outputs."""

    out_dir = Path(output_dir)
    table_dir = out_dir / "tables"

    paths: Dict[str, str] = {}
    claims_table = _claims_table(claims)
    metrics_table = _summary_table(results)
    paths["claims_table_csv"] = _write_csv(table_dir / "claims_summary.csv", claims_table)
    paths["metrics_table_csv"] = _write_csv(table_dir / "metrics_summary.csv", metrics_table)
    paths["claims_table_tex"] = _write_latex_table(table_dir / "claims_summary.tex", claims_table)
    paths["metrics_table_tex"] = _write_latex_table(table_dir / "metrics_summary.tex", metrics_table)

    hist = results.get("historical_backtest", {})
    if hist:
        paths["historical_eval_csv"] = _write_csv(
            table_dir / "historical_evaluations.csv",
            list(hist.get("validate_evaluations", [])) + list(hist.get("test_evaluations", [])),
        )
        paths["historical_drift_csv"] = _write_csv(
            table_dir / "historical_calibration_drift.csv",
            list(hist.get("date_calibrations", [])),
        )

    frontier = results.get("hedging_robustness", {}).get("transaction_cost_frontier_heston", [])
    if frontier:
        paths["hedging_frontier_csv"] = _write_csv(table_dir / "hedging_frontier.csv", frontier)

    stability = results.get("hedging_robustness", {}).get("rebalance_stability_heston", [])
    if stability:
        paths["hedging_stability_csv"] = _write_csv(table_dir / "hedging_stability.csv", stability)

    forecast_rows = _forecast_leaderboard_table(results)
    if forecast_rows:
        paths["forecast_leaderboard_csv"] = _write_csv(table_dir / "forecast_leaderboard.csv", forecast_rows)
        paths["forecast_leaderboard_tex"] = _write_latex_table(table_dir / "forecast_leaderboard.tex", forecast_rows)

    forecast_losses = _forecast_loss_rows(results)
    if forecast_losses:
        paths["forecast_losses_csv"] = _write_csv(table_dir / "forecast_losses.csv", forecast_losses)

    challenger_rows = _challenger_leaderboard_rows(results)
    if challenger_rows:
        paths["challenger_leaderboard_csv"] = _write_csv(table_dir / "challenger_leaderboard.csv", challenger_rows)
        paths["challenger_leaderboard_tex"] = _write_latex_table(
            table_dir / "challenger_leaderboard.tex", challenger_rows
        )

    walk_rows = _walkforward_rows(results)
    if walk_rows:
        paths["walkforward_windows_csv"] = _write_csv(table_dir / "walkforward_windows.csv", walk_rows)

    state_rows = _state_space_rows(results)
    if state_rows:
        paths["state_space_estimates_csv"] = _write_csv(table_dir / "state_space_estimates.csv", state_rows)

    crisis_rows = _crisis_rows(results)
    if crisis_rows:
        paths["crisis_episode_performance_csv"] = _write_csv(
            table_dir / "crisis_episode_performance.csv", crisis_rows
        )
        paths["crisis_episode_performance_tex"] = _write_latex_table(
            table_dir / "crisis_episode_performance.tex", crisis_rows
        )

    crisis_dm_rows = _crisis_dm_rows(results)
    if crisis_dm_rows:
        paths["crisis_dm_tests_csv"] = _write_csv(table_dir / "crisis_dm_tests.csv", crisis_dm_rows)

    ablation_rows = _ablation_rows(results)
    if ablation_rows:
        paths["ablation_study_csv"] = _write_csv(table_dir / "ablation_study.csv", ablation_rows)
        paths["ablation_study_tex"] = _write_latex_table(table_dir / "ablation_study.tex", ablation_rows)

    cuda_rows = _cuda_tuning_rows(results)
    if cuda_rows:
        paths["cuda_tuning_candidates_csv"] = _write_csv(table_dir / "cuda_tuning_candidates.csv", cuda_rows)

    structural_break_rows = _structural_break_rows(results)
    if structural_break_rows:
        paths["structural_breaks_csv"] = _write_csv(table_dir / "structural_breaks.csv", structural_break_rows)
        paths["structural_breaks_tex"] = _write_latex_table(table_dir / "structural_breaks.tex", structural_break_rows)

    mtest_rows = _global_multiple_testing_rows(results)
    if mtest_rows:
        paths["global_multiple_testing_csv"] = _write_csv(table_dir / "global_multiple_testing.csv", mtest_rows)
        paths["global_multiple_testing_tex"] = _write_latex_table(
            table_dir / "global_multiple_testing.tex", mtest_rows
        )

    regime_features = _regime_feature_rows(results)
    if regime_features:
        paths["regime_features_csv"] = _write_csv(table_dir / "regime_features.csv", regime_features)
        paths["regime_features_tex"] = _write_latex_table(table_dir / "regime_features.tex", regime_features)

    regime_model = _regime_model_rows(results)
    if regime_model:
        paths["regime_model_performance_csv"] = _write_csv(
            table_dir / "regime_model_performance.csv", regime_model
        )
        paths["regime_model_performance_tex"] = _write_latex_table(
            table_dir / "regime_model_performance.tex", regime_model
        )

    regime_transitions = _regime_transition_rows(results)
    if regime_transitions:
        paths["regime_transitions_csv"] = _write_csv(table_dir / "regime_transitions.csv", regime_transitions)

    econ_rows = _econometric_table(results)
    if econ_rows:
        paths["econometrics_table_csv"] = _write_csv(table_dir / "econometrics_summary.csv", econ_rows)
        paths["econometrics_table_tex"] = _write_latex_table(table_dir / "econometrics_summary.tex", econ_rows)

    paths.update(_maybe_generate_figures(results, out_dir))
    return paths
