"""Publication-style table/figure generation for research artifacts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


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

    def _as_float(v: Any, default: float = float("nan")) -> float:
        try:
            return float(v)
        except Exception:
            return default

    def _save(name: str, *, key: str) -> None:
        p = fig_dir / name
        plt.tight_layout()
        plt.savefig(p, dpi=220)
        plt.close()
        paths[key] = str(p)

    frontier = results.get("hedging_robustness", {}).get("transaction_cost_frontier_heston", [])
    if frontier:
        x = [_as_float(row.get("transaction_cost")) for row in frontier]
        y_delta = [_as_float(row.get("delta_only_cvar95_loss")) for row in frontier]
        y_dv = [_as_float(row.get("delta_vega_cvar95_loss")) for row in frontier]
        plt.figure(figsize=(6.5, 4.0))
        plt.plot(x, y_delta, marker="o", label="Delta only")
        plt.plot(x, y_dv, marker="o", label="Delta-vega")
        plt.xlabel("Transaction Cost")
        plt.ylabel("CVaR95 Loss")
        plt.title("Hedging Risk Frontier")
        plt.grid(alpha=0.25)
        plt.legend()
        _save("hedging_frontier.png", key="hedging_frontier_png")

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
        _save("benchmark_runtime.png", key="benchmark_runtime_png")

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
            _save("historical_rmse.png", key="historical_rmse_png")

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
            _save("forecast_rmse_by_model.png", key="forecast_rmse_png")

        mean_rmse = forecast.get("mean_rmse_by_model") or {}
        std_rmse = forecast.get("std_rmse_by_model") or {}
        if mean_rmse:
            rows = sorted(mean_rmse.items(), key=lambda kv: kv[1] if kv[1] is not None else float("inf"))
            mnames = [k for k, _ in rows]
            mvals = [_as_float(v) for _, v in rows]
            merr = [_as_float(std_rmse.get(k), default=0.0) for k in mnames]
            plt.figure(figsize=(8.0, 4.2))
            plt.bar(mnames, mvals, yerr=merr, capsize=4.0, alpha=0.85)
            plt.ylabel("RMSE (mean +/- std)")
            plt.title("Forecast Model Leaderboard")
            plt.xticks(rotation=25, ha="right")
            plt.grid(axis="y", alpha=0.25)
            _save("forecast_leaderboard.png", key="forecast_leaderboard_png")

        if target_dates and losses_by_model:
            models = sorted(losses_by_model.keys())
            n_dates = len(target_dates)
            mat = np.full((len(models), n_dates), np.nan, dtype=float)
            for i, model in enumerate(models):
                vals = list(losses_by_model.get(model) or [])
                for j in range(min(n_dates, len(vals))):
                    mat[i, j] = _as_float(vals[j])
            if np.isfinite(mat).any():
                plt.figure(figsize=(9.5, 4.2))
                im = plt.imshow(mat, aspect="auto", interpolation="nearest", cmap="viridis")
                plt.colorbar(im, label="RMSE")
                plt.yticks(range(len(models)), models)
                tick_idx = list(range(0, n_dates, max(1, n_dates // 8)))
                plt.xticks(tick_idx, [target_dates[i] for i in tick_idx], rotation=30, ha="right")
                plt.title("Forecast Loss Heatmap (Model x Date)")
                plt.xlabel("Target Date")
                plt.ylabel("Model")
                _save("forecast_heatmap.png", key="forecast_heatmap_png")

            box_vals = [np.asarray(list(losses_by_model.get(m) or []), dtype=float) for m in models]
            if any(v.size > 0 for v in box_vals):
                clean = [v[np.isfinite(v)] for v in box_vals]
                plt.figure(figsize=(8.5, 4.2))
                plt.boxplot(clean, labels=models)
                plt.ylabel("RMSE")
                plt.title("Forecast Error Distribution by Model")
                plt.xticks(rotation=25, ha="right")
                plt.grid(axis="y", alpha=0.25)
                _save("forecast_boxplot.png", key="forecast_boxplot_png")

    challengers = results.get("challenger_baselines", {})
    if challengers:
        losses = challengers.get("losses_by_model") or {}
        if losses:
            models = sorted(losses.keys())
            series = [np.asarray(list(losses.get(m) or []), dtype=float) for m in models]
            if any(v.size > 0 for v in series):
                clean = [v[np.isfinite(v)] for v in series]
                plt.figure(figsize=(7.5, 4.0))
                plt.boxplot(clean, labels=models)
                plt.ylabel("RMSE")
                plt.title("Challenger Baseline Error Distribution")
                plt.xticks(rotation=20, ha="right")
                plt.grid(axis="y", alpha=0.25)
                _save("challenger_boxplot.png", key="challenger_boxplot_png")

    crisis_rows = results.get("crisis_subperiod_study", {}).get("episode_performance", [])
    if crisis_rows:
        episodes = sorted({str(r.get("episode_id")) for r in crisis_rows})
        models = sorted({str(r.get("model")) for r in crisis_rows})
        mat = np.full((len(episodes), len(models)), np.nan, dtype=float)
        for r in crisis_rows:
            e = str(r.get("episode_id"))
            m = str(r.get("model"))
            mat[episodes.index(e), models.index(m)] = _as_float(r.get("mean_rmse"))
        if np.isfinite(mat).any():
            plt.figure(figsize=(9.0, 4.0))
            im = plt.imshow(mat, aspect="auto", interpolation="nearest", cmap="magma_r")
            plt.colorbar(im, label="Mean RMSE")
            plt.yticks(range(len(episodes)), episodes)
            plt.xticks(range(len(models)), models, rotation=25, ha="right")
            plt.title("Crisis/Subperiod Model Performance Heatmap")
            plt.xlabel("Model")
            plt.ylabel("Episode")
            _save("crisis_episode_heatmap.png", key="crisis_episode_heatmap_png")

    ablation_rows = results.get("ablation_study", {}).get("scenarios", [])
    if ablation_rows:
        ranked = [r for r in ablation_rows if r.get("effect_mean") is not None]
        ranked.sort(key=lambda r: abs(_as_float(r.get("effect_mean"), default=0.0)), reverse=True)
        if ranked:
            labels = [str(r.get("scenario_id")) for r in ranked]
            eff = np.asarray([_as_float(r.get("effect_mean")) for r in ranked], dtype=float)
            lo = np.asarray([_as_float(r.get("effect_ci_low")) for r in ranked], dtype=float)
            hi = np.asarray([_as_float(r.get("effect_ci_high")) for r in ranked], dtype=float)
            plt.figure(figsize=(10.0, 4.8))
            y = np.arange(len(labels))
            plt.barh(y, eff, alpha=0.85)
            finite_ci = np.isfinite(lo) & np.isfinite(hi)
            if finite_ci.any():
                plt.errorbar(
                    eff[finite_ci],
                    y[finite_ci],
                    xerr=[eff[finite_ci] - lo[finite_ci], hi[finite_ci] - eff[finite_ci]],
                    fmt="none",
                    ecolor="black",
                    capsize=3,
                    lw=1,
                )
            plt.axvline(0.0, color="black", lw=1, alpha=0.7)
            plt.yticks(y, labels)
            plt.xlabel("Ablation Effect (positive = ablation worsens)")
            plt.title("Ablation Tornado Plot with CI")
            _save("ablation_tornado.png", key="ablation_tornado_png")

    break_rows = results.get("structural_break_diagnostics", {}).get("entries", [])
    if break_rows:
        labels = [str(r.get("series_id")) for r in break_rows]
        shifts = np.asarray([_as_float(r.get("mean_shift")) for r in break_rows], dtype=float)
        pvals = np.asarray([_as_float(r.get("p_value"), default=1.0) for r in break_rows], dtype=float)
        colors = np.clip(1.0 - pvals, 0.0, 1.0)
        plt.figure(figsize=(10.0, 4.2))
        plt.bar(range(len(labels)), shifts, color=plt.cm.viridis(colors))
        plt.axhline(0.0, color="black", lw=1, alpha=0.6)
        plt.xticks(range(len(labels)), labels, rotation=25, ha="right")
        plt.ylabel("Mean Shift at Break")
        plt.title("Structural-Break Mean Shift by Series (color: stronger significance)")
        plt.grid(axis="y", alpha=0.25)
        _save("structural_break_shift.png", key="structural_break_shift_png")

    mtest_rows = results.get("global_multiple_testing", {}).get("records", [])
    alpha = _as_float(results.get("global_multiple_testing", {}).get("alpha"), default=0.05)
    if mtest_rows:
        test_ids = [str(r.get("test_id")) for r in mtest_rows]
        raw = np.asarray([max(_as_float(r.get("raw_p_value"), default=1.0), 1e-12) for r in mtest_rows], dtype=float)
        holm = np.asarray(
            [max(_as_float(r.get("holm_adjusted_p_value"), default=1.0), 1e-12) for r in mtest_rows],
            dtype=float,
        )
        bh = np.asarray([max(_as_float(r.get("bh_adjusted_p_value"), default=1.0), 1e-12) for r in mtest_rows], dtype=float)
        x = np.arange(len(test_ids))
        plt.figure(figsize=(10.5, 4.4))
        plt.plot(x, -np.log10(raw), "o-", label="-log10(raw p)")
        plt.plot(x, -np.log10(holm), "s-", label="-log10(Holm adj p)")
        plt.plot(x, -np.log10(bh), "^-", label="-log10(BH adj p)")
        plt.axhline(-np.log10(max(alpha, 1e-12)), color="red", linestyle="--", label=f"alpha={alpha:g}")
        plt.xticks(x, test_ids, rotation=25, ha="right")
        plt.ylabel("Significance Scale")
        plt.title("Global Multiple-Testing P-value Ladder")
        plt.grid(alpha=0.25)
        plt.legend()
        _save("global_mtesting_pvalues.png", key="global_mtesting_pvalues_png")

    regime = results.get("regime_diagnostics", {})
    transitions = regime.get("transition_probabilities") or {}
    if transitions:
        states = sorted(set(transitions.keys()) | {d for row in transitions.values() for d in (row or {}).keys()})
        mat = np.zeros((len(states), len(states)), dtype=float)
        for i, src in enumerate(states):
            for j, dst in enumerate(states):
                mat[i, j] = _as_float((transitions.get(src) or {}).get(dst), default=0.0)
        plt.figure(figsize=(7.0, 5.6))
        im = plt.imshow(mat, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, label="Transition Probability")
        plt.xticks(range(len(states)), states, rotation=25, ha="right")
        plt.yticks(range(len(states)), states)
        plt.title("Regime Transition Matrix")
        plt.xlabel("To Regime")
        plt.ylabel("From Regime")
        _save("regime_transition_heatmap.png", key="regime_transition_heatmap_png")

    regime_perf = regime.get("model_performance", [])
    if regime_perf:
        regimes = sorted({str(r.get("regime")) for r in regime_perf})
        models = sorted({str(r.get("model")) for r in regime_perf})
        if regimes and models:
            width = 0.8 / max(1, len(models))
            x = np.arange(len(regimes), dtype=float)
            plt.figure(figsize=(9.5, 4.4))
            for idx, model in enumerate(models):
                vals = []
                for reg in regimes:
                    match = next(
                        (
                            r
                            for r in regime_perf
                            if str(r.get("regime")) == reg and str(r.get("model")) == model
                        ),
                        {},
                    )
                    vals.append(_as_float(match.get("mean_rmse"), default=float("nan")))
                plt.bar(x + idx * width - (len(models) - 1) * width / 2.0, vals, width=width, label=model)
            plt.xticks(x, regimes, rotation=20, ha="right")
            plt.ylabel("Mean RMSE")
            plt.title("Model Performance by Regime")
            plt.grid(axis="y", alpha=0.25)
            plt.legend()
            _save("regime_model_performance.png", key="regime_model_performance_png")

    regime_feats = regime.get("features", [])
    if regime_feats:
        regs = sorted({str(r.get("regime")) for r in regime_feats})
        cmap = plt.cm.get_cmap("tab10", max(1, len(regs)))
        plt.figure(figsize=(7.8, 5.0))
        for idx, reg in enumerate(regs):
            chunk = [r for r in regime_feats if str(r.get("regime")) == reg]
            x = [_as_float(r.get("atm_iv")) for r in chunk]
            y = [_as_float(r.get("skew_25_75")) for r in chunk]
            plt.scatter(x, y, label=reg, alpha=0.85, s=34, color=cmap(idx))
        plt.xlabel("ATM IV")
        plt.ylabel("Skew (25-75)")
        plt.title("Regime Feature Map")
        plt.grid(alpha=0.25)
        plt.legend()
        _save("regime_feature_scatter.png", key="regime_feature_scatter_png")

    state = results.get("state_space_filter", {})
    est = list(state.get("estimates", []))
    if est:
        dates = [str(r.get("quote_date")) for r in est]
        panel = np.asarray([_as_float(r.get("panel_rmse")) for r in est], dtype=float)
        raw = np.asarray([_as_float(r.get("raw_panel_rmse")) for r in est], dtype=float)
        inn = np.asarray([_as_float(r.get("innovation_l2")) for r in est], dtype=float)
        fig, ax1 = plt.subplots(figsize=(9.0, 4.2))
        ax1.plot(dates, panel, marker="o", label="Filtered panel RMSE")
        ax1.plot(dates, raw, marker="o", label="Raw panel RMSE")
        ax1.set_ylabel("RMSE")
        ax1.grid(alpha=0.25)
        ax2 = ax1.twinx()
        ax2.plot(dates, inn, linestyle="--", marker="x", color="tab:red", label="Innovation L2")
        ax2.set_ylabel("Innovation L2")
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
        plt.xticks(rotation=30, ha="right")
        plt.title("State-Space Filter: Filtered vs Raw Calibration Error")
        _save("state_space_filter_vs_raw.png", key="state_space_filter_vs_raw_png")

    date_cal = list(hist.get("date_calibrations", [])) if hist else []
    if date_cal:
        param_keys = ["v0", "kappa", "theta", "xi", "rho", "hurst"]
        available = [k for k in param_keys if any(k in row for row in date_cal)]
        if available:
            dates = [str(r.get("quote_date")) for r in date_cal]
            n = len(available)
            cols = 2
            rows = int(np.ceil(n / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(10.0, 2.8 * rows), squeeze=False)
            for idx, k in enumerate(available):
                ax = axes[idx // cols][idx % cols]
                vals = [_as_float(r.get(k)) for r in date_cal]
                ax.plot(dates, vals, marker="o")
                ax.set_title(k)
                ax.grid(alpha=0.25)
                ax.tick_params(axis="x", rotation=30)
            for idx in range(n, rows * cols):
                axes[idx // cols][idx % cols].axis("off")
            fig.suptitle("Historical Calibration Parameter Drift", y=1.02)
            _save("historical_parameter_drift.png", key="historical_parameter_drift_png")

    hpc = results.get("hpc_scaling", {})
    hrows = list(hpc.get("rows", []))
    if hrows:
        backends = sorted({str(r.get("backend")) for r in hrows})
        plt.figure(figsize=(8.5, 4.2))
        for b in backends:
            chunk = sorted(
                [r for r in hrows if str(r.get("backend")) == b],
                key=lambda r: _as_float(r.get("problem_size")),
            )
            x = [_as_float(r.get("problem_size")) for r in chunk]
            y = [_as_float(r.get("runtime_seconds")) for r in chunk]
            plt.plot(x, y, marker="o", label=b)
        plt.xlabel("Problem Size")
        plt.ylabel("Runtime (s)")
        plt.title("HPC Runtime Scaling by Backend")
        plt.grid(alpha=0.25)
        plt.legend()
        _save("hpc_runtime_scaling.png", key="hpc_runtime_scaling_png")

        cpu_backend = str(hpc.get("cpu_backend", ""))
        base = {
            _as_float(r.get("problem_size")): _as_float(r.get("runtime_seconds"))
            for r in hrows
            if str(r.get("backend")) == cpu_backend
        }
        if base:
            plt.figure(figsize=(8.5, 4.2))
            for b in backends:
                if b == cpu_backend:
                    continue
                chunk = sorted(
                    [r for r in hrows if str(r.get("backend")) == b],
                    key=lambda r: _as_float(r.get("problem_size")),
                )
                x = []
                y = []
                for r in chunk:
                    ps = _as_float(r.get("problem_size"))
                    rt = _as_float(r.get("runtime_seconds"))
                    if ps in base and np.isfinite(base[ps]) and base[ps] > 0 and np.isfinite(rt) and rt > 0:
                        x.append(ps)
                        y.append(base[ps] / rt)
                if x:
                    plt.plot(x, y, marker="o", label=f"{b} speedup vs {cpu_backend}")
            if plt.gca().has_data():
                plt.xlabel("Problem Size")
                plt.ylabel("Speedup (x)")
                plt.title("HPC Speedup Scaling")
                plt.grid(alpha=0.25)
                plt.legend()
                _save("hpc_speedup_scaling.png", key="hpc_speedup_scaling_png")
            else:
                plt.close()

    cuda = results.get("cuda_tuning", {})
    cand = list(cuda.get("candidates", []))
    if cand:
        tpb = np.asarray([_as_float(r.get("threads_per_block")) for r in cand], dtype=float)
        nstreams = np.asarray([_as_float(r.get("num_streams")) for r in cand], dtype=float)
        runtime = np.asarray([_as_float(r.get("runtime_seconds")) for r in cand], dtype=float)
        if np.isfinite(runtime).any():
            plt.figure(figsize=(7.0, 5.0))
            sc = plt.scatter(tpb, nstreams, c=runtime, cmap="plasma", s=70, alpha=0.9)
            plt.colorbar(sc, label="Runtime (s)")
            plt.xlabel("Threads per Block")
            plt.ylabel("Num Streams")
            plt.title("CUDA Tuning Runtime Map")
            plt.grid(alpha=0.25)
            _save("cuda_tuning_runtime_map.png", key="cuda_tuning_runtime_map_png")

            baseline = _as_float(cuda.get("baseline_runtime_seconds"), default=float("nan"))
            if np.isfinite(baseline) and baseline > 0:
                speedup = baseline / runtime
                plt.figure(figsize=(8.0, 4.0))
                labels = [f"tpb={int(t)}|s={int(s)}" for t, s in zip(tpb, nstreams)]
                order = np.argsort(-speedup)
                plt.bar(np.arange(len(order)), speedup[order], alpha=0.85)
                plt.xticks(np.arange(len(order)), [labels[i] for i in order], rotation=30, ha="right")
                plt.ylabel("Speedup vs Baseline")
                plt.title("CUDA Candidate Speedups")
                plt.grid(axis="y", alpha=0.25)
                _save("cuda_tuning_speedup.png", key="cuda_tuning_speedup_png")

    port = results.get("portfolio_overlay", {})
    corr = port.get("correlation_matrix")
    assets = [str(r.get("symbol")) for r in (port.get("asset_risks") or [])]
    if isinstance(corr, list) and corr:
        mat = np.asarray(corr, dtype=float)
        if mat.ndim == 2:
            n = mat.shape[0]
            labels = assets if len(assets) == n else [f"A{i+1}" for i in range(n)]
            plt.figure(figsize=(6.4, 5.4))
            im = plt.imshow(mat, vmin=-1.0, vmax=1.0, cmap="coolwarm")
            plt.colorbar(im, label="Correlation")
            plt.xticks(range(n), labels, rotation=20, ha="right")
            plt.yticks(range(n), labels)
            plt.title("Portfolio Asset Correlation Matrix")
            _save("portfolio_corr_heatmap.png", key="portfolio_corr_heatmap_png")

    es_contrib = port.get("expected_shortfall_contributions") or {}
    if es_contrib:
        labels = list(es_contrib.keys())
        vals = [_as_float(es_contrib.get(k)) for k in labels]
        plt.figure(figsize=(7.4, 4.0))
        plt.bar(labels, vals, alpha=0.85)
        plt.axhline(0.0, color="black", lw=1, alpha=0.6)
        plt.ylabel("Expected Shortfall Contribution")
        plt.title("Portfolio Tail-Risk Contribution by Asset")
        plt.grid(axis="y", alpha=0.25)
        _save("portfolio_es_contrib.png", key="portfolio_es_contrib_png")

    micro = results.get("microstructure_hedging", {})
    fr = micro.get("frictionless") or {}
    st = micro.get("microstructure_stressed") or {}
    if fr and st:
        labels = ["VaR95 Loss", "CVaR95 Loss"]
        vals_fr = [_as_float(fr.get("var95_loss")), _as_float(fr.get("cvar95_loss"))]
        vals_st = [_as_float(st.get("var95_loss")), _as_float(st.get("cvar95_loss"))]
        x = np.arange(len(labels))
        width = 0.35
        plt.figure(figsize=(7.0, 4.0))
        plt.bar(x - width / 2.0, vals_fr, width=width, label="Frictionless")
        plt.bar(x + width / 2.0, vals_st, width=width, label="Microstructure-stressed")
        plt.xticks(x, labels)
        plt.ylabel("Tail Loss")
        plt.title("Microstructure Stress: VaR/CVaR Shift")
        plt.grid(axis="y", alpha=0.25)
        plt.legend()
        _save("microstructure_var_cvar.png", key="microstructure_var_cvar_png")

    exec_payload = results.get("execution_aware_hedging", {})
    if exec_payload:
        plt.figure(figsize=(8.5, 4.2))
        plotted = False
        for regime_name in ["frictionless", "execution_stressed"]:
            regime_payload = exec_payload.get(regime_name) or {}
            tfront = regime_payload.get("transaction_cost_frontier_heston") or []
            if not tfront:
                continue
            tc = [_as_float(r.get("transaction_cost")) for r in tfront]
            d = [_as_float(r.get("delta_only_cvar95_loss")) for r in tfront]
            dv = [_as_float(r.get("delta_vega_cvar95_loss")) for r in tfront]
            plt.plot(tc, d, marker="o", label=f"{regime_name}: delta-only")
            plt.plot(tc, dv, marker="x", linestyle="--", label=f"{regime_name}: delta-vega")
            plotted = True
        if plotted:
            plt.xlabel("Transaction Cost")
            plt.ylabel("CVaR95 Loss")
            plt.title("Execution-Aware Transaction-Cost Frontier")
            plt.grid(alpha=0.25)
            plt.legend(fontsize=8)
            _save("execution_transaction_frontier.png", key="execution_transaction_frontier_png")
        else:
            plt.close()

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
