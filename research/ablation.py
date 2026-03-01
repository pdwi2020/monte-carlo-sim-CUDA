"""Ablation study engine for doctoral-grade pipeline components."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class AblationScenario:
    """One ablation scenario result."""

    scenario_id: str
    metric: str
    objective: str
    baseline_mean: float
    ablated_mean: float
    effect_mean: float
    effect_ci_low: float
    effect_ci_high: float
    p_value: float
    n: int
    interpretation: str


@dataclass
class AblationStudyResult:
    """Aggregate ablation output."""

    scenarios: List[AblationScenario]
    num_scenarios: int
    top_positive_impact_scenario: Optional[str]
    mean_effect_across_scenarios: float


def _bootstrap_effect(
    effect_samples: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    if effect_samples.size < 2:
        val = float(effect_samples.mean()) if effect_samples.size == 1 else 0.0
        return val, val, val

    rng = np.random.default_rng(seed)
    n = int(effect_samples.size)
    samples = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        samples[i] = float(np.mean(effect_samples[idx]))
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return float(np.mean(effect_samples)), float(lo), float(hi)


def _p_value_two_sided(effect_samples: np.ndarray) -> float:
    if effect_samples.size < 2:
        return 1.0
    signs = np.sign(effect_samples)
    pos = float(np.mean(signs > 0))
    neg = float(np.mean(signs < 0))
    return float(np.clip(2.0 * min(pos, neg), 0.0, 1.0))


def _build_scenario(
    *,
    scenario_id: str,
    metric: str,
    objective: str,
    baseline: Sequence[float],
    ablated: Sequence[float],
    n_bootstrap: int,
    seed: int,
) -> AblationScenario:
    b = np.asarray(list(baseline), dtype=float).ravel()
    a = np.asarray(list(ablated), dtype=float).ravel()
    n = int(min(b.size, a.size))
    if n == 0:
        raise ValueError(f"Scenario '{scenario_id}' has no aligned observations")
    b = b[:n]
    a = a[:n]

    if objective == "lower_better":
        effect = a - b
        interpretation = "positive means ablation worsens (component is useful)"
    elif objective == "higher_better":
        effect = b - a
        interpretation = "positive means ablation worsens (component is useful)"
    else:
        raise ValueError("objective must be 'lower_better' or 'higher_better'")

    eff_mean, ci_low, ci_high = _bootstrap_effect(effect, n_bootstrap=n_bootstrap, seed=seed)
    return AblationScenario(
        scenario_id=scenario_id,
        metric=metric,
        objective=objective,
        baseline_mean=float(np.mean(b)),
        ablated_mean=float(np.mean(a)),
        effect_mean=eff_mean,
        effect_ci_low=ci_low,
        effect_ci_high=ci_high,
        p_value=_p_value_two_sided(effect),
        n=n,
        interpretation=interpretation,
    )


def _safe_series(x: Any) -> np.ndarray:
    arr = np.asarray(x if x is not None else [], dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    return arr


def run_ablation_study_from_results(
    results: Dict[str, Any],
    *,
    n_bootstrap: int = 700,
    seed: int = 42,
) -> AblationStudyResult:
    """Build standardized ablation scenarios from pipeline outputs."""

    scenarios: List[AblationScenario] = []

    forecast = results.get("forecasting_oos", {}) or {}
    losses = forecast.get("losses_by_model") or {}
    h = _safe_series(losses.get("heston"))
    r = _safe_series(losses.get("rough_heston"))
    n = _safe_series(losses.get("naive_last_surface"))
    m = int(min(h.size, r.size, n.size)) if h.size and r.size and n.size else 0
    if m >= 2:
        h = h[:m]
        r = r[:m]
        n = n[:m]
        full_best = np.minimum(np.minimum(h, r), n)
        no_rough = np.minimum(h, n)
        no_naive = np.minimum(h, r)
        scenarios.append(
            _build_scenario(
                scenario_id="remove_rough_model_class",
                metric="oos_forecast_rmse",
                objective="lower_better",
                baseline=full_best,
                ablated=no_rough,
                n_bootstrap=n_bootstrap,
                seed=seed + 1,
            )
        )
        scenarios.append(
            _build_scenario(
                scenario_id="remove_naive_surface_benchmark",
                metric="oos_forecast_rmse",
                objective="lower_better",
                baseline=full_best,
                ablated=no_naive,
                n_bootstrap=n_bootstrap,
                seed=seed + 2,
            )
        )

    state_space = results.get("state_space_filter", {}) or {}
    est = list(state_space.get("estimates") or [])
    filt_rmse = _safe_series([row.get("panel_rmse") for row in est if isinstance(row, dict)])
    raw_rmse = _safe_series([row.get("raw_panel_rmse") for row in est if isinstance(row, dict)])
    if filt_rmse.size >= 2 and raw_rmse.size >= 2:
        k = int(min(filt_rmse.size, raw_rmse.size))
        scenarios.append(
            _build_scenario(
                scenario_id="remove_state_space_filter",
                metric="panel_rmse",
                objective="lower_better",
                baseline=filt_rmse[:k],
                ablated=raw_rmse[:k],
                n_bootstrap=n_bootstrap,
                seed=seed + 3,
            )
        )

    micro = results.get("microstructure_hedging", {}) or {}
    micro_base = ((micro.get("frictionless") or {}).get("cvar95_loss"))
    micro_stress = ((micro.get("microstructure_stressed") or {}).get("cvar95_loss"))
    if micro_base is not None and micro_stress is not None:
        scenarios.append(
            _build_scenario(
                scenario_id="remove_microstructure_stress_awareness",
                metric="hedging_cvar95_loss",
                objective="lower_better",
                baseline=[float(micro_base)],
                ablated=[float(micro_stress)],
                n_bootstrap=n_bootstrap,
                seed=seed + 4,
            )
        )

    exec_aw = results.get("execution_aware_hedging", {}) or {}
    exec_base = ((exec_aw.get("frictionless") or {}).get("misspecified_heston") or {}).get("cvar95_loss")
    exec_stress = ((exec_aw.get("execution_stressed") or {}).get("misspecified_heston") or {}).get("cvar95_loss")
    if exec_base is not None and exec_stress is not None:
        scenarios.append(
            _build_scenario(
                scenario_id="remove_execution_aware_hedging_layer",
                metric="misspecified_delta_hedge_cvar95_loss",
                objective="lower_better",
                baseline=[float(exec_base)],
                ablated=[float(exec_stress)],
                n_bootstrap=n_bootstrap,
                seed=seed + 5,
            )
        )

    if scenarios:
        top = max(scenarios, key=lambda s: s.effect_mean)
        top_id = top.scenario_id
        mean_eff = float(np.mean([s.effect_mean for s in scenarios]))
    else:
        top_id = None
        mean_eff = 0.0

    return AblationStudyResult(
        scenarios=scenarios,
        num_scenarios=len(scenarios),
        top_positive_impact_scenario=top_id,
        mean_effect_across_scenarios=mean_eff,
    )


def ablation_to_dict(result: AblationStudyResult) -> Dict[str, object]:
    """Serialize ablation study payload."""

    return {
        "scenarios": [asdict(s) for s in result.scenarios],
        "num_scenarios": result.num_scenarios,
        "top_positive_impact_scenario": result.top_positive_impact_scenario,
        "mean_effect_across_scenarios": result.mean_effect_across_scenarios,
    }
