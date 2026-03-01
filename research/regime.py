"""Regime-aware diagnostics for option-surface forecasting studies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from calibration import MarketOption
from .historical_backtest import DatedOptionPanel


@dataclass
class RegimeFeature:
    """Per-date surface summary features used for regime tagging."""

    quote_date: str
    atm_iv: float
    skew_25_75: float
    term_slope: float
    regime: str


@dataclass
class RegimeModelPerformance:
    """Model error summary within one regime."""

    regime: str
    model: str
    mean_rmse: float
    std_rmse: float
    count: int


@dataclass
class RegimeDiagnosticsResult:
    """Aggregate regime diagnostics payload."""

    num_dates: int
    num_regimes: int
    regime_by_date: Dict[str, str]
    features: List[RegimeFeature]
    transition_counts: Dict[str, Dict[str, int]]
    transition_probabilities: Dict[str, Dict[str, float]]
    persistence_ratio: float
    model_performance: List[RegimeModelPerformance]
    best_model_by_regime: Dict[str, str]
    high_vol_regime: str
    high_vol_best_model: Optional[str]


def _safe_iv(opt: MarketOption, *, spot: float) -> Optional[float]:
    if opt.market_iv is not None:
        return float(opt.market_iv)
    price = opt.market_price if opt.market_price is not None else opt.mid_price
    if price is None or opt.maturity <= 0:
        return None
    # Brenner-Subrahmanyam style approximation as an emergency fallback.
    approx = np.sqrt(2.0 * np.pi / max(opt.maturity, 1e-8)) * float(price) / max(spot, 1e-12)
    return float(np.clip(approx, 0.01, 3.0))


def _atm_iv_for_maturity(options: Sequence[MarketOption], *, spot: float, maturity: float) -> Optional[float]:
    bucket = [o for o in options if abs(float(o.maturity) - float(maturity)) < 1e-10]
    if not bucket:
        return None
    atm = min(bucket, key=lambda o: abs(float(o.strike) - spot))
    return _safe_iv(atm, spot=spot)


def _panel_features(panel: DatedOptionPanel, *, spot: float) -> Tuple[float, float, float]:
    with_iv = [(o, _safe_iv(o, spot=spot)) for o in panel.options]
    with_iv = [(o, iv) for o, iv in with_iv if iv is not None]
    if len(with_iv) < 3:
        return 0.2, 0.0, 0.0

    maturities = sorted({float(o.maturity) for o, _ in with_iv})
    short_m = maturities[0]
    long_m = maturities[-1]
    med_m = maturities[len(maturities) // 2]

    atm_med = _atm_iv_for_maturity([o for o, _ in with_iv], spot=spot, maturity=med_m)
    if atm_med is None:
        atm_med = float(np.mean([iv for _, iv in with_iv]))

    atm_short = _atm_iv_for_maturity([o for o, _ in with_iv], spot=spot, maturity=short_m)
    atm_long = _atm_iv_for_maturity([o for o, _ in with_iv], spot=spot, maturity=long_m)
    if atm_short is None:
        atm_short = atm_med
    if atm_long is None:
        atm_long = atm_med
    term_slope = float(atm_long - atm_short)

    med_bucket = [(o, iv) for o, iv in with_iv if abs(float(o.maturity) - med_m) < 1e-10]
    med_bucket.sort(key=lambda x: float(x[0].strike))
    ivs = [float(iv) for _, iv in med_bucket]
    if len(ivs) >= 4:
        low_iv = ivs[max(0, int(np.floor(0.25 * (len(ivs) - 1))))]
        high_iv = ivs[min(len(ivs) - 1, int(np.ceil(0.75 * (len(ivs) - 1))))]
    else:
        low_iv = ivs[0]
        high_iv = ivs[-1]
    skew = float(low_iv - high_iv)
    return float(atm_med), skew, term_slope


def _extract_observation(obs: object) -> Optional[Tuple[str, str, float]]:
    if isinstance(obs, dict):
        model = obs.get("model")
        target_date = obs.get("target_date")
        rmse = obs.get("rmse")
    else:
        model = getattr(obs, "model", None)
        target_date = getattr(obs, "target_date", None)
        rmse = getattr(obs, "rmse", None)
    if model is None or target_date is None or rmse is None:
        return None
    return str(model), str(target_date), float(rmse)


def _transition_payload(regimes_by_date: Dict[str, str]) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, float]], float]:
    dates = sorted(regimes_by_date.keys(), key=lambda d: date.fromisoformat(d))
    counts: Dict[str, Dict[str, int]] = {}
    same = 0
    total = 0
    for d0, d1 in zip(dates[:-1], dates[1:]):
        r0 = regimes_by_date[d0]
        r1 = regimes_by_date[d1]
        counts.setdefault(r0, {})
        counts[r0][r1] = counts[r0].get(r1, 0) + 1
        total += 1
        if r0 == r1:
            same += 1

    probs: Dict[str, Dict[str, float]] = {}
    for src, row in counts.items():
        row_total = float(sum(row.values()))
        probs[src] = {dst: float(cnt / row_total) for dst, cnt in row.items()} if row_total > 0 else {}
    persistence = float(same / total) if total > 0 else 0.0
    return counts, probs, persistence


def run_regime_diagnostics(
    panels: Sequence[DatedOptionPanel],
    *,
    forecast_observations: Sequence[object],
    spot: float = 100.0,
) -> RegimeDiagnosticsResult:
    """Classify dated option-panels into regimes and summarize model performance by regime."""

    if len(panels) < 3:
        raise ValueError("Need at least 3 dated panels for regime diagnostics")

    ordered = sorted(panels, key=lambda p: p.quote_date)
    base_features: List[Tuple[str, float, float, float]] = []
    for panel in ordered:
        atm_iv, skew, term = _panel_features(panel, spot=spot)
        base_features.append((panel.quote_date, atm_iv, skew, term))

    atm = np.asarray([x[1] for x in base_features], dtype=float)
    skew = np.asarray([x[2] for x in base_features], dtype=float)
    term = np.asarray([x[3] for x in base_features], dtype=float)
    q_low, q_high = np.quantile(atm, [0.33, 0.67]) if atm.size >= 3 else (float(np.min(atm)), float(np.max(atm)))
    skew_med = float(np.median(skew))
    term_med = float(np.median(term))
    atm_dispersion = float(np.ptp(atm)) if atm.size > 0 else 0.0
    skew_dispersion = float(np.ptp(skew)) if skew.size > 0 else 0.0
    order = np.argsort(atm)
    rank = np.empty_like(order)
    rank[order] = np.arange(atm.size)

    features: List[RegimeFeature] = []
    regimes_by_date: Dict[str, str] = {}
    regime_iv_values: Dict[str, List[float]] = {}
    for i, (qd, atm_iv, skew_val, term_slope) in enumerate(base_features):
        if atm_dispersion < 1e-8:
            frac = float(rank[i] / max(atm.size - 1, 1))
            if frac <= 0.33:
                vol_state = "low_vol"
            elif frac <= 0.67:
                vol_state = "mid_vol"
            else:
                vol_state = "high_vol"
        elif atm_iv <= q_low:
            vol_state = "low_vol"
        elif atm_iv <= q_high:
            vol_state = "mid_vol"
        else:
            vol_state = "high_vol"

        if skew_dispersion < 1e-8:
            skew_state = "up_term" if term_slope >= term_med else "down_term"
        else:
            skew_state = "left_skew" if skew_val >= skew_med else "right_skew"
        regime = f"{vol_state}_{skew_state}"
        features.append(
            RegimeFeature(
                quote_date=qd,
                atm_iv=float(atm_iv),
                skew_25_75=float(skew_val),
                term_slope=float(term_slope),
                regime=regime,
            )
        )
        regimes_by_date[qd] = regime
        regime_iv_values.setdefault(regime, []).append(float(atm_iv))

    transition_counts, transition_probs, persistence = _transition_payload(regimes_by_date)

    by_regime_model: Dict[Tuple[str, str], List[float]] = {}
    for obs in forecast_observations:
        parsed = _extract_observation(obs)
        if parsed is None:
            continue
        model, target_date, rmse = parsed
        regime = regimes_by_date.get(target_date)
        if regime is None:
            continue
        by_regime_model.setdefault((regime, model), []).append(float(rmse))

    perf_rows: List[RegimeModelPerformance] = []
    best_model_by_regime: Dict[str, str] = {}
    regime_model_mean: Dict[str, Dict[str, float]] = {}
    for (regime, model), vals in sorted(by_regime_model.items()):
        arr = np.asarray(vals, dtype=float)
        perf_rows.append(
            RegimeModelPerformance(
                regime=regime,
                model=model,
                mean_rmse=float(np.mean(arr)),
                std_rmse=float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
                count=int(arr.size),
            )
        )
        regime_model_mean.setdefault(regime, {})[model] = float(np.mean(arr))

    for regime, model_scores in regime_model_mean.items():
        best_model_by_regime[regime] = min(model_scores.keys(), key=lambda m: model_scores[m])

    high_vol_regime = max(regime_iv_values.keys(), key=lambda r: float(np.mean(regime_iv_values[r])))
    high_vol_best_model = best_model_by_regime.get(high_vol_regime)

    return RegimeDiagnosticsResult(
        num_dates=len(ordered),
        num_regimes=len(set(regimes_by_date.values())),
        regime_by_date=regimes_by_date,
        features=features,
        transition_counts=transition_counts,
        transition_probabilities=transition_probs,
        persistence_ratio=float(persistence),
        model_performance=perf_rows,
        best_model_by_regime=best_model_by_regime,
        high_vol_regime=high_vol_regime,
        high_vol_best_model=high_vol_best_model,
    )


def regime_diagnostics_to_dict(result: RegimeDiagnosticsResult) -> Dict[str, object]:
    """Serialize regime diagnostics dataclasses for JSON artifacts."""

    return {
        "num_dates": result.num_dates,
        "num_regimes": result.num_regimes,
        "regime_by_date": dict(result.regime_by_date),
        "features": [asdict(x) for x in result.features],
        "transition_counts": result.transition_counts,
        "transition_probabilities": result.transition_probabilities,
        "persistence_ratio": result.persistence_ratio,
        "model_performance": [asdict(x) for x in result.model_performance],
        "best_model_by_regime": dict(result.best_model_by_regime),
        "high_vol_regime": result.high_vol_regime,
        "high_vol_best_model": result.high_vol_best_model,
    }
