"""Structural-break diagnostics for forecast and risk time series."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class StructuralBreakEntry:
    """One series structural-break summary."""

    series_id: str
    n: int
    break_index: int
    break_date: Optional[str]
    pre_mean: float
    post_mean: float
    mean_shift: float
    t_stat: float
    p_value: float
    cusum_max_abs: float


@dataclass
class StructuralBreakDiagnosticsResult:
    """Aggregate structural-break diagnostics."""

    min_segment_size: int
    n_bootstrap: int
    entries: List[StructuralBreakEntry]
    strongest_break_series: Optional[str]
    strongest_break_p_value: Optional[float]
    num_series: int


def _best_split_stats(x: np.ndarray, *, min_segment_size: int) -> tuple[int, float, float, float]:
    n = int(x.size)
    if n < 2 * min_segment_size:
        raise ValueError("Series too short for requested min_segment_size")

    best_idx = min_segment_size
    best_t = -1.0
    best_pre = float(np.mean(x[:min_segment_size]))
    best_post = float(np.mean(x[min_segment_size:]))

    for j in range(min_segment_size, n - min_segment_size + 1):
        a = x[:j]
        b = x[j:]
        ma = float(np.mean(a))
        mb = float(np.mean(b))
        va = float(np.var(a, ddof=1)) if a.size > 1 else 0.0
        vb = float(np.var(b, ddof=1)) if b.size > 1 else 0.0
        denom = np.sqrt(max(va / max(a.size, 1) + vb / max(b.size, 1), 1e-12))
        t = abs(mb - ma) / denom
        if t > best_t:
            best_t = float(t)
            best_idx = int(j)
            best_pre = ma
            best_post = mb

    return best_idx, best_t, best_pre, best_post


def _bootstrap_p_value(
    x: np.ndarray,
    *,
    observed_t: float,
    min_segment_size: int,
    n_bootstrap: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    n = int(x.size)
    if n < 2 * min_segment_size:
        return 1.0
    boot = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        xp = x[rng.permutation(n)]
        _, t, _, _ = _best_split_stats(xp, min_segment_size=min_segment_size)
        boot[i] = t
    return float((1.0 + np.sum(boot >= observed_t)) / (n_bootstrap + 1.0))


def _cusum_max_abs(x: np.ndarray) -> float:
    xc = x - float(np.mean(x))
    scale = float(np.std(xc, ddof=1))
    if scale < 1e-12:
        return 0.0
    c = np.cumsum(xc) / (scale * np.sqrt(max(x.size, 1)))
    return float(np.max(np.abs(c)))


def run_structural_break_diagnostics(
    *,
    dates_by_series: Dict[str, Sequence[str]],
    values_by_series: Dict[str, Sequence[float]],
    min_segment_size: int = 4,
    n_bootstrap: int = 400,
    seed: int = 42,
) -> StructuralBreakDiagnosticsResult:
    """Detect one dominant mean-shift break in each input series."""

    entries: List[StructuralBreakEntry] = []
    for idx, series_id in enumerate(sorted(values_by_series.keys())):
        x = np.asarray(values_by_series.get(series_id) or [], dtype=float).ravel()
        x = x[np.isfinite(x)]
        if x.size < max(8, 2 * min_segment_size):
            continue
        split, t_stat, pre_mean, post_mean = _best_split_stats(x, min_segment_size=min_segment_size)
        p_val = _bootstrap_p_value(
            x,
            observed_t=t_stat,
            min_segment_size=min_segment_size,
            n_bootstrap=n_bootstrap,
            seed=seed + 17 * idx,
        )
        dates = list(dates_by_series.get(series_id) or [])
        break_date = dates[split] if 0 <= split < len(dates) else None
        entries.append(
            StructuralBreakEntry(
                series_id=series_id,
                n=int(x.size),
                break_index=int(split),
                break_date=break_date,
                pre_mean=float(pre_mean),
                post_mean=float(post_mean),
                mean_shift=float(post_mean - pre_mean),
                t_stat=float(t_stat),
                p_value=float(p_val),
                cusum_max_abs=_cusum_max_abs(x),
            )
        )

    if entries:
        best = min(entries, key=lambda e: e.p_value)
        strongest = best.series_id
        strongest_p = best.p_value
    else:
        strongest = None
        strongest_p = None

    return StructuralBreakDiagnosticsResult(
        min_segment_size=int(min_segment_size),
        n_bootstrap=int(n_bootstrap),
        entries=entries,
        strongest_break_series=strongest,
        strongest_break_p_value=strongest_p,
        num_series=len(entries),
    )


def structural_breaks_to_dict(result: StructuralBreakDiagnosticsResult) -> Dict[str, object]:
    """Serialize structural-break output."""

    return {
        "min_segment_size": result.min_segment_size,
        "n_bootstrap": result.n_bootstrap,
        "entries": [asdict(x) for x in result.entries],
        "strongest_break_series": result.strongest_break_series,
        "strongest_break_p_value": result.strongest_break_p_value,
        "num_series": result.num_series,
    }
