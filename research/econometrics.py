"""Econometric validation tools for doctoral-grade research claims."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np

try:
    from scipy.stats import norm

    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover
    norm = None
    SCIPY_AVAILABLE = False


@dataclass
class DieboldMarianoResult:
    """Diebold-Mariano predictive accuracy test output."""

    statistic: float
    p_value: float
    mean_loss_diff: float
    n: int
    lag: int


@dataclass
class BootstrapSeriesCI:
    """Bootstrap confidence interval on time-series statistic."""

    mean: float
    low: float
    high: float
    n_bootstrap: int
    block_size: int


@dataclass
class WhiteRealityCheckResult:
    """White's Reality Check style max-performance test result."""

    statistic: float
    p_value: float
    benchmark_model: str
    best_model: str
    n: int
    n_bootstrap: int
    block_size: int


@dataclass
class SuperiorPredictiveAbilityResult:
    """Hansen-SPA style superior predictive ability test result."""

    statistic: float
    p_value: float
    benchmark_model: str
    best_model: str
    n: int
    n_bootstrap: int
    block_size: int


@dataclass
class ModelConfidenceSetResult:
    """Bootstrap elimination model-confidence-set style output."""

    alpha: float
    surviving_models: List[str]
    eliminated_models: List[str]
    elimination_p_values: Dict[str, float]
    n: int
    block_size: int


def _normal_cdf(x: float) -> float:
    if SCIPY_AVAILABLE:
        return float(norm.cdf(x))
    from math import erf, sqrt

    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _as_aligned_loss_matrix(losses_by_model: Dict[str, Sequence[float]]) -> tuple[List[str], np.ndarray]:
    """Align loss vectors and drop rows with non-finite values."""

    if len(losses_by_model) < 2:
        raise ValueError("Need at least two models")

    names = sorted(losses_by_model.keys())
    arrays = [np.asarray(losses_by_model[name], dtype=float).ravel() for name in names]
    n = min(arr.size for arr in arrays)
    if n < 5:
        raise ValueError("Need at least 5 observations per model")
    mat = np.vstack([arr[:n] for arr in arrays])
    mask = np.all(np.isfinite(mat), axis=0)
    mat = mat[:, mask]
    if mat.shape[1] < 5:
        raise ValueError("Need at least 5 aligned finite observations across models")
    return names, mat


def _moving_block_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Draw moving-block bootstrap indices."""

    b = int(max(1, min(block_size, n)))
    starts = np.arange(0, n - b + 1)
    out: List[int] = []
    while len(out) < n:
        s = int(rng.choice(starts))
        out.extend(range(s, s + b))
    return np.asarray(out[:n], dtype=int)


def diebold_mariano_test(
    loss_a: Sequence[float],
    loss_b: Sequence[float],
    *,
    alternative: str = "two-sided",
    lag: int = 1,
) -> DieboldMarianoResult:
    """Diebold-Mariano test with Newey-West style long-run variance."""

    a = np.asarray(loss_a, dtype=float).ravel()
    b = np.asarray(loss_b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError("loss_a and loss_b must have identical shape")
    if a.size < 5:
        raise ValueError("Need at least 5 paired observations")

    d = a - b
    n = d.size
    d_mean = float(np.mean(d))

    q = int(max(1, lag))
    gamma0 = float(np.var(d, ddof=1))
    lrv = gamma0
    centered = d - d_mean
    for k in range(1, min(q + 1, n - 1)):
        w = 1.0 - k / (q + 1.0)
        cov = float(np.dot(centered[k:], centered[:-k]) / n)
        lrv += 2.0 * w * cov
    lrv = max(lrv, 1e-12)
    stat = d_mean / np.sqrt(lrv / n)

    if alternative == "less":
        p = _normal_cdf(stat)
    elif alternative == "greater":
        p = 1.0 - _normal_cdf(stat)
    elif alternative == "two-sided":
        p = 2.0 * min(_normal_cdf(stat), 1.0 - _normal_cdf(stat))
    else:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")

    return DieboldMarianoResult(
        statistic=float(stat),
        p_value=float(np.clip(p, 0.0, 1.0)),
        mean_loss_diff=d_mean,
        n=int(n),
        lag=q,
    )


def moving_block_bootstrap_ci(
    series: Sequence[float],
    *,
    block_size: int = 4,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> BootstrapSeriesCI:
    """Moving-block bootstrap CI for dependent series mean."""

    x = np.asarray(series, dtype=float).ravel()
    if x.size < 2:
        raise ValueError("Need at least 2 observations")
    b = int(max(1, min(block_size, x.size)))
    n = x.size
    rng = np.random.default_rng(seed)

    starts = np.arange(0, n - b + 1)
    samples = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx: List[int] = []
        while len(idx) < n:
            s = int(rng.choice(starts))
            idx.extend(range(s, s + b))
        idx = idx[:n]
        samples[i] = float(np.mean(x[idx]))

    low_q = alpha / 2.0
    high_q = 1.0 - alpha / 2.0
    low, high = np.quantile(samples, [low_q, high_q])
    return BootstrapSeriesCI(
        mean=float(np.mean(samples)),
        low=float(low),
        high=float(high),
        n_bootstrap=int(n_bootstrap),
        block_size=b,
    )


def holm_bonferroni_correction(p_values: Iterable[float], *, alpha: float = 0.05) -> Dict[str, object]:
    """Holm-Bonferroni correction for multiple hypothesis tests."""

    p = [float(x) for x in p_values]
    m = len(p)
    indexed = sorted(enumerate(p), key=lambda x: x[1])
    adjusted = [0.0] * m
    reject = [False] * m

    running_max = 0.0
    for rank, (idx, pv) in enumerate(indexed, start=1):
        mult = m - rank + 1
        adj = min(1.0, pv * mult)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max

    for idx, adj in enumerate(adjusted):
        reject[idx] = adj <= alpha

    return {
        "alpha": alpha,
        "p_values": p,
        "adjusted_p_values": adjusted,
        "reject": reject,
    }


def white_reality_check(
    losses_by_model: Dict[str, Sequence[float]],
    *,
    benchmark_model: str,
    n_bootstrap: int = 1000,
    block_size: int = 4,
    seed: int = 42,
) -> WhiteRealityCheckResult:
    """White's Reality Check on maximum loss reduction vs a benchmark model."""

    names, mat = _as_aligned_loss_matrix(losses_by_model)
    if benchmark_model not in names:
        raise ValueError(f"benchmark_model '{benchmark_model}' not found")
    b_idx = names.index(benchmark_model)
    candidate_idx = [i for i in range(len(names)) if i != b_idx]
    if not candidate_idx:
        raise ValueError("Need at least one challenger model")

    benchmark = mat[b_idx]
    challengers = mat[candidate_idx]
    challenger_names = [names[i] for i in candidate_idx]

    diff = benchmark[None, :] - challengers
    mean_diff = np.mean(diff, axis=1)
    best_i = int(np.argmax(mean_diff))
    obs_stat = float(max(np.max(mean_diff), 0.0))

    centered = diff - mean_diff[:, None]
    n = mat.shape[1]
    rng = np.random.default_rng(seed)
    boot_stats = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = _moving_block_indices(n, block_size=block_size, rng=rng)
        boot_stats[i] = float(max(np.max(np.mean(centered[:, idx], axis=1)), 0.0))
    p_val = float((1.0 + np.sum(boot_stats >= obs_stat)) / (n_bootstrap + 1.0))

    return WhiteRealityCheckResult(
        statistic=obs_stat,
        p_value=p_val,
        benchmark_model=benchmark_model,
        best_model=challenger_names[best_i],
        n=int(n),
        n_bootstrap=int(n_bootstrap),
        block_size=int(max(1, min(block_size, n))),
    )


def superior_predictive_ability_test(
    losses_by_model: Dict[str, Sequence[float]],
    *,
    benchmark_model: str,
    n_bootstrap: int = 1000,
    block_size: int = 4,
    seed: int = 42,
) -> SuperiorPredictiveAbilityResult:
    """Hansen-SPA style max t-statistic test vs a benchmark model."""

    names, mat = _as_aligned_loss_matrix(losses_by_model)
    if benchmark_model not in names:
        raise ValueError(f"benchmark_model '{benchmark_model}' not found")
    b_idx = names.index(benchmark_model)
    candidate_idx = [i for i in range(len(names)) if i != b_idx]
    if not candidate_idx:
        raise ValueError("Need at least one challenger model")

    benchmark = mat[b_idx]
    challengers = mat[candidate_idx]
    challenger_names = [names[i] for i in candidate_idx]

    diff = benchmark[None, :] - challengers
    n = diff.shape[1]
    means = np.mean(diff, axis=1)
    stds = np.std(diff, axis=1, ddof=1)
    stds = np.where(stds < 1e-12, 1e-12, stds)

    t_obs_all = np.sqrt(n) * means / stds
    obs_stat = float(max(np.max(t_obs_all), 0.0))
    best_i = int(np.argmax(t_obs_all))

    centered = diff - means[:, None]
    rng = np.random.default_rng(seed)
    boot_stats = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = _moving_block_indices(n, block_size=block_size, rng=rng)
        boot_means = np.mean(centered[:, idx], axis=1)
        boot_stats[i] = float(max(np.max(np.sqrt(n) * boot_means / stds), 0.0))
    p_val = float((1.0 + np.sum(boot_stats >= obs_stat)) / (n_bootstrap + 1.0))

    return SuperiorPredictiveAbilityResult(
        statistic=obs_stat,
        p_value=p_val,
        benchmark_model=benchmark_model,
        best_model=challenger_names[best_i],
        n=int(n),
        n_bootstrap=int(n_bootstrap),
        block_size=int(max(1, min(block_size, n))),
    )


def model_confidence_set(
    losses_by_model: Dict[str, Sequence[float]],
    *,
    alpha: float = 0.10,
    n_bootstrap: int = 800,
    block_size: int = 4,
    seed: int = 42,
) -> ModelConfidenceSetResult:
    """Bootstrap elimination procedure approximating a model confidence set."""

    names, mat = _as_aligned_loss_matrix(losses_by_model)
    active = list(range(len(names)))
    eliminated: List[str] = []
    elimination_p: Dict[str, float] = {}
    n = mat.shape[1]
    rng = np.random.default_rng(seed)

    while len(active) > 1:
        active_losses = mat[active, :]
        means = np.mean(active_losses, axis=1)
        worst_local = int(np.argmax(means))
        best_local = int(np.argmin(means))
        worst_idx = active[worst_local]
        best_idx = active[best_local]

        diff = mat[worst_idx] - mat[best_idx]
        obs = float(np.mean(diff))
        centered = diff - obs
        boot = np.empty(n_bootstrap, dtype=float)
        for i in range(n_bootstrap):
            idx = _moving_block_indices(n, block_size=block_size, rng=rng)
            boot[i] = float(np.mean(centered[idx]))
        p_val = float((1.0 + np.sum(boot >= obs)) / (n_bootstrap + 1.0))
        elimination_p[names[worst_idx]] = p_val

        if p_val < alpha:
            eliminated.append(names[worst_idx])
            active = [j for j in active if j != worst_idx]
        else:
            break

    surviving = [names[i] for i in active]
    return ModelConfidenceSetResult(
        alpha=float(alpha),
        surviving_models=surviving,
        eliminated_models=eliminated,
        elimination_p_values=elimination_p,
        n=int(n),
        block_size=int(max(1, min(block_size, n))),
    )
