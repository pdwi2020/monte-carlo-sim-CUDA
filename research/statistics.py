"""Statistical utilities for research-grade experiment analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

try:
    from scipy import stats

    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback path
    stats = None
    SCIPY_AVAILABLE = False


@dataclass
class ConfidenceInterval:
    """Mean confidence interval summary."""

    mean: float
    low: float
    high: float
    std: float
    n: int


@dataclass
class HypothesisTestResult:
    """Hypothesis test result with test statistic and p-value."""

    statistic: float
    p_value: float
    alternative: str
    n: int


@dataclass
class SequentialStoppingResult:
    """Sequential stopping decision based on CI half-width."""

    stop: bool
    n: int
    half_width: float
    threshold: float
    reason: str


def _validate_1d_samples(samples: np.ndarray) -> np.ndarray:
    arr = np.asarray(samples, dtype=float).ravel()
    if arr.size < 2:
        raise ValueError("At least 2 samples are required")
    return arr


def mean_confidence_interval(samples: np.ndarray, confidence: float = 0.95) -> ConfidenceInterval:
    """Return a confidence interval for sample mean using t-interval (or normal fallback)."""

    x = _validate_1d_samples(samples)
    n = x.size
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1))
    alpha = 1.0 - confidence

    if SCIPY_AVAILABLE:
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
    else:  # pragma: no cover - fallback path
        # Conservative approximation close to 95% t critical for moderate n
        t_crit = 1.96 if confidence >= 0.95 else 1.645

    half_width = t_crit * std / np.sqrt(n)
    return ConfidenceInterval(
        mean=mean,
        low=mean - half_width,
        high=mean + half_width,
        std=std,
        n=n,
    )


def bootstrap_ci(
    samples: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: Optional[int] = 42,
) -> ConfidenceInterval:
    """Bootstrap confidence interval for an arbitrary scalar statistic."""

    x = _validate_1d_samples(samples)
    rng = np.random.default_rng(seed)
    n = x.size

    boot = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        resample = x[rng.integers(0, n, size=n)]
        boot[i] = float(statistic(resample))

    alpha = 1.0 - confidence
    low = float(np.quantile(boot, alpha / 2.0))
    high = float(np.quantile(boot, 1.0 - alpha / 2.0))

    return ConfidenceInterval(
        mean=float(np.mean(boot)),
        low=low,
        high=high,
        std=float(np.std(boot, ddof=1)),
        n=n_bootstrap,
    )


def paired_t_test(
    baseline: np.ndarray,
    candidate: np.ndarray,
    alternative: str = "two-sided",
) -> HypothesisTestResult:
    """Paired t-test on candidate - baseline differences."""

    x = _validate_1d_samples(baseline)
    y = _validate_1d_samples(candidate)
    if x.shape != y.shape:
        raise ValueError("baseline and candidate must have the same shape")

    d = y - x
    n = d.size
    mean_d = float(np.mean(d))
    std_d = float(np.std(d, ddof=1))
    if std_d == 0.0:
        # Degenerate case: no uncertainty.
        if mean_d == 0.0:
            p = 1.0
            stat = 0.0
        else:
            p = 0.0
            stat = np.inf if mean_d > 0 else -np.inf
        return HypothesisTestResult(statistic=float(stat), p_value=float(p), alternative=alternative, n=n)

    stat = mean_d / (std_d / np.sqrt(n))

    if SCIPY_AVAILABLE:
        if alternative == "two-sided":
            p = float(2.0 * stats.t.sf(abs(stat), df=n - 1))
        elif alternative == "greater":
            p = float(stats.t.sf(stat, df=n - 1))
        elif alternative == "less":
            p = float(stats.t.cdf(stat, df=n - 1))
        else:
            raise ValueError("alternative must be one of: two-sided, greater, less")
    else:  # pragma: no cover - fallback path
        # Normal approximation fallback
        from math import erf, sqrt

        z = float(stat)
        cdf = 0.5 * (1.0 + erf(z / sqrt(2.0)))
        if alternative == "two-sided":
            p = float(2.0 * min(cdf, 1.0 - cdf))
        elif alternative == "greater":
            p = float(1.0 - cdf)
        elif alternative == "less":
            p = float(cdf)
        else:
            raise ValueError("alternative must be one of: two-sided, greater, less")

    return HypothesisTestResult(
        statistic=float(stat),
        p_value=p,
        alternative=alternative,
        n=n,
    )


def sequential_halfwidth_stop(
    samples: np.ndarray,
    rel_tolerance: float,
    confidence: float = 0.95,
    min_n: int = 20,
) -> SequentialStoppingResult:
    """Check whether sample mean CI half-width is below relative tolerance."""

    x = _validate_1d_samples(samples)
    n = x.size
    if n < min_n:
        return SequentialStoppingResult(
            stop=False,
            n=n,
            half_width=float("inf"),
            threshold=float("inf"),
            reason=f"need at least {min_n} samples",
        )

    ci = mean_confidence_interval(x, confidence=confidence)
    half_width = 0.5 * (ci.high - ci.low)
    threshold = rel_tolerance * max(abs(ci.mean), 1e-12)

    if half_width <= threshold:
        return SequentialStoppingResult(
            stop=True,
            n=n,
            half_width=half_width,
            threshold=threshold,
            reason="confidence interval target reached",
        )

    return SequentialStoppingResult(
        stop=False,
        n=n,
        half_width=half_width,
        threshold=threshold,
        reason="insufficient precision",
    )
