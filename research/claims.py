"""Research claims and evaluation logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .statistics import ConfidenceInterval, bootstrap_ci, paired_t_test


@dataclass
class ResearchClaim:
    """A falsifiable claim evaluated on paired experiment metrics."""

    claim_id: str
    title: str
    metric: str
    direction: str  # "increase" or "decrease"
    minimum_effect: float
    alpha: float = 0.05


@dataclass
class ClaimEvaluation:
    """Result of evaluating one claim."""

    claim_id: str
    passed: bool
    observed_effect: float
    confidence_interval: ConfidenceInterval
    p_value: float
    details: str


def default_research_claims() -> List[ResearchClaim]:
    """Default doctoral-level claims for the benchmark pipeline."""

    return [
        ResearchClaim(
            claim_id="C1",
            title="Variance reduction improves RMSE-cost efficiency",
            metric="std_error_times_runtime",
            direction="decrease",
            minimum_effect=0.10,
            alpha=0.05,
        ),
        ResearchClaim(
            claim_id="C2",
            title="MLMC runtime improves for matched error",
            metric="runtime_seconds",
            direction="decrease",
            minimum_effect=0.10,
            alpha=0.05,
        ),
        ResearchClaim(
            claim_id="C3",
            title="Calibration bootstrap dispersion remains below tolerance",
            metric="heston_stability_error",
            direction="decrease",
            minimum_effect=0.10,
            alpha=0.05,
        ),
    ]


def evaluate_claim(
    claim: ResearchClaim,
    baseline_samples: np.ndarray,
    candidate_samples: np.ndarray,
    seed: int = 42,
) -> ClaimEvaluation:
    """Evaluate a claim using paired differences and bootstrap CI."""

    baseline = np.asarray(baseline_samples, dtype=float).ravel()
    candidate = np.asarray(candidate_samples, dtype=float).ravel()
    if baseline.shape != candidate.shape:
        raise ValueError("baseline and candidate samples must have identical shape")

    # Effect is relative improvement from baseline.
    denom = np.maximum(np.abs(baseline), 1e-12)
    relative_improvement = (baseline - candidate) / denom

    if claim.direction == "increase":
        # For increase claims, invert semantics so positive means improvement.
        relative_improvement = -relative_improvement
        test = paired_t_test(baseline, candidate, alternative="greater")
    elif claim.direction == "decrease":
        test = paired_t_test(baseline, candidate, alternative="less")
    else:
        raise ValueError("claim.direction must be 'increase' or 'decrease'")

    ci = bootstrap_ci(relative_improvement, statistic=np.mean, seed=seed)
    effect = float(np.mean(relative_improvement))

    passed = (effect >= claim.minimum_effect) and (test.p_value < claim.alpha)
    details = (
        f"effect={effect:.4f}, threshold={claim.minimum_effect:.4f}, "
        f"p={test.p_value:.4g}, alpha={claim.alpha:.4g}, "
        f"ci=[{ci.low:.4f}, {ci.high:.4f}]"
    )

    return ClaimEvaluation(
        claim_id=claim.claim_id,
        passed=passed,
        observed_effect=effect,
        confidence_interval=ci,
        p_value=test.p_value,
        details=details,
    )


def evaluate_claim_bundle(
    claims: List[ResearchClaim],
    metric_samples: Dict[str, Dict[str, np.ndarray]],
    seed: int = 42,
) -> List[ClaimEvaluation]:
    """Evaluate all claims from named baseline/candidate metric samples."""

    evaluations: List[ClaimEvaluation] = []
    for claim in claims:
        samples = metric_samples.get(claim.metric)
        if samples is None or "baseline" not in samples or "candidate" not in samples:
            raise KeyError(f"Missing metric sample bundle for '{claim.metric}'")

        evaluations.append(
            evaluate_claim(
                claim,
                baseline_samples=samples["baseline"],
                candidate_samples=samples["candidate"],
                seed=seed,
            )
        )

    return evaluations
