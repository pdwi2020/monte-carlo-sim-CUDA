"""Tests for research claim evaluation."""

import numpy as np

from research.claims import ResearchClaim, evaluate_claim


def test_decrease_claim_passes_when_candidate_is_lower():
    claim = ResearchClaim(
        claim_id="C-test",
        title="Lower is better",
        metric="runtime",
        direction="decrease",
        minimum_effect=0.10,
        alpha=0.05,
    )
    baseline = np.array([10.0, 11.0, 9.5, 10.5, 10.2])
    candidate = baseline * 0.7

    ev = evaluate_claim(claim, baseline, candidate, seed=123)
    assert ev.passed is True
    assert ev.observed_effect > 0.1
