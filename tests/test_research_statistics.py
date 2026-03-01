"""Tests for research.statistics module."""

import numpy as np

from research.statistics import mean_confidence_interval, paired_t_test, sequential_halfwidth_stop


def test_mean_confidence_interval_contains_sample_mean():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ci = mean_confidence_interval(x, confidence=0.95)
    assert ci.low <= np.mean(x) <= ci.high
    assert ci.n == 5


def test_paired_t_test_detects_improvement():
    baseline = np.array([1.0, 1.2, 0.9, 1.1, 1.3, 1.0])
    candidate = baseline - 0.2
    res = paired_t_test(baseline, candidate, alternative="less")
    assert res.p_value < 0.05


def test_sequential_halfwidth_stop_reaches_precision():
    x = np.linspace(0.99, 1.01, 60)
    decision = sequential_halfwidth_stop(x, rel_tolerance=0.02, min_n=20)
    assert decision.stop is True
