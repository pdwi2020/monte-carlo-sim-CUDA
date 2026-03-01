"""Tests for microstructure-aware hedging stress module."""

from research.execution_microstructure import run_microstructure_hedging_study


def test_microstructure_hedging_runs():
    out = run_microstructure_hedging_study(
        true_model="gbm",
        num_paths=500,
        num_steps=20,
        seed=67,
    )
    assert "frictionless" in out
    assert "microstructure_stressed" in out
    assert out["cvar95_degradation_ratio"] > 0.0
