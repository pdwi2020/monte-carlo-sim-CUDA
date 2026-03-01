"""Tests for econometric validation utilities."""

import numpy as np

from research.econometrics import (
    diebold_mariano_test,
    holm_bonferroni_correction,
    model_confidence_set,
    moving_block_bootstrap_ci,
    superior_predictive_ability_test,
    white_reality_check,
)


def test_diebold_mariano_and_holm():
    rng = np.random.default_rng(11)
    loss_a = rng.normal(1.0, 0.1, size=40)
    loss_b = rng.normal(0.9, 0.1, size=40)
    out = diebold_mariano_test(loss_a, loss_b, alternative="greater", lag=1)
    assert 0.0 <= out.p_value <= 1.0
    assert out.n == 40

    holm = holm_bonferroni_correction([0.01, 0.04, 0.2], alpha=0.05)
    assert len(holm["adjusted_p_values"]) == 3
    assert len(holm["reject"]) == 3


def test_block_bootstrap_ci():
    x = np.linspace(0.0, 1.0, 20)
    out = moving_block_bootstrap_ci(x, block_size=4, n_bootstrap=200, seed=7)
    assert out.low <= out.mean <= out.high


def test_spa_white_and_mcs():
    rng = np.random.default_rng(77)
    n = 50
    naive = rng.normal(1.0, 0.06, size=n)
    heston = naive - 0.06 + rng.normal(0.0, 0.01, size=n)
    rough = naive - 0.09 + rng.normal(0.0, 0.01, size=n)
    losses = {
        "naive_last_surface": naive,
        "heston": heston,
        "rough_heston": rough,
    }

    spa = superior_predictive_ability_test(
        losses,
        benchmark_model="naive_last_surface",
        n_bootstrap=300,
        block_size=3,
        seed=78,
    )
    assert 0.0 <= spa.p_value <= 1.0
    assert spa.best_model in {"heston", "rough_heston"}

    rc = white_reality_check(
        losses,
        benchmark_model="naive_last_surface",
        n_bootstrap=300,
        block_size=3,
        seed=79,
    )
    assert 0.0 <= rc.p_value <= 1.0
    assert rc.best_model in {"heston", "rough_heston"}

    mcs = model_confidence_set(losses, alpha=0.10, n_bootstrap=300, block_size=3, seed=80)
    assert len(mcs.surviving_models) >= 1
    assert "naive_last_surface" in mcs.elimination_p_values
