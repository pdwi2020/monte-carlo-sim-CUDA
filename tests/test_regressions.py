"""Regression tests for previously identified pricing bugs."""

import numpy as np
import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mc_pricer import (
    Backend,
    DiscretizationScheme,
    HestonParams,
    JumpParams,
    MarketData,
    PayoffType,
    BatchOption,
    SimulationConfig,
    VarianceReduction,
    price_batch_options,
    price_option,
    qe_variance_step,
)


def test_qe_variance_step_mixed_branch_assignment():
    """QE scheme should preserve masked-path assignment when branches are mixed."""
    v = np.array([0.0001, 0.01, 0.04, 0.5, 2.0], dtype=float)
    U = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)
    dt = 1.0 / 12.0
    kappa, theta, xi = 2.0, 0.04, 1.2

    exp_kdt = np.exp(-kappa * dt)
    m = theta + (v - theta) * exp_kdt
    s2 = (
        v * xi**2 * exp_kdt / kappa * (1 - exp_kdt)
        + theta * xi**2 / (2 * kappa) * (1 - exp_kdt) ** 2
    )
    psi = s2 / (m**2 + 1e-10)
    mask_quad = psi <= 1.5

    # Ensure this is actually a mixed-branch case.
    assert np.any(mask_quad)
    assert np.any(~mask_quad)

    out = qe_variance_step(v, U, dt, kappa, theta, xi, np)

    assert np.all(out >= 0.0)
    # Regression: with the old indexing bug, these were incorrectly zeroed.
    assert np.any(out[mask_quad] > 0.0)


def test_batch_parallel_uses_full_jump_parameter_key():
    """Parallel batch pricing should match sequential pricing for Bates options."""
    market = MarketData(S0=100.0, r=0.03)
    config = SimulationConfig(
        num_paths=12000,
        num_steps=40,
        backend=Backend.NUMPY,
        scheme=DiscretizationScheme.EULER,
        variance_reduction=VarianceReduction.NONE,
        seed=123,
    )
    heston = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.4)

    # Same lambda_j, different jump distribution moments.
    jump_a = JumpParams(lambda_j=0.5, mu_j=-0.15, sigma_j=0.10)
    jump_b = JumpParams(lambda_j=0.5, mu_j=0.15, sigma_j=0.35)

    options = [
        BatchOption(
            K=100.0,
            T=1.0,
            payoff_type=PayoffType.EUROPEAN_CALL,
            heston=heston,
            jump=jump_a,
        ),
        BatchOption(
            K=100.0,
            T=1.0,
            payoff_type=PayoffType.EUROPEAN_CALL,
            heston=heston,
            jump=jump_b,
        ),
    ]

    seq = price_batch_options(market, options, config=config, parallel=False)
    par = price_batch_options(market, options, config=config, parallel=True)

    assert np.allclose(par.prices, seq.prices)


def test_barrier_payoff_requires_barrier_parameters():
    """Barrier payoffs must raise a clear validation error when barrier is missing."""
    market = MarketData(S0=100.0, r=0.05)
    config = SimulationConfig(
        num_paths=2000,
        num_steps=30,
        backend=Backend.NUMPY,
        scheme=DiscretizationScheme.EULER,
        variance_reduction=VarianceReduction.ANTITHETIC,
        seed=42,
    )

    with pytest.raises(ValueError, match="Barrier parameters are required"):
        price_option(
            market=market,
            K=100.0,
            T=1.0,
            payoff_type=PayoffType.BARRIER_UP_OUT_CALL,
            config=config,
            sigma=0.2,
        )


def test_american_payoff_type_dispatches_through_price_option():
    """American payoffs should be supported through price_option dispatch."""
    market = MarketData(S0=100.0, r=0.05)
    config = SimulationConfig(
        num_paths=3000,
        num_steps=40,
        backend=Backend.NUMPY,
        scheme=DiscretizationScheme.EULER,
        variance_reduction=VarianceReduction.ANTITHETIC,
        seed=7,
    )

    result = price_option(
        market=market,
        K=100.0,
        T=1.0,
        payoff_type=PayoffType.AMERICAN_PUT,
        config=config,
        sigma=0.2,
    )

    assert result.price > 0.0
