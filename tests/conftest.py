"""Pytest fixtures for Monte Carlo tests."""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mc_pricer import (
    MarketData, HestonParams, JumpParams, SimulationConfig,
    Backend, DiscretizationScheme, VarianceReduction, PayoffType
)


@pytest.fixture
def market_data():
    """Standard market data for tests."""
    return MarketData(S0=100.0, r=0.05, q=0.0)


@pytest.fixture
def heston_params():
    """Standard Heston parameters satisfying Feller condition."""
    return HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)


@pytest.fixture
def jump_params():
    """Standard jump parameters."""
    return JumpParams(lambda_j=0.1, mu_j=-0.05, sigma_j=0.1)


@pytest.fixture
def sim_config():
    """Standard simulation configuration."""
    return SimulationConfig(
        num_paths=10000,
        num_steps=100,
        backend=Backend.NUMPY,
        scheme=DiscretizationScheme.QE,
        variance_reduction=VarianceReduction.ANTITHETIC,
        seed=42
    )


@pytest.fixture
def high_accuracy_config():
    """High accuracy configuration for precision tests."""
    return SimulationConfig(
        num_paths=100000,
        num_steps=252,
        backend=Backend.NUMPY,
        scheme=DiscretizationScheme.QE,
        variance_reduction=VarianceReduction.ANTITHETIC_CV,
        seed=42
    )


@pytest.fixture
def rng():
    """Reproducible random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_returns(rng):
    """Sample daily returns for risk metrics tests."""
    return rng.normal(0.0005, 0.02, 252)


@pytest.fixture
def sample_paths(rng):
    """Sample price paths for testing."""
    num_paths = 1000
    num_steps = 100
    S0 = 100.0
    drift = 0.05 / 252
    vol = 0.2 / np.sqrt(252)

    Z = rng.standard_normal((num_paths, num_steps))
    log_returns = (drift - 0.5 * vol**2) + vol * Z
    log_returns = np.cumsum(log_returns, axis=1)

    paths = np.zeros((num_paths, num_steps + 1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(log_returns)

    return paths
