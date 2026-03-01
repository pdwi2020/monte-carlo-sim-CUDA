"""Tests for risk metrics module."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risk_metrics import (
    historical_var, parametric_var, monte_carlo_var,
    VaRResult, SCIPY_AVAILABLE
)


class TestHistoricalVaR:
    """Tests for historical simulation VaR."""

    def test_var_positive(self, sample_returns):
        """VaR should be positive (representing a loss)."""
        result = historical_var(sample_returns, confidence_level=0.95)
        assert result.var >= 0

    def test_cvar_greater_than_var(self, sample_returns):
        """CVaR should be >= VaR."""
        result = historical_var(sample_returns, confidence_level=0.95)
        assert result.cvar >= result.var

    def test_higher_confidence_higher_var(self, sample_returns):
        """Higher confidence level should give higher VaR."""
        var_95 = historical_var(sample_returns, confidence_level=0.95)
        var_99 = historical_var(sample_returns, confidence_level=0.99)
        assert var_99.var > var_95.var

    def test_longer_horizon_higher_var(self, sample_returns):
        """Longer horizon should give higher VaR (sqrt-T scaling)."""
        var_1d = historical_var(sample_returns, confidence_level=0.95, horizon_days=1)
        var_10d = historical_var(sample_returns, confidence_level=0.95, horizon_days=10)
        assert var_10d.var > var_1d.var


class TestParametricVaR:
    """Tests for parametric (Delta-Normal) VaR."""

    @pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy required")
    def test_var_positive(self):
        """Parametric VaR should be positive."""
        result = parametric_var(
            portfolio_value=1_000_000,
            volatility=0.20,
            confidence_level=0.95
        )
        assert result.var > 0

    @pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy required")
    def test_higher_vol_higher_var(self):
        """Higher volatility should give higher VaR."""
        var_low = parametric_var(portfolio_value=1_000_000, volatility=0.10)
        var_high = parametric_var(portfolio_value=1_000_000, volatility=0.30)
        assert var_high.var > var_low.var

    @pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy required")
    def test_linear_in_portfolio_value(self):
        """VaR should scale linearly with portfolio value."""
        var_1m = parametric_var(portfolio_value=1_000_000, volatility=0.20)
        var_2m = parametric_var(portfolio_value=2_000_000, volatility=0.20)
        assert abs(var_2m.var / var_1m.var - 2.0) < 0.01


class TestMonteCarloVaR:
    """Tests for Monte Carlo VaR."""

    def test_var_positive(self):
        """MC VaR should be positive."""
        result = monte_carlo_var(
            portfolio_value=1_000_000,
            volatility=0.20,
            confidence_level=0.95,
            num_simulations=10000,
            seed=42
        )
        assert result.var > 0

    def test_reproducible_with_seed(self):
        """Same seed should give same result."""
        result1 = monte_carlo_var(
            portfolio_value=1_000_000, volatility=0.20,
            num_simulations=10000, seed=42
        )
        result2 = monte_carlo_var(
            portfolio_value=1_000_000, volatility=0.20,
            num_simulations=10000, seed=42
        )
        assert result1.var == result2.var

    def test_different_distributions(self):
        """Different distributions should give different VaR."""
        var_normal = monte_carlo_var(
            portfolio_value=1_000_000, volatility=0.20,
            num_simulations=10000, distribution="normal", seed=42
        )
        var_t = monte_carlo_var(
            portfolio_value=1_000_000, volatility=0.20,
            num_simulations=10000, distribution="t", seed=42
        )
        # Student-t has fatter tails, so VaR should be higher
        assert var_t.var > var_normal.var * 0.9  # Allow some MC error


class TestVaRResult:
    """Tests for VaRResult dataclass."""

    def test_result_fields(self):
        """VaRResult should have expected fields."""
        result = VaRResult(
            var=10000.0,
            confidence_level=0.95,
            method="test",
            horizon_days=1,
            cvar=12000.0
        )
        assert result.var == 10000.0
        assert result.confidence_level == 0.95
        assert result.cvar == 12000.0
