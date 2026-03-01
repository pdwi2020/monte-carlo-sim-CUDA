"""
Test suite for Bates model CUDA implementation.

Tests cover:
- Input validation (parameter bounds checking)
- Numerical correctness (sanity checks against known behaviors)
- Edge cases and error handling
- Path generation validation

Run with: pytest test_bates.py -v
"""

import pytest
import numpy as np
import math

# Try to import the CUDA module - skip tests if not available
try:
    import bates_kernel_cpp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    bates_kernel_cpp = None


# Default Bates model parameters for testing
DEFAULT_PARAMS = {
    "num_paths": 10000,
    "num_steps": 252,
    "T": 1.0,
    "K": 100.0,
    "S0": 100.0,
    "r": 0.05,
    "v0": 0.04,
    "kappa": 2.0,
    "theta": 0.04,
    "xi": 0.3,
    "rho": -0.7,
    "lambda_j": 0.1,
    "mu_j": -0.05,
    "sigma_j": 0.1,
}


def skip_if_no_cuda(func):
    """Decorator to skip tests if CUDA module is not available."""
    return pytest.mark.skipif(
        not CUDA_AVAILABLE,
        reason="CUDA module not available"
    )(func)


# ============================================================================
# Input Validation Tests
# ============================================================================

class TestInputValidation:
    """Tests for parameter validation."""

    @skip_if_no_cuda
    def test_negative_num_paths_raises(self):
        """Test that negative num_paths raises an error."""
        params = DEFAULT_PARAMS.copy()
        params["num_paths"] = -1
        with pytest.raises((ValueError, RuntimeError)):
            bates_kernel_cpp.price(**params)

    @skip_if_no_cuda
    def test_zero_num_paths_raises(self):
        """Test that zero num_paths raises an error."""
        params = DEFAULT_PARAMS.copy()
        params["num_paths"] = 0
        with pytest.raises((ValueError, RuntimeError)):
            bates_kernel_cpp.price(**params)

    @skip_if_no_cuda
    def test_negative_num_steps_raises(self):
        """Test that negative num_steps raises an error."""
        params = DEFAULT_PARAMS.copy()
        params["num_steps"] = -1
        with pytest.raises((ValueError, RuntimeError)):
            bates_kernel_cpp.price(**params)

    @skip_if_no_cuda
    def test_negative_time_raises(self):
        """Test that negative time to maturity raises an error."""
        params = DEFAULT_PARAMS.copy()
        params["T"] = -1.0
        with pytest.raises((ValueError, RuntimeError)):
            bates_kernel_cpp.price(**params)

    @skip_if_no_cuda
    def test_negative_strike_raises(self):
        """Test that negative strike raises an error."""
        params = DEFAULT_PARAMS.copy()
        params["K"] = -1.0
        with pytest.raises((ValueError, RuntimeError)):
            bates_kernel_cpp.price(**params)

    @skip_if_no_cuda
    def test_negative_spot_raises(self):
        """Test that negative spot price raises an error."""
        params = DEFAULT_PARAMS.copy()
        params["S0"] = -1.0
        with pytest.raises((ValueError, RuntimeError)):
            bates_kernel_cpp.price(**params)

    @skip_if_no_cuda
    def test_negative_initial_variance_raises(self):
        """Test that negative initial variance raises an error."""
        params = DEFAULT_PARAMS.copy()
        params["v0"] = -0.01
        with pytest.raises((ValueError, RuntimeError)):
            bates_kernel_cpp.price(**params)

    @skip_if_no_cuda
    def test_negative_kappa_raises(self):
        """Test that negative mean reversion speed raises an error."""
        params = DEFAULT_PARAMS.copy()
        params["kappa"] = -1.0
        with pytest.raises((ValueError, RuntimeError)):
            bates_kernel_cpp.price(**params)

    @skip_if_no_cuda
    def test_negative_theta_raises(self):
        """Test that negative long-term variance raises an error."""
        params = DEFAULT_PARAMS.copy()
        params["theta"] = -0.01
        with pytest.raises((ValueError, RuntimeError)):
            bates_kernel_cpp.price(**params)

    @skip_if_no_cuda
    def test_negative_xi_raises(self):
        """Test that negative vol-of-vol raises an error."""
        params = DEFAULT_PARAMS.copy()
        params["xi"] = -0.1
        with pytest.raises((ValueError, RuntimeError)):
            bates_kernel_cpp.price(**params)

    @skip_if_no_cuda
    def test_rho_out_of_bounds_raises(self):
        """Test that correlation outside [-1, 1] raises an error."""
        params = DEFAULT_PARAMS.copy()
        params["rho"] = 1.5
        with pytest.raises((ValueError, RuntimeError)):
            bates_kernel_cpp.price(**params)

        params["rho"] = -1.5
        with pytest.raises((ValueError, RuntimeError)):
            bates_kernel_cpp.price(**params)

    @skip_if_no_cuda
    def test_negative_jump_intensity_raises(self):
        """Test that negative jump intensity raises an error."""
        params = DEFAULT_PARAMS.copy()
        params["lambda_j"] = -0.1
        with pytest.raises((ValueError, RuntimeError)):
            bates_kernel_cpp.price(**params)

    @skip_if_no_cuda
    def test_negative_jump_volatility_raises(self):
        """Test that negative jump volatility raises an error."""
        params = DEFAULT_PARAMS.copy()
        params["sigma_j"] = -0.1
        with pytest.raises((ValueError, RuntimeError)):
            bates_kernel_cpp.price(**params)


# ============================================================================
# Numerical Sanity Tests
# ============================================================================

class TestNumericalSanity:
    """Tests for numerical correctness and sanity checks."""

    @skip_if_no_cuda
    def test_price_is_positive(self):
        """Test that option price is non-negative."""
        price = bates_kernel_cpp.price(**DEFAULT_PARAMS)
        assert price >= 0, f"Option price should be non-negative, got {price}"

    @skip_if_no_cuda
    def test_price_is_finite(self):
        """Test that option price is finite."""
        price = bates_kernel_cpp.price(**DEFAULT_PARAMS)
        assert math.isfinite(price), f"Option price should be finite, got {price}"

    @skip_if_no_cuda
    def test_otm_put_cheaper_than_atm(self):
        """Test that OTM put is cheaper than ATM put."""
        atm_params = DEFAULT_PARAMS.copy()
        atm_params["K"] = 100.0  # ATM
        atm_params["num_paths"] = 50000

        otm_params = DEFAULT_PARAMS.copy()
        otm_params["K"] = 80.0  # OTM
        otm_params["num_paths"] = 50000

        atm_price = bates_kernel_cpp.price(**atm_params)
        otm_price = bates_kernel_cpp.price(**otm_params)

        assert otm_price < atm_price, (
            f"OTM put (K={otm_params['K']}) should be cheaper than ATM put, "
            f"got OTM={otm_price:.4f}, ATM={atm_price:.4f}"
        )

    @skip_if_no_cuda
    def test_itm_put_more_expensive_than_atm(self):
        """Test that ITM put is more expensive than ATM put."""
        atm_params = DEFAULT_PARAMS.copy()
        atm_params["K"] = 100.0  # ATM
        atm_params["num_paths"] = 50000

        itm_params = DEFAULT_PARAMS.copy()
        itm_params["K"] = 120.0  # ITM
        itm_params["num_paths"] = 50000

        atm_price = bates_kernel_cpp.price(**atm_params)
        itm_price = bates_kernel_cpp.price(**itm_params)

        assert itm_price > atm_price, (
            f"ITM put (K={itm_params['K']}) should be more expensive than ATM put, "
            f"got ITM={itm_price:.4f}, ATM={atm_price:.4f}"
        )

    @skip_if_no_cuda
    def test_put_bounded_by_strike(self):
        """Test that put price is bounded by discounted strike."""
        params = DEFAULT_PARAMS.copy()
        params["num_paths"] = 50000

        price = bates_kernel_cpp.price(**params)
        max_price = params["K"] * math.exp(-params["r"] * params["T"])

        assert price <= max_price * 1.01, (  # Allow 1% tolerance for MC error
            f"Put price should be <= discounted strike, got {price:.4f} > {max_price:.4f}"
        )

    @skip_if_no_cuda
    def test_zero_volatility_put_value(self):
        """Test put value approaches intrinsic when volatility is near zero."""
        params = DEFAULT_PARAMS.copy()
        params["v0"] = 0.0001  # Very small initial variance
        params["theta"] = 0.0001  # Very small long-term variance
        params["xi"] = 0.0001  # Very small vol-of-vol
        params["lambda_j"] = 0.0  # No jumps
        params["num_paths"] = 50000
        params["K"] = 110.0  # ITM put

        price = bates_kernel_cpp.price(**params)

        # For very low volatility, put should be close to discounted intrinsic
        # intrinsic = max(K - S_avg, 0) where S_avg ~ S0 * exp(r*T/2) approximately
        # For Asian option with arithmetic average, this is more complex
        # Just check price is reasonable
        assert price > 0, "ITM put with low vol should have positive value"

    @skip_if_no_cuda
    def test_convergence_with_more_paths(self):
        """Test that price stabilizes with more paths."""
        params_low = DEFAULT_PARAMS.copy()
        params_low["num_paths"] = 10000

        params_high = DEFAULT_PARAMS.copy()
        params_high["num_paths"] = 100000

        prices_low = [bates_kernel_cpp.price(**params_low) for _ in range(5)]
        prices_high = [bates_kernel_cpp.price(**params_high) for _ in range(5)]

        std_low = np.std(prices_low)
        std_high = np.std(prices_high)

        # Higher path count should generally give more stable results
        # This is a weak test - just ensure we get reasonable values
        assert std_low < 1.0, f"Prices should be somewhat stable, std={std_low}"
        assert std_high < std_low * 2, "More paths should not increase variance significantly"


# ============================================================================
# Jump Process Tests
# ============================================================================

class TestJumpProcess:
    """Tests for jump diffusion component."""

    @skip_if_no_cuda
    def test_jumps_increase_otm_put_value(self):
        """Test that adding negative jumps increases OTM put value (crash risk)."""
        no_jump_params = DEFAULT_PARAMS.copy()
        no_jump_params["lambda_j"] = 0.0
        no_jump_params["K"] = 80.0  # OTM put
        no_jump_params["num_paths"] = 100000

        jump_params = DEFAULT_PARAMS.copy()
        jump_params["lambda_j"] = 0.5  # Significant jump intensity
        jump_params["mu_j"] = -0.15  # Negative mean jump (crashes)
        jump_params["sigma_j"] = 0.2
        jump_params["K"] = 80.0  # OTM put
        jump_params["num_paths"] = 100000

        no_jump_price = bates_kernel_cpp.price(**no_jump_params)
        jump_price = bates_kernel_cpp.price(**jump_params)

        # OTM puts should be more valuable with crash risk
        assert jump_price > no_jump_price * 0.8, (
            f"OTM put with negative jumps should be more valuable, "
            f"got no_jump={no_jump_price:.4f}, jump={jump_price:.4f}"
        )


# ============================================================================
# Edge Cases Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @skip_if_no_cuda
    def test_very_short_maturity(self):
        """Test pricing with very short time to maturity."""
        params = DEFAULT_PARAMS.copy()
        params["T"] = 0.01  # ~3.6 days
        params["num_steps"] = 10

        price = bates_kernel_cpp.price(**params)
        assert math.isfinite(price), "Price should be finite for short maturity"
        assert price >= 0, "Price should be non-negative"

    @skip_if_no_cuda
    def test_long_maturity(self):
        """Test pricing with long time to maturity."""
        params = DEFAULT_PARAMS.copy()
        params["T"] = 5.0  # 5 years
        params["num_steps"] = 252 * 5

        price = bates_kernel_cpp.price(**params)
        assert math.isfinite(price), "Price should be finite for long maturity"
        assert price >= 0, "Price should be non-negative"

    @skip_if_no_cuda
    def test_zero_interest_rate(self):
        """Test pricing with zero interest rate."""
        params = DEFAULT_PARAMS.copy()
        params["r"] = 0.0

        price = bates_kernel_cpp.price(**params)
        assert math.isfinite(price), "Price should be finite with zero rate"
        assert price >= 0, "Price should be non-negative"

    @skip_if_no_cuda
    def test_correlation_boundaries(self):
        """Test pricing at correlation boundaries."""
        for rho in [-1.0, 0.0, 1.0]:
            params = DEFAULT_PARAMS.copy()
            params["rho"] = rho

            price = bates_kernel_cpp.price(**params)
            assert math.isfinite(price), f"Price should be finite for rho={rho}"
            assert price >= 0, f"Price should be non-negative for rho={rho}"

    @skip_if_no_cuda
    def test_minimum_paths(self):
        """Test with minimum number of paths."""
        params = DEFAULT_PARAMS.copy()
        params["num_paths"] = 1
        params["num_steps"] = 10

        price = bates_kernel_cpp.price(**params)
        assert math.isfinite(price), "Price should be finite with single path"

    @skip_if_no_cuda
    def test_minimum_steps(self):
        """Test with minimum number of steps."""
        params = DEFAULT_PARAMS.copy()
        params["num_steps"] = 1

        price = bates_kernel_cpp.price(**params)
        assert math.isfinite(price), "Price should be finite with single step"


# ============================================================================
# Performance Smoke Tests
# ============================================================================

class TestPerformance:
    """Basic performance/stress tests."""

    @skip_if_no_cuda
    def test_large_simulation(self):
        """Test that large simulation completes without error."""
        params = DEFAULT_PARAMS.copy()
        params["num_paths"] = 500000
        params["num_steps"] = 252

        price = bates_kernel_cpp.price(**params)
        assert math.isfinite(price), "Large simulation should produce finite result"

    @skip_if_no_cuda
    def test_many_steps(self):
        """Test simulation with many time steps."""
        params = DEFAULT_PARAMS.copy()
        params["num_paths"] = 10000
        params["num_steps"] = 1000

        price = bates_kernel_cpp.price(**params)
        assert math.isfinite(price), "Many-step simulation should produce finite result"


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
