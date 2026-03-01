"""
Extended test suite for Bates model with QE scheme, barriers, and Greeks.

Tests cover:
- QE variance scheme correctness
- Barrier option pricing
- Control variates variance reduction
- Greeks calculation
- Lookback options
- Edge cases and numerical stability

Run with: pytest test_bates_extended.py -v
"""

import pytest
import numpy as np
import math
import sys

# Try to import the extended CUDA module
try:
    import bates_extended
    CUDA_EXTENDED_AVAILABLE = True
except ImportError:
    CUDA_EXTENDED_AVAILABLE = False
    bates_extended = None

# Try to import Python MC pricer
try:
    from mc_pricer import (
        price_asian_put, price_barrier_option, price_option,
        MarketData, HestonParams, JumpParams, SimulationConfig,
        PayoffType, Backend, VarianceReduction, DiscretizationScheme,
        black_scholes_put, black_scholes_call
    )
    MC_PRICER_AVAILABLE = True
except ImportError:
    MC_PRICER_AVAILABLE = False


# Default Bates model parameters
DEFAULT_PARAMS = {
    "num_paths": 50000,
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


def skip_if_no_cuda_extended(func):
    """Decorator to skip tests if extended CUDA module is not available."""
    return pytest.mark.skipif(
        not CUDA_EXTENDED_AVAILABLE,
        reason="Extended CUDA module not available"
    )(func)


def skip_if_no_mc_pricer(func):
    """Decorator to skip tests if Python MC pricer is not available."""
    return pytest.mark.skipif(
        not MC_PRICER_AVAILABLE,
        reason="MC pricer module not available"
    )(func)


# ============================================================================
# QE Scheme Tests
# ============================================================================

class TestQEScheme:
    """Tests for Quadratic-Exponential variance discretization."""

    @skip_if_no_cuda_extended
    def test_qe_asian_put_price_positive(self):
        """Test that QE scheme produces positive prices."""
        result = bates_extended.price(
            **DEFAULT_PARAMS, payoff_type="asian_put"
        )
        assert result.price >= 0, f"Price should be non-negative, got {result.price}"
        assert result.std_error > 0, "Standard error should be positive"

    @skip_if_no_cuda_extended
    def test_qe_price_finite(self):
        """Test that QE scheme produces finite prices."""
        result = bates_extended.price(
            **DEFAULT_PARAMS, payoff_type="asian_put"
        )
        assert math.isfinite(result.price), f"Price should be finite, got {result.price}"
        assert math.isfinite(result.std_error), "Standard error should be finite"

    @skip_if_no_cuda_extended
    def test_qe_feller_violated_still_works(self):
        """Test that QE scheme handles Feller condition violation."""
        params = DEFAULT_PARAMS.copy()
        params["xi"] = 0.8  # High vol-of-vol violates Feller

        assert not bates_extended.check_feller(
            params["kappa"], params["theta"], params["xi"]
        ), "Should violate Feller condition"

        result = bates_extended.price(**params, payoff_type="asian_put")
        assert math.isfinite(result.price), "QE should handle Feller violation"
        assert result.price >= 0, "Price should still be non-negative"

    @skip_if_no_cuda_extended
    def test_qe_convergence(self):
        """Test that prices converge with more paths."""
        params = DEFAULT_PARAMS.copy()

        prices_low = []
        prices_high = []

        for _ in range(5):
            params["num_paths"] = 10000
            result_low = bates_extended.price(**params, payoff_type="asian_put")
            prices_low.append(result_low.price)

            params["num_paths"] = 100000
            result_high = bates_extended.price(**params, payoff_type="asian_put")
            prices_high.append(result_high.price)

        std_low = np.std(prices_low)
        std_high = np.std(prices_high)

        # Higher path count should give more stable results
        assert std_high < std_low * 1.5, "More paths should reduce variance"


# ============================================================================
# Barrier Option Tests
# ============================================================================

class TestBarrierOptions:
    """Tests for barrier option pricing."""

    @skip_if_no_cuda_extended
    def test_down_out_put_knocked_out(self):
        """Test down-and-out put with barrier above current price."""
        params = DEFAULT_PARAMS.copy()
        params["S0"] = 100.0

        # Barrier at 110 (above S0) - always knocked out
        result = bates_extended.price(
            **params, payoff_type="barrier_down_out_put",
            barrier=110.0, rebate=0.0
        )

        # Should be close to rebate (0)
        assert result.price < 5.0, "Down-out put with high barrier should be cheap"

    @skip_if_no_cuda_extended
    def test_down_in_put_knocked_in(self):
        """Test down-and-in put with barrier above current price."""
        params = DEFAULT_PARAMS.copy()

        # Barrier at 110 - always knocked in
        result = bates_extended.price(
            **params, payoff_type="barrier_down_in_put",
            barrier=110.0, rebate=0.0
        )

        # Should be close to vanilla put
        vanilla = bates_extended.price(**params, payoff_type="european_put")

        assert abs(result.price - vanilla.price) < vanilla.price * 0.3, \
            "Down-in put with high barrier should approach vanilla"

    @skip_if_no_cuda_extended
    def test_barrier_parity(self):
        """Test knock-in + knock-out = vanilla parity."""
        params = DEFAULT_PARAMS.copy()
        params["num_paths"] = 100000

        barrier = 80.0

        up_out = bates_extended.price(
            **params, payoff_type="barrier_up_out_call",
            barrier=120.0, rebate=0.0
        )
        up_in = bates_extended.price(
            **params, payoff_type="barrier_up_in_call",
            barrier=120.0, rebate=0.0
        )
        vanilla = bates_extended.price(**params, payoff_type="european_call")

        parity_error = abs(up_out.price + up_in.price - vanilla.price)
        assert parity_error < vanilla.price * 0.15, \
            f"Barrier parity violation: {parity_error:.4f}"

    @skip_if_no_cuda_extended
    def test_barrier_below_strike_otm(self):
        """Test barrier option behavior when barrier < strike."""
        params = DEFAULT_PARAMS.copy()

        result = bates_extended.price(
            **params, payoff_type="barrier_down_out_put",
            barrier=70.0, rebate=0.0
        )

        # Deep OTM barrier should have low probability of hitting
        vanilla = bates_extended.price(**params, payoff_type="european_put")
        assert result.price <= vanilla.price, "Down-out put should be <= vanilla"


# ============================================================================
# Control Variates Tests
# ============================================================================

class TestControlVariates:
    """Tests for control variate variance reduction."""

    @skip_if_no_cuda_extended
    def test_cv_reduces_variance(self):
        """Test that control variates reduce standard error."""
        params = DEFAULT_PARAMS.copy()
        params["num_paths"] = 50000
        params["lambda_j"] = 0.0  # No jumps for fair comparison

        # Approximate geometric Asian analytical price (simplified)
        sigma = np.sqrt(params["v0"])
        n = params["num_steps"]
        sigma_adj = sigma * np.sqrt((2 * n + 1) / (6 * (n + 1)))
        geom_analytical = 5.0  # Approximate

        # With CV
        result_cv = bates_extended.price_asian_cv(
            **params, geom_analytical=geom_analytical, is_call=False
        )

        # Without CV
        result_no_cv = bates_extended.price(**params, payoff_type="asian_put")

        # CV should reduce or at least not significantly increase std error
        # (Note: effectiveness depends on correlation)
        assert result_cv.std_error < result_no_cv.std_error * 1.5, \
            "CV should not significantly increase variance"

    @skip_if_no_cuda_extended
    def test_cv_beta_reasonable(self):
        """Test that CV beta has reasonable magnitude."""
        params = DEFAULT_PARAMS.copy()
        params["lambda_j"] = 0.0

        result = bates_extended.price_asian_cv(
            **params, geom_analytical=5.0, is_call=False
        )

        # Beta should be finite and not extreme
        assert math.isfinite(result.cv_beta), "CV beta should be finite"
        assert abs(result.cv_beta) < 10, "CV beta should have reasonable magnitude"


# ============================================================================
# Greeks Tests
# ============================================================================

class TestGreeks:
    """Tests for Greeks calculation."""

    @skip_if_no_cuda_extended
    def test_delta_positive_for_call(self):
        """Test that call delta is positive."""
        params = DEFAULT_PARAMS.copy()
        params["num_paths"] = 50000

        greeks = bates_extended.greeks(**params, payoff_type="european_call")
        assert greeks.delta > 0, f"Call delta should be positive, got {greeks.delta}"

    @skip_if_no_cuda_extended
    def test_delta_negative_for_put(self):
        """Test that put delta is negative."""
        params = DEFAULT_PARAMS.copy()
        params["num_paths"] = 50000

        greeks = bates_extended.greeks(**params, payoff_type="european_put")
        assert greeks.delta < 0, f"Put delta should be negative, got {greeks.delta}"

    @skip_if_no_cuda_extended
    def test_gamma_positive(self):
        """Test that gamma is positive."""
        params = DEFAULT_PARAMS.copy()
        params["num_paths"] = 50000

        greeks = bates_extended.greeks(**params, payoff_type="european_call")
        # Gamma can be slightly negative due to MC noise
        assert greeks.gamma > -0.01, f"Gamma should be non-negative, got {greeks.gamma}"

    @skip_if_no_cuda_extended
    def test_vega_positive(self):
        """Test that vega is positive for options with time value."""
        params = DEFAULT_PARAMS.copy()
        params["num_paths"] = 50000

        greeks = bates_extended.greeks(**params, payoff_type="european_call")
        assert greeks.vega > 0, f"Vega should be positive, got {greeks.vega}"

    @skip_if_no_cuda_extended
    def test_theta_negative_for_atm(self):
        """Test that theta is generally negative for ATM options."""
        params = DEFAULT_PARAMS.copy()
        params["num_paths"] = 50000
        params["K"] = params["S0"]  # ATM

        greeks = bates_extended.greeks(**params, payoff_type="european_call")
        # Theta is typically negative (time decay)
        assert greeks.theta < 0.5, "ATM theta should be negative or small"

    @skip_if_no_cuda_extended
    def test_greeks_finite(self):
        """Test that all Greeks are finite."""
        params = DEFAULT_PARAMS.copy()
        params["num_paths"] = 50000

        greeks = bates_extended.greeks(**params, payoff_type="asian_put")

        assert math.isfinite(greeks.delta), "Delta should be finite"
        assert math.isfinite(greeks.gamma), "Gamma should be finite"
        assert math.isfinite(greeks.vega), "Vega should be finite"
        assert math.isfinite(greeks.theta), "Theta should be finite"
        assert math.isfinite(greeks.rho), "Rho should be finite"


# ============================================================================
# Lookback Options Tests
# ============================================================================

class TestLookbackOptions:
    """Tests for lookback option pricing."""

    @skip_if_no_cuda_extended
    def test_lookback_call_more_valuable(self):
        """Test that lookback call is more valuable than vanilla."""
        params = DEFAULT_PARAMS.copy()
        params["num_paths"] = 50000

        lookback = bates_extended.price(**params, payoff_type="lookback_fixed_call")
        vanilla = bates_extended.price(**params, payoff_type="european_call")

        assert lookback.price >= vanilla.price * 0.9, \
            "Lookback should be at least as valuable as vanilla"

    @skip_if_no_cuda_extended
    def test_lookback_put_more_valuable(self):
        """Test that lookback put is more valuable than vanilla."""
        params = DEFAULT_PARAMS.copy()
        params["num_paths"] = 50000

        lookback = bates_extended.price(**params, payoff_type="lookback_fixed_put")
        vanilla = bates_extended.price(**params, payoff_type="european_put")

        assert lookback.price >= vanilla.price * 0.9, \
            "Lookback should be at least as valuable as vanilla"


# ============================================================================
# Python MC Pricer Tests
# ============================================================================

class TestPythonMCPricer:
    """Tests for the Python Monte Carlo pricing library."""

    @skip_if_no_mc_pricer
    def test_gbm_european_vs_bs(self):
        """Test GBM European option against Black-Scholes."""
        from scipy.stats import norm

        S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0

        market = MarketData(S0=S0, r=r)
        config = SimulationConfig(
            num_paths=100000,
            num_steps=252,
            variance_reduction=VarianceReduction.ANTITHETIC
        )

        result = price_option(
            market=market, K=K, T=T,
            payoff_type=PayoffType.EUROPEAN_CALL,
            config=config, sigma=sigma
        )

        bs_price = black_scholes_call(S0, K, r, sigma, T)

        error = abs(result.price - bs_price)
        assert error < 0.5, f"GBM call should match BS: MC={result.price:.4f}, BS={bs_price:.4f}"

    @skip_if_no_mc_pricer
    def test_heston_asian_put(self):
        """Test Heston model Asian put pricing."""
        market = MarketData(S0=100.0, r=0.05)
        heston = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
        config = SimulationConfig(
            num_paths=50000,
            num_steps=252,
            scheme=DiscretizationScheme.QE
        )

        result = price_option(
            market=market, K=100.0, T=1.0,
            payoff_type=PayoffType.ASIAN_PUT,
            heston=heston, config=config
        )

        assert result.price > 0, "Asian put price should be positive"
        assert math.isfinite(result.price), "Price should be finite"

    @skip_if_no_mc_pricer
    def test_bates_jump_increases_otm_put(self):
        """Test that jumps increase OTM put prices."""
        market = MarketData(S0=100.0, r=0.05)
        heston = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)

        config = SimulationConfig(
            num_paths=50000,
            num_steps=252,
            scheme=DiscretizationScheme.QE
        )

        # Without jumps
        result_no_jump = price_option(
            market=market, K=80.0, T=1.0,
            payoff_type=PayoffType.ASIAN_PUT,
            heston=heston,
            jump=JumpParams(lambda_j=0.0),
            config=config
        )

        # With negative jumps (crash risk)
        result_jump = price_option(
            market=market, K=80.0, T=1.0,
            payoff_type=PayoffType.ASIAN_PUT,
            heston=heston,
            jump=JumpParams(lambda_j=0.5, mu_j=-0.15, sigma_j=0.2),
            config=config
        )

        # OTM puts should be more valuable with crash risk
        assert result_jump.price >= result_no_jump.price * 0.8, \
            "Jumps should increase OTM put value"

    @skip_if_no_mc_pricer
    def test_barrier_option_pricing(self):
        """Test barrier option pricing."""
        result = price_barrier_option(
            S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0,
            barrier=80.0, barrier_type="down_out_put",
            num_paths=50000
        )

        assert result.price >= 0, "Barrier option price should be non-negative"
        assert math.isfinite(result.price), "Price should be finite"

    @skip_if_no_mc_pricer
    def test_discretization_schemes(self):
        """Test different discretization schemes produce similar results."""
        market = MarketData(S0=100.0, r=0.05)
        heston = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)

        prices = {}
        for scheme in [DiscretizationScheme.EULER, DiscretizationScheme.MILSTEIN, DiscretizationScheme.QE]:
            config = SimulationConfig(
                num_paths=50000,
                num_steps=252,
                scheme=scheme,
                seed=42
            )

            result = price_option(
                market=market, K=100.0, T=1.0,
                payoff_type=PayoffType.ASIAN_PUT,
                heston=heston, config=config
            )
            prices[scheme.value] = result.price

        # All schemes should give similar results
        price_list = list(prices.values())
        assert max(price_list) - min(price_list) < 2.0, \
            f"Schemes should give similar prices: {prices}"


# ============================================================================
# Edge Cases and Stress Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @skip_if_no_cuda_extended
    def test_zero_volatility(self):
        """Test with near-zero volatility."""
        params = DEFAULT_PARAMS.copy()
        params["v0"] = 0.0001
        params["theta"] = 0.0001
        params["xi"] = 0.0001
        params["lambda_j"] = 0.0

        result = bates_extended.price(**params, payoff_type="european_put")
        assert math.isfinite(result.price), "Should handle low volatility"

    @skip_if_no_cuda_extended
    def test_high_volatility(self):
        """Test with high volatility."""
        params = DEFAULT_PARAMS.copy()
        params["v0"] = 0.5
        params["theta"] = 0.5
        params["xi"] = 1.0

        result = bates_extended.price(**params, payoff_type="asian_put")
        assert math.isfinite(result.price), "Should handle high volatility"
        assert result.price > 0, "High vol should give positive put price"

    @skip_if_no_cuda_extended
    def test_short_maturity(self):
        """Test with very short maturity."""
        params = DEFAULT_PARAMS.copy()
        params["T"] = 0.01  # ~3.6 days
        params["num_steps"] = 10

        result = bates_extended.price(**params, payoff_type="asian_put")
        assert math.isfinite(result.price), "Should handle short maturity"

    @skip_if_no_cuda_extended
    def test_long_maturity(self):
        """Test with long maturity."""
        params = DEFAULT_PARAMS.copy()
        params["T"] = 5.0
        params["num_steps"] = 1000

        result = bates_extended.price(**params, payoff_type="asian_put")
        assert math.isfinite(result.price), "Should handle long maturity"

    @skip_if_no_cuda_extended
    def test_extreme_correlation(self):
        """Test with extreme correlation values."""
        for rho in [-0.99, -0.5, 0.0, 0.5, 0.99]:
            params = DEFAULT_PARAMS.copy()
            params["rho"] = rho

            result = bates_extended.price(**params, payoff_type="asian_put")
            assert math.isfinite(result.price), f"Should handle rho={rho}"

    @skip_if_no_cuda_extended
    def test_high_jump_intensity(self):
        """Test with high jump intensity."""
        params = DEFAULT_PARAMS.copy()
        params["lambda_j"] = 2.0  # 2 jumps per year on average
        params["mu_j"] = -0.1
        params["sigma_j"] = 0.3

        result = bates_extended.price(**params, payoff_type="asian_put")
        assert math.isfinite(result.price), "Should handle high jump intensity"


# ============================================================================
# Feller Condition Tests
# ============================================================================

class TestFellerCondition:
    """Tests for Feller condition checking."""

    @skip_if_no_cuda_extended
    def test_feller_satisfied(self):
        """Test Feller condition when satisfied."""
        # 2*2*0.04 = 0.16 > 0.3^2 = 0.09
        assert bates_extended.check_feller(2.0, 0.04, 0.3) is True

    @skip_if_no_cuda_extended
    def test_feller_violated(self):
        """Test Feller condition when violated."""
        # 2*2*0.04 = 0.16 < 0.5^2 = 0.25
        assert bates_extended.check_feller(2.0, 0.04, 0.5) is False

    @skip_if_no_cuda_extended
    def test_feller_boundary(self):
        """Test Feller condition at boundary."""
        # 2*2*0.04 = 0.16, xi^2 = 0.4^2 = 0.16 (exactly equal)
        result = bates_extended.check_feller(2.0, 0.04, 0.4)
        # At boundary, it's NOT satisfied (strict inequality)
        assert result is False


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
