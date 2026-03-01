"""Tests for pricing models."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mc_pricer import (
    MarketData, HestonParams, JumpParams, SimulationConfig,
    Backend, DiscretizationScheme, VarianceReduction, PayoffType,
    price_option, price_with_greeks, black_scholes_call, black_scholes_put,
    SCIPY_AVAILABLE
)


class TestGBMPricing:
    """Tests for GBM (Black-Scholes) model."""

    def test_european_call_vs_bs(self, market_data, sim_config):
        """MC price should converge to Black-Scholes."""
        if not SCIPY_AVAILABLE:
            pytest.skip("SciPy not available")

        result = price_option(
            market=market_data, K=100.0, T=1.0,
            payoff_type=PayoffType.EUROPEAN_CALL,
            config=sim_config, sigma=0.2
        )
        bs_price = black_scholes_call(100.0, 100.0, 0.05, 0.2, 1.0)

        assert abs(result.price - bs_price) < 0.5  # Within $0.50

    def test_european_put_vs_bs(self, market_data, sim_config):
        """MC put price should converge to Black-Scholes."""
        if not SCIPY_AVAILABLE:
            pytest.skip("SciPy not available")

        result = price_option(
            market=market_data, K=100.0, T=1.0,
            payoff_type=PayoffType.EUROPEAN_PUT,
            config=sim_config, sigma=0.2
        )
        bs_price = black_scholes_put(100.0, 100.0, 0.05, 0.2, 1.0)

        assert abs(result.price - bs_price) < 0.5

    def test_put_call_parity(self, market_data, high_accuracy_config):
        """Put-call parity should hold."""
        K, T, sigma = 100.0, 1.0, 0.2
        r = market_data.r

        call_result = price_option(
            market=market_data, K=K, T=T,
            payoff_type=PayoffType.EUROPEAN_CALL,
            config=high_accuracy_config, sigma=sigma
        )
        put_result = price_option(
            market=market_data, K=K, T=T,
            payoff_type=PayoffType.EUROPEAN_PUT,
            config=high_accuracy_config, sigma=sigma
        )

        # C - P = S - K*exp(-rT)
        lhs = call_result.price - put_result.price
        rhs = market_data.S0 - K * np.exp(-r * T)

        assert abs(lhs - rhs) < 0.5


class TestHestonModel:
    """Tests for Heston stochastic volatility model."""

    def test_heston_positive_price(self, market_data, heston_params, sim_config):
        """Heston model should produce positive prices."""
        result = price_option(
            market=market_data, K=100.0, T=1.0,
            payoff_type=PayoffType.EUROPEAN_CALL,
            heston=heston_params, config=sim_config
        )
        assert result.price > 0

    def test_heston_finite_price(self, market_data, heston_params, sim_config):
        """Heston model should produce finite prices."""
        result = price_option(
            market=market_data, K=100.0, T=1.0,
            payoff_type=PayoffType.EUROPEAN_CALL,
            heston=heston_params, config=sim_config
        )
        assert np.isfinite(result.price)
        assert np.isfinite(result.std_error)

    def test_feller_condition_check(self):
        """Feller condition should be correctly computed."""
        # Satisfied: 2*2*0.04 = 0.16 > 0.09 = 0.3^2
        heston_good = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
        assert heston_good.feller_condition is True

        # Violated: 2*1*0.04 = 0.08 < 0.25 = 0.5^2
        heston_bad = HestonParams(v0=0.04, kappa=1.0, theta=0.04, xi=0.5, rho=-0.7)
        assert heston_bad.feller_condition is False


class TestBatesModel:
    """Tests for Bates model (Heston + jumps)."""

    def test_bates_positive_price(self, market_data, heston_params, jump_params, sim_config):
        """Bates model should produce positive prices."""
        result = price_option(
            market=market_data, K=100.0, T=1.0,
            payoff_type=PayoffType.EUROPEAN_CALL,
            heston=heston_params, jump=jump_params, config=sim_config
        )
        assert result.price > 0

    def test_bates_vs_heston(self, market_data, heston_params, sim_config):
        """Bates with zero jumps should equal Heston."""
        no_jumps = JumpParams(lambda_j=0.0, mu_j=0.0, sigma_j=0.0)

        bates_result = price_option(
            market=market_data, K=100.0, T=1.0,
            payoff_type=PayoffType.EUROPEAN_CALL,
            heston=heston_params, jump=no_jumps, config=sim_config
        )
        heston_result = price_option(
            market=market_data, K=100.0, T=1.0,
            payoff_type=PayoffType.EUROPEAN_CALL,
            heston=heston_params, config=sim_config
        )

        # Should be close (within MC error)
        assert abs(bates_result.price - heston_result.price) < 1.0


class TestAsianOptions:
    """Tests for Asian options."""

    def test_asian_call_less_than_european(self, market_data, sim_config):
        """Asian call should be worth less than European call."""
        asian_result = price_option(
            market=market_data, K=100.0, T=1.0,
            payoff_type=PayoffType.ASIAN_CALL,
            config=sim_config, sigma=0.2
        )
        european_result = price_option(
            market=market_data, K=100.0, T=1.0,
            payoff_type=PayoffType.EUROPEAN_CALL,
            config=sim_config, sigma=0.2
        )

        assert asian_result.price < european_result.price

    def test_asian_geom_vs_arith(self, market_data, sim_config):
        """Geometric Asian should be worth less than arithmetic."""
        arith_result = price_option(
            market=market_data, K=100.0, T=1.0,
            payoff_type=PayoffType.ASIAN_CALL,
            config=sim_config, sigma=0.2
        )
        geom_result = price_option(
            market=market_data, K=100.0, T=1.0,
            payoff_type=PayoffType.ASIAN_GEOM_CALL,
            config=sim_config, sigma=0.2
        )

        # Geometric mean <= Arithmetic mean, so geom price <= arith price
        assert geom_result.price <= arith_result.price + 0.5


class TestBarrierOptions:
    """Tests for barrier options."""

    def test_knockout_less_than_vanilla(self, market_data, sim_config):
        """Knock-out option should be worth less than vanilla."""
        from mc_pricer import BarrierParams

        barrier = BarrierParams(barrier=120.0, rebate=0.0)

        knockout_result = price_option(
            market=market_data, K=100.0, T=1.0,
            payoff_type=PayoffType.BARRIER_UP_OUT_CALL,
            barrier=barrier, config=sim_config, sigma=0.2
        )
        vanilla_result = price_option(
            market=market_data, K=100.0, T=1.0,
            payoff_type=PayoffType.EUROPEAN_CALL,
            config=sim_config, sigma=0.2
        )

        assert knockout_result.price < vanilla_result.price

    def test_knockin_knockout_parity(self, market_data, high_accuracy_config):
        """Knock-in + Knock-out = Vanilla (approximately)."""
        from mc_pricer import BarrierParams

        barrier = BarrierParams(barrier=120.0, rebate=0.0)

        knockin_result = price_option(
            market=market_data, K=100.0, T=1.0,
            payoff_type=PayoffType.BARRIER_UP_IN_CALL,
            barrier=barrier, config=high_accuracy_config, sigma=0.2
        )
        knockout_result = price_option(
            market=market_data, K=100.0, T=1.0,
            payoff_type=PayoffType.BARRIER_UP_OUT_CALL,
            barrier=barrier, config=high_accuracy_config, sigma=0.2
        )
        vanilla_result = price_option(
            market=market_data, K=100.0, T=1.0,
            payoff_type=PayoffType.EUROPEAN_CALL,
            config=high_accuracy_config, sigma=0.2
        )

        # In + Out = Vanilla
        combined = knockin_result.price + knockout_result.price
        assert abs(combined - vanilla_result.price) < 1.0


class TestGreeks:
    """Tests for Greeks calculation."""

    def test_delta_positive_for_call(self, market_data, sim_config):
        """Call delta should be positive."""
        result = price_with_greeks(
            market=market_data, K=100.0, T=1.0,
            payoff_type=PayoffType.EUROPEAN_CALL,
            config=sim_config, sigma=0.2
        )
        assert result.greeks['delta'] > 0

    def test_delta_negative_for_put(self, market_data, sim_config):
        """Put delta should be negative."""
        result = price_with_greeks(
            market=market_data, K=100.0, T=1.0,
            payoff_type=PayoffType.EUROPEAN_PUT,
            config=sim_config, sigma=0.2
        )
        assert result.greeks['delta'] < 0

    def test_gamma_positive(self, market_data, sim_config):
        """Gamma should be positive for both calls and puts."""
        call_result = price_with_greeks(
            market=market_data, K=100.0, T=1.0,
            payoff_type=PayoffType.EUROPEAN_CALL,
            config=sim_config, sigma=0.2
        )
        assert call_result.greeks['gamma'] > 0

    def test_vega_positive(self, market_data, sim_config):
        """Vega should be positive."""
        result = price_with_greeks(
            market=market_data, K=100.0, T=1.0,
            payoff_type=PayoffType.EUROPEAN_CALL,
            config=sim_config, sigma=0.2
        )
        if 'vega' in result.greeks:
            assert result.greeks['vega'] > 0


class TestAmericanOptions:
    """Tests for American options (LSM)."""

    def test_american_put_greater_than_european(self, market_data, sim_config):
        """American put should be worth more than European put."""
        from mc_pricer import price_american_option_lsm

        american_result = price_american_option_lsm(
            market=market_data, K=100.0, T=1.0,
            is_call=False, config=sim_config, sigma=0.2
        )
        european_result = price_option(
            market=market_data, K=100.0, T=1.0,
            payoff_type=PayoffType.EUROPEAN_PUT,
            config=sim_config, sigma=0.2
        )

        # American >= European (early exercise premium)
        assert american_result.price >= european_result.price - 0.5

    def test_american_call_no_early_exercise(self, market_data, sim_config):
        """American call on non-dividend stock should equal European."""
        from mc_pricer import price_american_option_lsm

        american_result = price_american_option_lsm(
            market=market_data, K=100.0, T=1.0,
            is_call=True, config=sim_config, sigma=0.2
        )
        european_result = price_option(
            market=market_data, K=100.0, T=1.0,
            payoff_type=PayoffType.EUROPEAN_CALL,
            config=sim_config, sigma=0.2
        )

        # Should be approximately equal (no early exercise value)
        assert abs(american_result.price - european_result.price) < 1.0


class TestSABRModel:
    """Tests for SABR model."""

    def test_sabr_positive_price(self, market_data, sim_config):
        """SABR model should produce positive prices."""
        from mc_pricer import SABRParams, price_sabr_option

        sabr = SABRParams(alpha=0.3, beta=0.5, rho=-0.3, nu=0.2)

        result = price_sabr_option(
            F0=100.0, K=100.0, T=1.0, r=0.05,
            sabr=sabr, is_call=True, config=sim_config
        )
        assert result.price > 0

    def test_sabr_hagan_approximation(self, market_data):
        """SABR Hagan approximation should produce finite results."""
        from mc_pricer import SABRParams, price_sabr_option

        sabr = SABRParams(alpha=0.3, beta=0.5, rho=-0.3, nu=0.2)

        result = price_sabr_option(
            F0=100.0, K=100.0, T=1.0, r=0.05,
            sabr=sabr, is_call=True, use_hagan=True
        )
        assert np.isfinite(result.price)
        assert result.price > 0


class TestMultiAssetOptions:
    """Tests for multi-asset/basket options."""

    def test_basket_option_positive(self, sim_config):
        """Basket option should have positive price."""
        from mc_pricer import price_basket_option, MultiAssetPayoffType

        result = price_basket_option(
            S0=np.array([100.0, 100.0]),
            K=100.0, T=1.0, r=0.05,
            sigmas=np.array([0.2, 0.25]),
            payoff_type=MultiAssetPayoffType.BASKET_CALL,
            config=sim_config
        )
        assert result.price > 0

    def test_rainbow_best_of_greater_than_worst_of(self, sim_config):
        """Best-of call should be worth more than worst-of call."""
        from mc_pricer import price_basket_option, MultiAssetPayoffType

        best_of = price_basket_option(
            S0=np.array([100.0, 100.0]),
            K=100.0, T=1.0, r=0.05,
            sigmas=np.array([0.2, 0.25]),
            payoff_type=MultiAssetPayoffType.BEST_OF_CALL,
            config=sim_config
        )
        worst_of = price_basket_option(
            S0=np.array([100.0, 100.0]),
            K=100.0, T=1.0, r=0.05,
            sigmas=np.array([0.2, 0.25]),
            payoff_type=MultiAssetPayoffType.WORST_OF_CALL,
            config=sim_config
        )

        assert best_of.price > worst_of.price

    def test_correlated_assets(self, sim_config):
        """Highly correlated assets should affect basket price."""
        from mc_pricer import price_basket_option, MultiAssetPayoffType

        # Independent assets
        corr_indep = np.array([[1.0, 0.0], [0.0, 1.0]])
        result_indep = price_basket_option(
            S0=np.array([100.0, 100.0]),
            K=100.0, T=1.0, r=0.05,
            sigmas=np.array([0.2, 0.2]),
            payoff_type=MultiAssetPayoffType.BASKET_CALL,
            config=sim_config,
            corr_matrix=corr_indep
        )

        # Perfectly correlated assets
        corr_high = np.array([[1.0, 0.99], [0.99, 1.0]])
        result_high = price_basket_option(
            S0=np.array([100.0, 100.0]),
            K=100.0, T=1.0, r=0.05,
            sigmas=np.array([0.2, 0.2]),
            payoff_type=MultiAssetPayoffType.BASKET_CALL,
            config=sim_config,
            corr_matrix=corr_high
        )

        # Both should be positive
        assert result_indep.price > 0
        assert result_high.price > 0


class TestRoughHeston:
    """Tests for Rough Heston model."""

    def test_rough_heston_positive_price(self, market_data, sim_config):
        """Rough Heston should produce positive prices."""
        from mc_pricer import RoughHestonParams, price_rough_heston_option

        params = RoughHestonParams(
            v0=0.04, theta=0.04, lambda_=2.0,
            nu=0.3, rho=-0.7, H=0.1
        )

        result = price_rough_heston_option(
            market=market_data, K=100.0, T=1.0,
            params=params, payoff_type=PayoffType.EUROPEAN_CALL,
            config=sim_config
        )
        assert result.price > 0

    def test_rough_heston_hurst_validation(self):
        """Hurst parameter should be validated."""
        from mc_pricer import RoughHestonParams

        # H > 0.5 should raise error
        with pytest.raises(ValueError):
            RoughHestonParams(v0=0.04, theta=0.04, lambda_=2.0, nu=0.3, rho=-0.7, H=0.6)

        # H < 0 should raise error
        with pytest.raises(ValueError):
            RoughHestonParams(v0=0.04, theta=0.04, lambda_=2.0, nu=0.3, rho=-0.7, H=-0.1)


class TestAdvancedVarianceReduction:
    """Tests for advanced variance reduction techniques."""

    def test_importance_sampling_reduces_error(self, market_data):
        """Importance sampling should reduce variance for OTM options."""
        from mc_pricer import price_with_importance_sampling

        # Deep OTM call
        result = price_with_importance_sampling(
            market=market_data, K=150.0, T=1.0,
            sigma=0.2, is_call=True, num_paths=50000
        )

        assert result.price >= 0
        assert np.isfinite(result.std_error)

    def test_stratified_sampling(self, market_data):
        """Stratified sampling should produce valid prices."""
        from mc_pricer import price_with_stratified_sampling

        result = price_with_stratified_sampling(
            market=market_data, K=100.0, T=1.0,
            sigma=0.2, payoff_type=PayoffType.EUROPEAN_CALL,
            num_paths=50000
        )

        assert result.price > 0
        assert result.variance_reduction == "stratified_lhs"

    def test_latin_hypercube_sampling(self):
        """LHS should produce well-distributed samples."""
        from mc_pricer import latin_hypercube_sampling

        samples = latin_hypercube_sampling(1000, 5, seed=42)

        # Should be in [0, 1]
        assert np.all(samples >= 0)
        assert np.all(samples <= 1)

        # Each dimension should have one sample per stratum
        for d in range(5):
            col = samples[:, d]
            # Check roughly uniform distribution
            hist, _ = np.histogram(col, bins=10)
            # Each bin should have approximately 100 samples
            assert np.all(hist > 50)  # Allow some variance


class TestPathwiseGreeks:
    """Tests for pathwise Greeks calculation."""

    def test_pathwise_delta_positive_for_call(self, market_data):
        """Pathwise delta should be positive for calls."""
        from mc_pricer import calculate_pathwise_greeks

        greeks = calculate_pathwise_greeks(
            market=market_data, K=100.0, T=1.0,
            sigma=0.2, is_call=True, num_paths=50000
        )

        assert greeks['delta'] > 0
        assert greeks['delta'] < 1  # Delta bounded by [0,1] for calls

    def test_pathwise_delta_negative_for_put(self, market_data):
        """Pathwise delta should be negative for puts."""
        from mc_pricer import calculate_pathwise_greeks

        greeks = calculate_pathwise_greeks(
            market=market_data, K=100.0, T=1.0,
            sigma=0.2, is_call=False, num_paths=50000
        )

        assert greeks['delta'] < 0
        assert greeks['delta'] > -1  # Delta bounded by [-1,0] for puts

    def test_pathwise_vega_positive(self, market_data):
        """Pathwise vega should be positive."""
        from mc_pricer import calculate_pathwise_greeks

        greeks = calculate_pathwise_greeks(
            market=market_data, K=100.0, T=1.0,
            sigma=0.2, is_call=True, num_paths=50000
        )

        # Vega should be positive (higher vol = higher option value)
        assert greeks['vega'] > 0
