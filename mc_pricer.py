"""
Monte Carlo Option Pricing Library with Advanced Features

This module implements a comprehensive Monte Carlo framework for exotic option
pricing with the following features:

Models:
    - Black-Scholes (GBM)
    - Heston stochastic volatility
    - Bates (Heston + jumps)

Discretization Schemes:
    - Euler-Maruyama (standard)
    - Milstein (higher order for variance)
    - Quadratic-Exponential (QE) - Andersen's scheme for variance

Variance Reduction:
    - Antithetic variates
    - Control variates (geometric Asian as control)
    - Importance sampling (optional)

Payoff Types:
    - European (call/put)
    - Asian (arithmetic/geometric average, call/put)
    - Barrier (knock-in/knock-out, up/down)
    - Lookback (floating/fixed strike)

Greeks:
    - Delta, Gamma, Vega, Theta, Rho
    - Bump-and-reprice method with antithetic paths

References:
    - Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering.
    - Andersen, L. (2008). Simple and efficient simulation of the Heston model.
    - Broadie, M. & Kaya, O. (2006). Exact simulation of stochastic volatility.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Callable, Union, Literal, List
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import warnings

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

# Try to import SciPy for QMC
try:
    from scipy.stats import qmc
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    qmc = None
    norm = None
    SCIPY_AVAILABLE = False


# =============================================================================
# Enums and Configuration
# =============================================================================

class Backend(Enum):
    """Computation backend selection."""
    NUMPY = "numpy"
    CUPY = "cupy"


class DiscretizationScheme(Enum):
    """Discretization scheme for SDEs."""
    EULER = "euler"
    MILSTEIN = "milstein"
    QE = "qe"  # Quadratic-Exponential


class VarianceReduction(Enum):
    """Variance reduction technique."""
    NONE = "none"
    ANTITHETIC = "antithetic"
    CONTROL_VARIATE = "control_variate"
    ANTITHETIC_CV = "antithetic_cv"  # Both


class PayoffType(Enum):
    """Option payoff type."""
    EUROPEAN_CALL = "european_call"
    EUROPEAN_PUT = "european_put"
    ASIAN_CALL = "asian_call"
    ASIAN_PUT = "asian_put"
    ASIAN_GEOM_CALL = "asian_geom_call"
    ASIAN_GEOM_PUT = "asian_geom_put"
    BARRIER_UP_OUT_CALL = "barrier_up_out_call"
    BARRIER_UP_IN_CALL = "barrier_up_in_call"
    BARRIER_DOWN_OUT_CALL = "barrier_down_out_call"
    BARRIER_DOWN_IN_CALL = "barrier_down_in_call"
    BARRIER_UP_OUT_PUT = "barrier_up_out_put"
    BARRIER_UP_IN_PUT = "barrier_up_in_put"
    BARRIER_DOWN_OUT_PUT = "barrier_down_out_put"
    BARRIER_DOWN_IN_PUT = "barrier_down_in_put"
    LOOKBACK_FIXED_CALL = "lookback_fixed_call"
    LOOKBACK_FIXED_PUT = "lookback_fixed_put"
    LOOKBACK_FLOATING_CALL = "lookback_floating_call"
    LOOKBACK_FLOATING_PUT = "lookback_floating_put"
    AMERICAN_CALL = "american_call"
    AMERICAN_PUT = "american_put"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MarketData:
    """Market data for option pricing."""
    S0: float  # Initial stock price
    r: float  # Risk-free rate
    q: float = 0.0  # Dividend yield


@dataclass
class HestonParams:
    """Heston model parameters."""
    v0: float  # Initial variance
    kappa: float  # Mean reversion speed
    theta: float  # Long-term variance
    xi: float  # Vol of vol
    rho: float  # Correlation

    def __post_init__(self):
        """Validate parameters."""
        if self.v0 < 0:
            raise ValueError(f"v0 must be non-negative, got {self.v0}")
        if self.kappa <= 0:
            raise ValueError(f"kappa must be positive, got {self.kappa}")
        if self.theta < 0:
            raise ValueError(f"theta must be non-negative, got {self.theta}")
        if self.xi < 0:
            raise ValueError(f"xi must be non-negative, got {self.xi}")
        if not -1 <= self.rho <= 1:
            raise ValueError(f"rho must be in [-1, 1], got {self.rho}")

    @property
    def feller_condition(self) -> bool:
        """Check if Feller condition is satisfied: 2*kappa*theta > xi^2."""
        return 2 * self.kappa * self.theta > self.xi ** 2


@dataclass
class JumpParams:
    """Merton jump diffusion parameters."""
    lambda_j: float = 0.0  # Jump intensity (jumps per year)
    mu_j: float = 0.0  # Mean of log jump size
    sigma_j: float = 0.0  # Std of log jump size

    def __post_init__(self):
        """Validate parameters."""
        if self.lambda_j < 0:
            raise ValueError(f"lambda_j must be non-negative, got {self.lambda_j}")
        if self.sigma_j < 0:
            raise ValueError(f"sigma_j must be non-negative, got {self.sigma_j}")

    @property
    def drift_compensation(self) -> float:
        """Jump-compensated drift: k = lambda * (E[J] - 1)."""
        return self.lambda_j * (np.exp(self.mu_j + 0.5 * self.sigma_j ** 2) - 1)


@dataclass
class BarrierParams:
    """Barrier option parameters."""
    barrier: float  # Barrier level
    rebate: float = 0.0  # Rebate if knocked out
    monitoring: Literal["continuous", "discrete"] = "discrete"


@dataclass
class AmericanConfig:
    """Configuration for American option pricing using LSM."""
    polynomial_degree: int = 3  # Degree of polynomial basis
    num_exercise_dates: Optional[int] = None  # None = use num_steps


@dataclass
class SABRParams:
    """SABR model parameters for FX/Rates volatility.

    The SABR model dynamics:
        dF = σ * F^β * dW1
        dσ = α * σ * dW2
        dW1 * dW2 = ρ * dt

    Where:
        F = forward price
        σ = stochastic volatility
        α = vol of vol
        β = CEV exponent (0 = normal, 1 = lognormal)
        ρ = correlation between F and σ
        ν = initial volatility (sigma_0)
    """
    alpha: float  # Vol of vol
    beta: float  # CEV exponent [0, 1]
    rho: float  # Correlation
    nu: float  # Initial volatility

    def __post_init__(self):
        """Validate parameters."""
        if self.alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {self.alpha}")
        if not 0 <= self.beta <= 1:
            raise ValueError(f"beta must be in [0, 1], got {self.beta}")
        if not -1 <= self.rho <= 1:
            raise ValueError(f"rho must be in [-1, 1], got {self.rho}")
        if self.nu <= 0:
            raise ValueError(f"nu must be positive, got {self.nu}")

    def implied_vol_hagan(self, F: float, K: float, T: float) -> float:
        """
        Hagan's approximation for SABR implied volatility.

        This is the industry-standard closed-form approximation.
        """
        if abs(F - K) < 1e-10:
            # ATM case
            FK_mid = F
            term1 = self.nu * (1 + (
                ((1 - self.beta) ** 2 / 24) * (self.nu ** 2 / FK_mid ** (2 - 2 * self.beta)) +
                (self.rho * self.beta * self.alpha * self.nu) / (4 * FK_mid ** (1 - self.beta)) +
                ((2 - 3 * self.rho ** 2) / 24) * self.alpha ** 2
            ) * T)
            return term1 / (FK_mid ** (1 - self.beta))
        else:
            # OTM/ITM case
            log_FK = np.log(F / K)
            FK_mid = np.sqrt(F * K)

            z = (self.alpha / self.nu) * (FK_mid ** (1 - self.beta)) * log_FK
            x_z = np.log((np.sqrt(1 - 2 * self.rho * z + z ** 2) + z - self.rho) / (1 - self.rho))

            if abs(x_z) < 1e-10:
                x_z = z

            numerator = self.nu * (1 + (
                ((1 - self.beta) ** 2 / 24) * (self.nu ** 2 / FK_mid ** (2 - 2 * self.beta)) +
                (self.rho * self.beta * self.alpha * self.nu) / (4 * FK_mid ** (1 - self.beta)) +
                ((2 - 3 * self.rho ** 2) / 24) * self.alpha ** 2
            ) * T)

            denominator = (FK_mid ** (1 - self.beta)) * (
                1 + ((1 - self.beta) ** 2 / 24) * log_FK ** 2 +
                ((1 - self.beta) ** 4 / 1920) * log_FK ** 4
            )

            return (numerator / denominator) * (z / x_z) if abs(x_z) > 1e-10 else numerator / denominator


@dataclass
class BasketConfig:
    """Configuration for multi-asset basket options."""
    weights: Optional[np.ndarray] = None  # Asset weights (default: equal weighted)
    correlation_matrix: Optional[np.ndarray] = None  # Correlation matrix


class MultiAssetPayoffType(Enum):
    """Multi-asset option payoff types."""
    BASKET_CALL = "basket_call"
    BASKET_PUT = "basket_put"
    BEST_OF_CALL = "best_of_call"  # Rainbow: call on max
    WORST_OF_CALL = "worst_of_call"  # Rainbow: call on min
    BEST_OF_PUT = "best_of_put"  # Rainbow: put on max
    WORST_OF_PUT = "worst_of_put"  # Rainbow: put on min
    SPREAD_CALL = "spread_call"  # S1 - S2
    SPREAD_PUT = "spread_put"


@dataclass
class SimulationConfig:
    """Monte Carlo simulation configuration."""
    num_paths: int = 100000
    num_steps: int = 252
    backend: Backend = Backend.NUMPY
    scheme: DiscretizationScheme = DiscretizationScheme.QE
    variance_reduction: VarianceReduction = VarianceReduction.ANTITHETIC_CV
    seed: Optional[int] = None
    use_sobol: bool = False

    def __post_init__(self):
        """Validate configuration."""
        if self.num_paths <= 0:
            raise ValueError(f"num_paths must be positive, got {self.num_paths}")
        if self.num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {self.num_steps}")
        if self.backend == Backend.CUPY and not CUPY_AVAILABLE:
            warnings.warn("CuPy not available, falling back to NumPy")
            self.backend = Backend.NUMPY
        if self.use_sobol and not SCIPY_AVAILABLE:
            warnings.warn("SciPy not available, falling back to pseudo-random")
            self.use_sobol = False


@dataclass
class PricingResult:
    """Result of option pricing."""
    price: float
    std_error: float
    paths_used: int
    variance_reduction: str
    control_variate_beta: Optional[float] = None
    elapsed_time: Optional[float] = None
    greeks: Optional[Dict[str, float]] = None


# =============================================================================
# Array Operations (Backend-agnostic)
# =============================================================================

def get_array_module(backend: Backend):
    """Get the appropriate array module."""
    if backend == Backend.CUPY and CUPY_AVAILABLE:
        return cp
    return np


def to_numpy(arr, backend: Backend):
    """Convert array to NumPy."""
    if backend == Backend.CUPY and CUPY_AVAILABLE:
        return cp.asnumpy(arr)
    return arr


# =============================================================================
# Random Number Generation
# =============================================================================

def generate_random_numbers(
    shape: Tuple[int, ...],
    backend: Backend,
    seed: Optional[int] = None,
    use_sobol: bool = False,
    antithetic: bool = False
) -> np.ndarray:
    """
    Generate random numbers for Monte Carlo simulation.

    Args:
        shape: Shape of the random array (num_paths, num_steps)
        backend: Computation backend
        seed: Random seed
        use_sobol: Use Sobol quasi-random sequence
        antithetic: Generate antithetic pairs

    Returns:
        Array of standard normal random numbers
    """
    xp = get_array_module(backend)
    num_paths, num_steps = shape

    if antithetic:
        # Generate half the paths and create antithetic pairs
        half_paths = num_paths // 2

        if use_sobol and SCIPY_AVAILABLE:
            sampler = qmc.Sobol(d=num_steps, scramble=True, seed=seed)
            U = sampler.random(n=half_paths)
            U = np.clip(U, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
            Z_half = norm.ppf(U)
        else:
            rng = np.random.default_rng(seed)
            Z_half = rng.standard_normal((half_paths, num_steps))

        Z = np.concatenate([Z_half, -Z_half], axis=0)

        # Handle odd number of paths
        if num_paths % 2 == 1:
            rng = np.random.default_rng(seed + 1 if seed else None)
            Z_extra = rng.standard_normal((1, num_steps))
            Z = np.concatenate([Z, Z_extra], axis=0)
    else:
        if use_sobol and SCIPY_AVAILABLE:
            sampler = qmc.Sobol(d=num_steps, scramble=True, seed=seed)
            U = sampler.random(n=num_paths)
            U = np.clip(U, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
            Z = norm.ppf(U)
        else:
            rng = np.random.default_rng(seed)
            Z = rng.standard_normal(shape)

    if backend == Backend.CUPY and CUPY_AVAILABLE:
        Z = cp.asarray(Z)

    return Z


# =============================================================================
# Discretization Schemes
# =============================================================================

def euler_variance_step(
    v: np.ndarray,
    Z_v: np.ndarray,
    dt: float,
    kappa: float,
    theta: float,
    xi: float,
    xp=np
) -> np.ndarray:
    """
    Euler-Maruyama step for variance process.

    Uses full truncation scheme to prevent negative variance.
    """
    sqrt_dt = np.sqrt(dt)
    v_pos = xp.maximum(v, 0.0)
    sqrt_v = xp.sqrt(v_pos)

    dv = kappa * (theta - v_pos) * dt + xi * sqrt_v * Z_v * sqrt_dt
    return v + dv


def milstein_variance_step(
    v: np.ndarray,
    Z_v: np.ndarray,
    dt: float,
    kappa: float,
    theta: float,
    xi: float,
    xp=np
) -> np.ndarray:
    """
    Milstein step for variance process.

    Adds the Milstein correction term for higher-order convergence.
    """
    sqrt_dt = np.sqrt(dt)
    v_pos = xp.maximum(v, 0.0)
    sqrt_v = xp.sqrt(v_pos)

    # Euler term
    dv = kappa * (theta - v_pos) * dt + xi * sqrt_v * Z_v * sqrt_dt

    # Milstein correction: 0.25 * xi^2 * (Z^2 - 1) * dt
    milstein_correction = 0.25 * xi ** 2 * (Z_v ** 2 - 1) * dt

    return v + dv + milstein_correction


def qe_variance_step(
    v: np.ndarray,
    U: np.ndarray,
    dt: float,
    kappa: float,
    theta: float,
    xi: float,
    xp=np,
    psi_c: float = 1.5
) -> np.ndarray:
    """
    Quadratic-Exponential (QE) scheme for variance process.

    Andersen's (2008) moment-matching scheme that:
    - Exactly matches first two moments
    - Handles low variance regimes without going negative
    - Uses exponential for low variance, quadratic for high variance

    Args:
        v: Current variance values
        U: Uniform random numbers in (0, 1)
        dt: Time step
        kappa, theta, xi: Heston parameters
        xp: Array module (numpy or cupy)
        psi_c: Critical threshold for switching (typically 1.5)
    """
    # Compute moments
    exp_kdt = xp.exp(-kappa * dt)
    m = theta + (v - theta) * exp_kdt  # Mean of v(t+dt)

    s2 = (v * xi ** 2 * exp_kdt / kappa * (1 - exp_kdt) +
          theta * xi ** 2 / (2 * kappa) * (1 - exp_kdt) ** 2)  # Variance

    psi = s2 / (m ** 2 + 1e-10)  # Coefficient of variation squared

    # Initialize output and fill branch-by-branch using explicit masked assignment.
    # This preserves path alignment when only a subset of paths take each branch.
    v_new = xp.zeros_like(v)

    # Case 1: psi <= psi_c (quadratic scheme)
    mask_quad = psi <= psi_c
    if xp.any(mask_quad):
        inv_psi = 1.0 / (psi[mask_quad] + 1e-10)
        b2 = 2 * inv_psi - 1 + xp.sqrt(2 * inv_psi) * xp.sqrt(2 * inv_psi - 1)
        b2 = xp.maximum(b2, 0.0)
        a = m[mask_quad] / (1 + b2)

        # Convert uniform to normal using inverse CDF
        if hasattr(xp, 'asnumpy'):  # CuPy
            Z = xp.asarray(norm.ppf(to_numpy(U[mask_quad], Backend.CUPY)))
        else:
            Z = norm.ppf(U[mask_quad]) if SCIPY_AVAILABLE else xp.sqrt(2) * xp.erfinv(2 * U[mask_quad] - 1)

        v_new_quad = a * (xp.sqrt(b2) + Z) ** 2
        v_new[mask_quad] = v_new_quad

    # Case 2: psi > psi_c (exponential scheme)
    mask_exp = ~mask_quad
    if xp.any(mask_exp):
        p = (psi[mask_exp] - 1) / (psi[mask_exp] + 1)
        beta = (1 - p) / (m[mask_exp] + 1e-10)

        # Inverse CDF of exponential mixture
        U_exp = U[mask_exp]
        v_new_exp = xp.where(
            U_exp <= p,
            0.0,
            xp.log((1 - p) / (1 - U_exp + 1e-10)) / (beta + 1e-10)
        )

        v_new[mask_exp] = v_new_exp

    return xp.maximum(v_new, 0.0)


# =============================================================================
# Path Simulation
# =============================================================================

def simulate_gbm_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    num_paths: int,
    num_steps: int,
    config: SimulationConfig
) -> np.ndarray:
    """
    Simulate GBM (Black-Scholes) paths.

    Returns:
        Array of shape (num_paths, num_steps + 1) with price paths
    """
    xp = get_array_module(config.backend)
    dt = T / num_steps
    sqrt_dt = np.sqrt(dt)

    use_antithetic = config.variance_reduction in [
        VarianceReduction.ANTITHETIC,
        VarianceReduction.ANTITHETIC_CV
    ]

    Z = generate_random_numbers(
        (num_paths, num_steps),
        config.backend,
        config.seed,
        config.use_sobol,
        use_antithetic
    )

    # Log-Euler scheme
    drift = (r - 0.5 * sigma ** 2) * dt
    diffusion = sigma * sqrt_dt * Z

    log_S = xp.zeros((num_paths, num_steps + 1))
    log_S[:, 0] = np.log(S0)
    log_S[:, 1:] = xp.cumsum(drift + diffusion, axis=1) + np.log(S0)

    return xp.exp(log_S)


def simulate_heston_paths(
    market: MarketData,
    heston: HestonParams,
    T: float,
    config: SimulationConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate Heston stochastic volatility paths.

    Returns:
        Tuple of (S_paths, v_paths), each of shape (num_paths, num_steps + 1)
    """
    xp = get_array_module(config.backend)
    num_paths = config.num_paths
    num_steps = config.num_steps
    dt = T / num_steps
    sqrt_dt = np.sqrt(dt)

    use_antithetic = config.variance_reduction in [
        VarianceReduction.ANTITHETIC,
        VarianceReduction.ANTITHETIC_CV
    ]

    # Generate correlated random numbers
    Z1 = generate_random_numbers(
        (num_paths, num_steps), config.backend, config.seed, config.use_sobol, use_antithetic
    )
    Z2 = generate_random_numbers(
        (num_paths, num_steps), config.backend,
        config.seed + 1 if config.seed else None, config.use_sobol, use_antithetic
    )

    # Cholesky decomposition for correlation
    rho = heston.rho
    Z_v = Z1
    Z_S = rho * Z1 + np.sqrt(1 - rho ** 2) * Z2

    # Initialize paths
    S = xp.zeros((num_paths, num_steps + 1))
    v = xp.zeros((num_paths, num_steps + 1))
    S[:, 0] = market.S0
    v[:, 0] = heston.v0

    # Simulate paths
    for t in range(num_steps):
        v_curr = v[:, t]

        if config.scheme == DiscretizationScheme.QE:
            # Generate uniform for QE scheme
            if hasattr(xp, 'random'):
                U = xp.random.random(num_paths)
            else:
                U = np.random.random(num_paths)
                if config.backend == Backend.CUPY:
                    U = cp.asarray(U)

            v_next = qe_variance_step(
                v_curr, U, dt, heston.kappa, heston.theta, heston.xi, xp
            )
        elif config.scheme == DiscretizationScheme.MILSTEIN:
            v_next = milstein_variance_step(
                v_curr, Z_v[:, t], dt, heston.kappa, heston.theta, heston.xi, xp
            )
        else:  # Euler
            v_next = euler_variance_step(
                v_curr, Z_v[:, t], dt, heston.kappa, heston.theta, heston.xi, xp
            )

        v[:, t + 1] = xp.maximum(v_next, 0.0)

        # Stock price update (log-Euler)
        v_pos = xp.maximum(v_curr, 0.0)
        sqrt_v = xp.sqrt(v_pos)

        drift = (market.r - market.q - 0.5 * v_pos) * dt
        diffusion = sqrt_v * Z_S[:, t] * sqrt_dt

        S[:, t + 1] = S[:, t] * xp.exp(drift + diffusion)

    return S, v


def simulate_bates_paths(
    market: MarketData,
    heston: HestonParams,
    jump: JumpParams,
    T: float,
    config: SimulationConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate Bates model paths (Heston + jumps).

    Returns:
        Tuple of (S_paths, v_paths), each of shape (num_paths, num_steps + 1)
    """
    xp = get_array_module(config.backend)
    num_paths = config.num_paths
    num_steps = config.num_steps
    dt = T / num_steps
    sqrt_dt = np.sqrt(dt)

    use_antithetic = config.variance_reduction in [
        VarianceReduction.ANTITHETIC,
        VarianceReduction.ANTITHETIC_CV
    ]

    # Generate correlated random numbers
    Z1 = generate_random_numbers(
        (num_paths, num_steps), config.backend, config.seed, config.use_sobol, use_antithetic
    )
    Z2 = generate_random_numbers(
        (num_paths, num_steps), config.backend,
        config.seed + 1 if config.seed else None, config.use_sobol, use_antithetic
    )

    rho = heston.rho
    Z_v = Z1
    Z_S = rho * Z1 + np.sqrt(1 - rho ** 2) * Z2

    # Initialize paths
    S = xp.zeros((num_paths, num_steps + 1))
    v = xp.zeros((num_paths, num_steps + 1))
    S[:, 0] = market.S0
    v[:, 0] = heston.v0

    # Jump compensation
    k_drift = jump.drift_compensation

    # Random generator for jumps
    rng = np.random.default_rng(config.seed + 2 if config.seed else None)

    # Simulate paths
    for t in range(num_steps):
        v_curr = v[:, t]

        # Variance update
        if config.scheme == DiscretizationScheme.QE:
            if hasattr(xp, 'random'):
                U = xp.random.random(num_paths)
            else:
                U = np.random.random(num_paths)
                if config.backend == Backend.CUPY:
                    U = cp.asarray(U)

            v_next = qe_variance_step(
                v_curr, U, dt, heston.kappa, heston.theta, heston.xi, xp
            )
        elif config.scheme == DiscretizationScheme.MILSTEIN:
            v_next = milstein_variance_step(
                v_curr, Z_v[:, t], dt, heston.kappa, heston.theta, heston.xi, xp
            )
        else:
            v_next = euler_variance_step(
                v_curr, Z_v[:, t], dt, heston.kappa, heston.theta, heston.xi, xp
            )

        v[:, t + 1] = xp.maximum(v_next, 0.0)

        # Generate jumps
        if jump.lambda_j > 0:
            N_jumps = rng.poisson(jump.lambda_j * dt, num_paths)
            jump_sizes = xp.zeros(num_paths)

            for i in range(num_paths):
                if N_jumps[i] > 0:
                    J = rng.normal(jump.mu_j, jump.sigma_j, N_jumps[i])
                    jump_sizes[i] = np.sum(J)

            if config.backend == Backend.CUPY:
                jump_sizes = cp.asarray(jump_sizes)
        else:
            jump_sizes = 0.0

        # Stock price update
        v_pos = xp.maximum(v_curr, 0.0)
        sqrt_v = xp.sqrt(v_pos)

        drift = (market.r - market.q - k_drift - 0.5 * v_pos) * dt
        diffusion = sqrt_v * Z_S[:, t] * sqrt_dt

        S[:, t + 1] = S[:, t] * xp.exp(drift + diffusion + jump_sizes)

    return S, v


# =============================================================================
# Payoff Functions
# =============================================================================

def european_call_payoff(S_paths: np.ndarray, K: float, xp=np) -> np.ndarray:
    """European call payoff: max(S_T - K, 0)."""
    return xp.maximum(S_paths[:, -1] - K, 0.0)


def european_put_payoff(S_paths: np.ndarray, K: float, xp=np) -> np.ndarray:
    """European put payoff: max(K - S_T, 0)."""
    return xp.maximum(K - S_paths[:, -1], 0.0)


def asian_arithmetic_call_payoff(S_paths: np.ndarray, K: float, xp=np) -> np.ndarray:
    """Asian call payoff with arithmetic average: max(A - K, 0)."""
    A = xp.mean(S_paths, axis=1)
    return xp.maximum(A - K, 0.0)


def asian_arithmetic_put_payoff(S_paths: np.ndarray, K: float, xp=np) -> np.ndarray:
    """Asian put payoff with arithmetic average: max(K - A, 0)."""
    A = xp.mean(S_paths, axis=1)
    return xp.maximum(K - A, 0.0)


def asian_geometric_call_payoff(S_paths: np.ndarray, K: float, xp=np) -> np.ndarray:
    """Asian call payoff with geometric average: max(G - K, 0)."""
    G = xp.exp(xp.mean(xp.log(S_paths), axis=1))
    return xp.maximum(G - K, 0.0)


def asian_geometric_put_payoff(S_paths: np.ndarray, K: float, xp=np) -> np.ndarray:
    """Asian put payoff with geometric average: max(K - G, 0)."""
    G = xp.exp(xp.mean(xp.log(S_paths), axis=1))
    return xp.maximum(K - G, 0.0)


def barrier_up_out_call_payoff(
    S_paths: np.ndarray, K: float, barrier: float, rebate: float = 0.0, xp=np
) -> np.ndarray:
    """Up-and-out call: Call payoff if max(S) < barrier, else rebate."""
    max_S = xp.max(S_paths, axis=1)
    knocked_out = max_S >= barrier
    vanilla_payoff = xp.maximum(S_paths[:, -1] - K, 0.0)
    return xp.where(knocked_out, rebate, vanilla_payoff)


def barrier_up_in_call_payoff(
    S_paths: np.ndarray, K: float, barrier: float, rebate: float = 0.0, xp=np
) -> np.ndarray:
    """Up-and-in call: Call payoff if max(S) >= barrier, else rebate."""
    max_S = xp.max(S_paths, axis=1)
    knocked_in = max_S >= barrier
    vanilla_payoff = xp.maximum(S_paths[:, -1] - K, 0.0)
    return xp.where(knocked_in, vanilla_payoff, rebate)


def barrier_down_out_call_payoff(
    S_paths: np.ndarray, K: float, barrier: float, rebate: float = 0.0, xp=np
) -> np.ndarray:
    """Down-and-out call: Call payoff if min(S) > barrier, else rebate."""
    min_S = xp.min(S_paths, axis=1)
    knocked_out = min_S <= barrier
    vanilla_payoff = xp.maximum(S_paths[:, -1] - K, 0.0)
    return xp.where(knocked_out, rebate, vanilla_payoff)


def barrier_down_in_call_payoff(
    S_paths: np.ndarray, K: float, barrier: float, rebate: float = 0.0, xp=np
) -> np.ndarray:
    """Down-and-in call: Call payoff if min(S) <= barrier, else rebate."""
    min_S = xp.min(S_paths, axis=1)
    knocked_in = min_S <= barrier
    vanilla_payoff = xp.maximum(S_paths[:, -1] - K, 0.0)
    return xp.where(knocked_in, vanilla_payoff, rebate)


def barrier_up_out_put_payoff(
    S_paths: np.ndarray, K: float, barrier: float, rebate: float = 0.0, xp=np
) -> np.ndarray:
    """Up-and-out put: Put payoff if max(S) < barrier, else rebate."""
    max_S = xp.max(S_paths, axis=1)
    knocked_out = max_S >= barrier
    vanilla_payoff = xp.maximum(K - S_paths[:, -1], 0.0)
    return xp.where(knocked_out, rebate, vanilla_payoff)


def barrier_up_in_put_payoff(
    S_paths: np.ndarray, K: float, barrier: float, rebate: float = 0.0, xp=np
) -> np.ndarray:
    """Up-and-in put: Put payoff if max(S) >= barrier, else rebate."""
    max_S = xp.max(S_paths, axis=1)
    knocked_in = max_S >= barrier
    vanilla_payoff = xp.maximum(K - S_paths[:, -1], 0.0)
    return xp.where(knocked_in, vanilla_payoff, rebate)


def barrier_down_out_put_payoff(
    S_paths: np.ndarray, K: float, barrier: float, rebate: float = 0.0, xp=np
) -> np.ndarray:
    """Down-and-out put: Put payoff if min(S) > barrier, else rebate."""
    min_S = xp.min(S_paths, axis=1)
    knocked_out = min_S <= barrier
    vanilla_payoff = xp.maximum(K - S_paths[:, -1], 0.0)
    return xp.where(knocked_out, rebate, vanilla_payoff)


def barrier_down_in_put_payoff(
    S_paths: np.ndarray, K: float, barrier: float, rebate: float = 0.0, xp=np
) -> np.ndarray:
    """Down-and-in put: Put payoff if min(S) <= barrier, else rebate."""
    min_S = xp.min(S_paths, axis=1)
    knocked_in = min_S <= barrier
    vanilla_payoff = xp.maximum(K - S_paths[:, -1], 0.0)
    return xp.where(knocked_in, vanilla_payoff, rebate)


def lookback_fixed_call_payoff(S_paths: np.ndarray, K: float, xp=np) -> np.ndarray:
    """Fixed strike lookback call: max(max(S) - K, 0)."""
    max_S = xp.max(S_paths, axis=1)
    return xp.maximum(max_S - K, 0.0)


def lookback_fixed_put_payoff(S_paths: np.ndarray, K: float, xp=np) -> np.ndarray:
    """Fixed strike lookback put: max(K - min(S), 0)."""
    min_S = xp.min(S_paths, axis=1)
    return xp.maximum(K - min_S, 0.0)


def lookback_floating_call_payoff(S_paths: np.ndarray, K: float = 0.0, xp=np) -> np.ndarray:
    """Floating strike lookback call: S_T - min(S)."""
    min_S = xp.min(S_paths, axis=1)
    return S_paths[:, -1] - min_S


def lookback_floating_put_payoff(S_paths: np.ndarray, K: float = 0.0, xp=np) -> np.ndarray:
    """Floating strike lookback put: max(S) - S_T."""
    max_S = xp.max(S_paths, axis=1)
    return max_S - S_paths[:, -1]


# Payoff function registry
PAYOFF_FUNCTIONS = {
    PayoffType.EUROPEAN_CALL: european_call_payoff,
    PayoffType.EUROPEAN_PUT: european_put_payoff,
    PayoffType.ASIAN_CALL: asian_arithmetic_call_payoff,
    PayoffType.ASIAN_PUT: asian_arithmetic_put_payoff,
    PayoffType.ASIAN_GEOM_CALL: asian_geometric_call_payoff,
    PayoffType.ASIAN_GEOM_PUT: asian_geometric_put_payoff,
    PayoffType.BARRIER_UP_OUT_CALL: barrier_up_out_call_payoff,
    PayoffType.BARRIER_UP_IN_CALL: barrier_up_in_call_payoff,
    PayoffType.BARRIER_DOWN_OUT_CALL: barrier_down_out_call_payoff,
    PayoffType.BARRIER_DOWN_IN_CALL: barrier_down_in_call_payoff,
    PayoffType.BARRIER_UP_OUT_PUT: barrier_up_out_put_payoff,
    PayoffType.BARRIER_UP_IN_PUT: barrier_up_in_put_payoff,
    PayoffType.BARRIER_DOWN_OUT_PUT: barrier_down_out_put_payoff,
    PayoffType.BARRIER_DOWN_IN_PUT: barrier_down_in_put_payoff,
    PayoffType.LOOKBACK_FIXED_CALL: lookback_fixed_call_payoff,
    PayoffType.LOOKBACK_FIXED_PUT: lookback_fixed_put_payoff,
    PayoffType.LOOKBACK_FLOATING_CALL: lookback_floating_call_payoff,
    PayoffType.LOOKBACK_FLOATING_PUT: lookback_floating_put_payoff,
}

BARRIER_PAYOFF_TYPES = {
    PayoffType.BARRIER_UP_OUT_CALL,
    PayoffType.BARRIER_UP_IN_CALL,
    PayoffType.BARRIER_DOWN_OUT_CALL,
    PayoffType.BARRIER_DOWN_IN_CALL,
    PayoffType.BARRIER_UP_OUT_PUT,
    PayoffType.BARRIER_UP_IN_PUT,
    PayoffType.BARRIER_DOWN_OUT_PUT,
    PayoffType.BARRIER_DOWN_IN_PUT,
}


# =============================================================================
# Control Variates
# =============================================================================

def geometric_asian_call_analytical(
    S0: float, K: float, r: float, sigma: float, T: float, n: int
) -> float:
    """
    Analytical price of geometric average Asian call.

    The geometric average of GBM follows a log-normal distribution,
    allowing for a closed-form Black-Scholes-like formula.
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("SciPy required for analytical geometric Asian price")

    # Adjusted volatility for discrete monitoring
    sigma_adj = sigma * np.sqrt((2 * n + 1) / (6 * (n + 1)))

    # Adjusted drift
    mu = (r - 0.5 * sigma ** 2) * (n + 1) / (2 * n) + 0.5 * sigma_adj ** 2

    # Effective rate
    r_adj = mu - 0.5 * sigma_adj ** 2

    # Black-Scholes formula with adjusted parameters
    d1 = (np.log(S0 / K) + (r_adj + 0.5 * sigma_adj ** 2) * T) / (sigma_adj * np.sqrt(T))
    d2 = d1 - sigma_adj * np.sqrt(T)

    price = S0 * np.exp((r_adj - r) * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return max(price, 0.0)


def geometric_asian_put_analytical(
    S0: float, K: float, r: float, sigma: float, T: float, n: int
) -> float:
    """Analytical price of geometric average Asian put via put-call parity."""
    if not SCIPY_AVAILABLE:
        raise ImportError("SciPy required for analytical geometric Asian price")

    call_price = geometric_asian_call_analytical(S0, K, r, sigma, T, n)

    # Adjusted forward for geometric average
    sigma_adj = sigma * np.sqrt((2 * n + 1) / (6 * (n + 1)))
    mu = (r - 0.5 * sigma ** 2) * (n + 1) / (2 * n) + 0.5 * sigma_adj ** 2
    r_adj = mu - 0.5 * sigma_adj ** 2

    # Put-call parity
    put_price = call_price - S0 * np.exp((r_adj - r) * T) + K * np.exp(-r * T)
    return max(put_price, 0.0)


def apply_control_variate(
    payoffs: np.ndarray,
    control_payoffs: np.ndarray,
    control_mean: float,
    xp=np
) -> Tuple[np.ndarray, float]:
    """
    Apply control variate technique.

    Args:
        payoffs: Original payoffs
        control_payoffs: Control variate payoffs
        control_mean: Known expected value of control

    Returns:
        Tuple of (adjusted payoffs, optimal beta)
    """
    # Compute optimal beta
    cov = xp.cov(payoffs, control_payoffs)[0, 1]
    var_control = xp.var(control_payoffs)
    beta = cov / (var_control + 1e-10)

    # Adjust payoffs
    adjusted = payoffs - float(beta) * (control_payoffs - control_mean)

    return adjusted, float(beta)


# =============================================================================
# Greeks Calculation
# =============================================================================

def calculate_delta(
    pricer_func: Callable,
    S0: float,
    bump_pct: float = 0.01,
    **kwargs
) -> float:
    """
    Calculate Delta using central difference.

    Delta = (V(S+dS) - V(S-dS)) / (2*dS)
    """
    dS = S0 * bump_pct

    price_up = pricer_func(S0=S0 + dS, **kwargs)
    price_down = pricer_func(S0=S0 - dS, **kwargs)

    return (price_up - price_down) / (2 * dS)


def calculate_gamma(
    pricer_func: Callable,
    S0: float,
    bump_pct: float = 0.01,
    **kwargs
) -> float:
    """
    Calculate Gamma using central difference.

    Gamma = (V(S+dS) - 2*V(S) + V(S-dS)) / (dS^2)
    """
    dS = S0 * bump_pct

    price_up = pricer_func(S0=S0 + dS, **kwargs)
    price_mid = pricer_func(S0=S0, **kwargs)
    price_down = pricer_func(S0=S0 - dS, **kwargs)

    return (price_up - 2 * price_mid + price_down) / (dS ** 2)


def calculate_vega(
    pricer_func: Callable,
    sigma: float,
    bump_pct: float = 0.01,
    **kwargs
) -> float:
    """
    Calculate Vega using central difference.

    Vega = (V(sigma+dsigma) - V(sigma-dsigma)) / (2*dsigma)
    Returned per 1% change in volatility.
    """
    d_sigma = sigma * bump_pct

    price_up = pricer_func(sigma=sigma + d_sigma, **kwargs)
    price_down = pricer_func(sigma=sigma - d_sigma, **kwargs)

    # Scale to per 1% vol change
    return (price_up - price_down) / (2 * d_sigma) * 0.01


def calculate_theta(
    pricer_func: Callable,
    T: float,
    bump_days: float = 1.0,
    **kwargs
) -> float:
    """
    Calculate Theta using forward difference.

    Theta = (V(T-dt) - V(T)) / dt
    Returned per day.
    """
    dt = bump_days / 365.0

    if T - dt <= 0:
        # Can't bump forward if too close to expiry
        return 0.0

    price_now = pricer_func(T=T, **kwargs)
    price_later = pricer_func(T=T - dt, **kwargs)

    return (price_later - price_now) / bump_days


def calculate_rho(
    pricer_func: Callable,
    r: float,
    bump_bps: float = 1.0,
    **kwargs
) -> float:
    """
    Calculate Rho using central difference.

    Rho = (V(r+dr) - V(r-dr)) / (2*dr)
    Returned per 1% change in rate.
    """
    dr = bump_bps / 10000.0

    price_up = pricer_func(r=r + dr, **kwargs)
    price_down = pricer_func(r=r - dr, **kwargs)

    # Scale to per 1% rate change
    return (price_up - price_down) / (2 * dr) * 0.01


def calculate_all_greeks(
    pricer_func: Callable,
    S0: float,
    sigma: float,
    r: float,
    T: float,
    **kwargs
) -> Dict[str, float]:
    """Calculate all Greeks for an option."""
    greeks = {
        'delta': calculate_delta(pricer_func, S0, sigma=sigma, r=r, T=T, **kwargs),
        'gamma': calculate_gamma(pricer_func, S0, sigma=sigma, r=r, T=T, **kwargs),
        'vega': calculate_vega(pricer_func, sigma, S0=S0, r=r, T=T, **kwargs),
        'theta': calculate_theta(pricer_func, T, S0=S0, sigma=sigma, r=r, **kwargs),
        'rho': calculate_rho(pricer_func, r, S0=S0, sigma=sigma, T=T, **kwargs),
    }
    return greeks


# =============================================================================
# Main Pricing Functions
# =============================================================================

def price_option(
    market: MarketData,
    K: float,
    T: float,
    payoff_type: PayoffType,
    heston: Optional[HestonParams] = None,
    jump: Optional[JumpParams] = None,
    barrier: Optional[BarrierParams] = None,
    config: Optional[SimulationConfig] = None,
    sigma: Optional[float] = None
) -> PricingResult:
    """
    Price an option using Monte Carlo simulation.

    Args:
        market: Market data (S0, r, q)
        K: Strike price
        T: Time to maturity
        payoff_type: Type of option payoff
        heston: Heston parameters (optional, for stochastic vol)
        jump: Jump parameters (optional, for Bates model)
        barrier: Barrier parameters (optional, for barrier options)
        config: Simulation configuration
        sigma: Constant volatility (required if heston is None)

    Returns:
        PricingResult with price, std error, and diagnostics
    """
    import time
    start_time = time.time()

    if config is None:
        config = SimulationConfig()

    # American options use a dedicated LSM pricer path.
    if payoff_type in (PayoffType.AMERICAN_CALL, PayoffType.AMERICAN_PUT):
        return price_american_option_lsm(
            market=market,
            K=K,
            T=T,
            is_call=payoff_type == PayoffType.AMERICAN_CALL,
            heston=heston,
            jump=jump,
            config=config,
            sigma=sigma
        )

    if payoff_type in BARRIER_PAYOFF_TYPES and barrier is None:
        raise ValueError(
            f"Barrier parameters are required for payoff type '{payoff_type.value}'"
        )

    xp = get_array_module(config.backend)

    # Simulate paths
    if heston is not None:
        if jump is not None and jump.lambda_j > 0:
            S_paths, v_paths = simulate_bates_paths(market, heston, jump, T, config)
        else:
            S_paths, v_paths = simulate_heston_paths(market, heston, T, config)
    else:
        if sigma is None:
            raise ValueError("sigma required when heston is None")
        S_paths = simulate_gbm_paths(
            market.S0, market.r, sigma, T, config.num_paths, config.num_steps, config
        )

    # Get payoff function
    payoff_func = PAYOFF_FUNCTIONS.get(payoff_type)
    if payoff_func is None:
        raise ValueError(f"Unknown payoff type: {payoff_type}")

    # Compute payoffs
    if payoff_type in BARRIER_PAYOFF_TYPES:
        payoffs = payoff_func(S_paths, K, barrier.barrier, barrier.rebate, xp)
    else:
        payoffs = payoff_func(S_paths, K, xp)

    # Apply control variates if applicable
    control_beta = None
    if config.variance_reduction in [VarianceReduction.CONTROL_VARIATE, VarianceReduction.ANTITHETIC_CV]:
        if payoff_type in [PayoffType.ASIAN_CALL, PayoffType.ASIAN_PUT] and sigma is not None:
            # Use geometric Asian as control
            geom_payoffs = asian_geometric_call_payoff(S_paths, K, xp) if payoff_type == PayoffType.ASIAN_CALL \
                          else asian_geometric_put_payoff(S_paths, K, xp)

            if SCIPY_AVAILABLE:
                geom_analytical = geometric_asian_call_analytical(market.S0, K, market.r, sigma, T, config.num_steps) \
                                 if payoff_type == PayoffType.ASIAN_CALL \
                                 else geometric_asian_put_analytical(market.S0, K, market.r, sigma, T, config.num_steps)

                payoffs, control_beta = apply_control_variate(
                    to_numpy(payoffs, config.backend),
                    to_numpy(geom_payoffs, config.backend),
                    geom_analytical
                )
                payoffs = xp.asarray(payoffs) if config.backend == Backend.CUPY else payoffs

    # Discount payoffs
    discount = np.exp(-market.r * T)
    discounted_payoffs = payoffs * discount

    # Compute statistics
    price = float(to_numpy(xp.mean(discounted_payoffs), config.backend))
    std_error = float(to_numpy(xp.std(discounted_payoffs), config.backend)) / np.sqrt(config.num_paths)

    elapsed_time = time.time() - start_time

    return PricingResult(
        price=price,
        std_error=std_error,
        paths_used=config.num_paths,
        variance_reduction=config.variance_reduction.value,
        control_variate_beta=control_beta,
        elapsed_time=elapsed_time
    )


def price_with_greeks(
    market: MarketData,
    K: float,
    T: float,
    payoff_type: PayoffType,
    heston: Optional[HestonParams] = None,
    jump: Optional[JumpParams] = None,
    barrier: Optional[BarrierParams] = None,
    config: Optional[SimulationConfig] = None,
    sigma: Optional[float] = None
) -> PricingResult:
    """
    Price an option and calculate all Greeks.

    Uses bump-and-reprice with shared random numbers for stability.
    """
    # Base price
    result = price_option(market, K, T, payoff_type, heston, jump, barrier, config, sigma)

    # Create pricing function for Greeks
    def pricer(S0=None, r=None, sigma_override=None, T_override=None):
        m = MarketData(
            S0=S0 if S0 is not None else market.S0,
            r=r if r is not None else market.r,
            q=market.q
        )
        _T = T_override if T_override is not None else T
        _sigma = sigma_override if sigma_override is not None else sigma

        res = price_option(m, K, _T, payoff_type, heston, jump, barrier, config, _sigma)
        return res.price

    # Calculate Greeks
    greeks = {}

    # Delta and Gamma
    dS = market.S0 * 0.01
    price_up = pricer(S0=market.S0 + dS)
    price_down = pricer(S0=market.S0 - dS)
    greeks['delta'] = (price_up - price_down) / (2 * dS)
    greeks['gamma'] = (price_up - 2 * result.price + price_down) / (dS ** 2)

    # Vega (for constant vol models)
    if sigma is not None:
        d_sigma = sigma * 0.01
        price_up_vol = pricer(sigma_override=sigma + d_sigma)
        price_down_vol = pricer(sigma_override=sigma - d_sigma)
        greeks['vega'] = (price_up_vol - price_down_vol) / (2 * d_sigma) * 0.01

    # Theta
    dt = 1.0 / 365.0
    if T > dt:
        price_later = pricer(T_override=T - dt)
        greeks['theta'] = (price_later - result.price)
    else:
        greeks['theta'] = 0.0

    # Rho
    dr = 0.0001  # 1 bp
    price_up_r = pricer(r=market.r + dr)
    price_down_r = pricer(r=market.r - dr)
    greeks['rho'] = (price_up_r - price_down_r) / (2 * dr) * 0.01

    result.greeks = greeks
    return result


# =============================================================================
# Convenience Functions
# =============================================================================

def price_asian_put(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    num_paths: int = 100000,
    num_steps: int = 252,
    heston: Optional[HestonParams] = None,
    jump: Optional[JumpParams] = None,
    use_gpu: bool = False
) -> PricingResult:
    """
    Convenience function to price an Asian put option.

    Args:
        S0: Initial stock price
        K: Strike price
        r: Risk-free rate
        sigma: Volatility (ignored if heston provided)
        T: Time to maturity
        num_paths: Number of Monte Carlo paths
        num_steps: Number of time steps
        heston: Optional Heston parameters for stochastic vol
        jump: Optional jump parameters for Bates model
        use_gpu: Use GPU acceleration if available

    Returns:
        PricingResult with price and statistics
    """
    market = MarketData(S0=S0, r=r)
    config = SimulationConfig(
        num_paths=num_paths,
        num_steps=num_steps,
        backend=Backend.CUPY if use_gpu and CUPY_AVAILABLE else Backend.NUMPY,
        scheme=DiscretizationScheme.QE if heston else DiscretizationScheme.EULER,
        variance_reduction=VarianceReduction.ANTITHETIC_CV
    )

    return price_option(
        market=market,
        K=K,
        T=T,
        payoff_type=PayoffType.ASIAN_PUT,
        heston=heston,
        jump=jump,
        config=config,
        sigma=sigma if heston is None else None
    )


def price_barrier_option(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    barrier: float,
    barrier_type: str = "down_out_put",
    rebate: float = 0.0,
    num_paths: int = 100000,
    num_steps: int = 252,
    use_gpu: bool = False
) -> PricingResult:
    """
    Convenience function to price a barrier option.

    Args:
        S0: Initial stock price
        K: Strike price
        r: Risk-free rate
        sigma: Volatility
        T: Time to maturity
        barrier: Barrier level
        barrier_type: Type of barrier (up_out_call, down_in_put, etc.)
        rebate: Rebate if knocked out
        num_paths: Number of Monte Carlo paths
        num_steps: Number of time steps
        use_gpu: Use GPU acceleration if available

    Returns:
        PricingResult with price and statistics
    """
    market = MarketData(S0=S0, r=r)
    barrier_params = BarrierParams(barrier=barrier, rebate=rebate)

    payoff_map = {
        "up_out_call": PayoffType.BARRIER_UP_OUT_CALL,
        "up_in_call": PayoffType.BARRIER_UP_IN_CALL,
        "down_out_call": PayoffType.BARRIER_DOWN_OUT_CALL,
        "down_in_call": PayoffType.BARRIER_DOWN_IN_CALL,
        "up_out_put": PayoffType.BARRIER_UP_OUT_PUT,
        "up_in_put": PayoffType.BARRIER_UP_IN_PUT,
        "down_out_put": PayoffType.BARRIER_DOWN_OUT_PUT,
        "down_in_put": PayoffType.BARRIER_DOWN_IN_PUT,
    }

    payoff_type = payoff_map.get(barrier_type)
    if payoff_type is None:
        raise ValueError(f"Unknown barrier type: {barrier_type}")

    config = SimulationConfig(
        num_paths=num_paths,
        num_steps=num_steps,
        backend=Backend.CUPY if use_gpu and CUPY_AVAILABLE else Backend.NUMPY,
        variance_reduction=VarianceReduction.ANTITHETIC
    )

    return price_option(
        market=market,
        K=K,
        T=T,
        payoff_type=payoff_type,
        barrier=barrier_params,
        config=config,
        sigma=sigma
    )


# =============================================================================
# Black-Scholes Analytical (for validation)
# =============================================================================

def black_scholes_call(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """Analytical Black-Scholes call price."""
    if not SCIPY_AVAILABLE:
        raise ImportError("SciPy required for Black-Scholes formula")

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """Analytical Black-Scholes put price."""
    if not SCIPY_AVAILABLE:
        raise ImportError("SciPy required for Black-Scholes formula")

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


# =============================================================================
# American Options - Longstaff-Schwartz LSM
# =============================================================================

def laguerre_basis(x: np.ndarray, degree: int, xp=np) -> np.ndarray:
    """
    Generate Laguerre polynomial basis functions.

    L_0(x) = 1
    L_1(x) = 1 - x
    L_2(x) = 1 - 2x + x^2/2
    """
    n = len(x)
    basis = xp.zeros((n, degree + 1))
    basis[:, 0] = 1.0

    if degree >= 1:
        basis[:, 1] = 1.0 - x
    if degree >= 2:
        basis[:, 2] = 1.0 - 2.0 * x + x ** 2 / 2.0
    if degree >= 3:
        for k in range(3, degree + 1):
            basis[:, k] = ((2 * k - 1 - x) * basis[:, k-1] - (k - 1) * basis[:, k-2]) / k

    return basis


def american_put_payoff_immediate(S: np.ndarray, K: float, xp=np) -> np.ndarray:
    """Immediate exercise value for American put."""
    return xp.maximum(K - S, 0.0)


def american_call_payoff_immediate(S: np.ndarray, K: float, xp=np) -> np.ndarray:
    """Immediate exercise value for American call."""
    return xp.maximum(S - K, 0.0)


def price_american_option_lsm(
    market: MarketData,
    K: float,
    T: float,
    is_call: bool = False,
    heston: Optional[HestonParams] = None,
    jump: Optional[JumpParams] = None,
    config: Optional[SimulationConfig] = None,
    sigma: Optional[float] = None,
    american_config: Optional[AmericanConfig] = None
) -> PricingResult:
    """
    Price American option using Longstaff-Schwartz Least Squares Monte Carlo.

    The LSM algorithm:
    1. Simulate price paths forward
    2. At maturity, option value = intrinsic value
    3. Work backwards: at each exercise date
       - Identify in-the-money paths
       - Regress discounted continuation value on basis functions of stock price
       - Compare immediate exercise to continuation value
       - Update cash flow matrix
    4. Discount expected payoffs to t=0

    Args:
        market: Market data
        K: Strike price
        T: Time to maturity
        is_call: True for call, False for put
        heston: Heston parameters (optional)
        jump: Jump parameters (optional)
        config: Simulation configuration
        sigma: Constant volatility (if not using Heston)
        american_config: LSM-specific configuration

    Returns:
        PricingResult with American option price
    """
    import time
    start_time = time.time()

    if config is None:
        config = SimulationConfig()
    if american_config is None:
        american_config = AmericanConfig()

    xp = get_array_module(config.backend)
    num_paths = config.num_paths
    num_steps = config.num_steps
    dt = T / num_steps
    discount_factor = np.exp(-market.r * dt)

    # Simulate paths
    if heston is not None:
        if jump is not None and jump.lambda_j > 0:
            S_paths, _ = simulate_bates_paths(market, heston, jump, T, config)
        else:
            S_paths, _ = simulate_heston_paths(market, heston, T, config)
    else:
        if sigma is None:
            raise ValueError("sigma required when heston is None")
        S_paths = simulate_gbm_paths(
            market.S0, market.r, sigma, T, num_paths, num_steps, config
        )

    # Convert to numpy for regression if using CuPy
    if config.backend == Backend.CUPY:
        S_paths = to_numpy(S_paths, config.backend)
        xp = np

    # Payoff function
    payoff_func = american_call_payoff_immediate if is_call else american_put_payoff_immediate

    # Initialize cash flow matrix (time when each path exercises and its payoff)
    cash_flows = xp.zeros(num_paths)
    exercise_time = xp.full(num_paths, num_steps)  # Default: exercise at maturity

    # Terminal payoff
    cash_flows = payoff_func(S_paths[:, -1], K, xp)

    # Backward induction
    for t in range(num_steps - 1, 0, -1):
        S_t = S_paths[:, t]

        # Immediate exercise value
        exercise_value = payoff_func(S_t, K, xp)

        # Find in-the-money paths
        itm = exercise_value > 0

        if xp.sum(itm) > american_config.polynomial_degree + 1:
            # Discounted continuation value from future
            continuation = cash_flows * xp.exp(-market.r * (exercise_time - t) * dt)

            # Regression on ITM paths only
            S_itm = S_t[itm]
            cont_itm = continuation[itm]

            # Normalize S for numerical stability
            S_norm = S_itm / K

            # Build basis functions
            basis = laguerre_basis(S_norm, american_config.polynomial_degree, xp)

            # Least squares regression
            try:
                coeffs = xp.linalg.lstsq(basis, cont_itm, rcond=None)[0]
                expected_continuation = basis @ coeffs
            except:
                # Fallback if regression fails
                expected_continuation = cont_itm

            # Exercise decision: exercise if immediate value > continuation
            exercise_mask = xp.zeros(num_paths, dtype=bool)
            itm_indices = xp.where(itm)[0]
            exercise_decisions = exercise_value[itm] > expected_continuation
            exercise_mask[itm_indices[exercise_decisions]] = True

            # Update cash flows for paths that exercise
            cash_flows = xp.where(exercise_mask, exercise_value, cash_flows)
            exercise_time = xp.where(exercise_mask, t, exercise_time)

    # Check exercise at t=0 (though typically not allowed)
    # Most American options can't be exercised at inception

    # Discount all cash flows to t=0
    discounted_payoffs = cash_flows * xp.exp(-market.r * exercise_time * dt)

    # Compute price and standard error
    price = float(xp.mean(discounted_payoffs))
    std_error = float(xp.std(discounted_payoffs)) / np.sqrt(num_paths)

    elapsed_time = time.time() - start_time

    return PricingResult(
        price=price,
        std_error=std_error,
        paths_used=num_paths,
        variance_reduction=config.variance_reduction.value,
        elapsed_time=elapsed_time
    )


# =============================================================================
# SABR Model
# =============================================================================

def simulate_sabr_paths(
    F0: float,
    sabr: SABRParams,
    T: float,
    config: SimulationConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate SABR model paths using Euler-Maruyama discretization.

    SABR dynamics:
        dF = σ * F^β * dW1
        dσ = α * σ * dW2
        dW1 * dW2 = ρ * dt

    Args:
        F0: Initial forward price
        sabr: SABR model parameters
        T: Time to maturity
        config: Simulation configuration

    Returns:
        Tuple of (F_paths, sigma_paths)
    """
    xp = get_array_module(config.backend)
    num_paths = config.num_paths
    num_steps = config.num_steps
    dt = T / num_steps
    sqrt_dt = np.sqrt(dt)

    use_antithetic = config.variance_reduction in [
        VarianceReduction.ANTITHETIC,
        VarianceReduction.ANTITHETIC_CV
    ]

    # Generate correlated random numbers
    Z1 = generate_random_numbers(
        (num_paths, num_steps), config.backend, config.seed, config.use_sobol, use_antithetic
    )
    Z2 = generate_random_numbers(
        (num_paths, num_steps), config.backend,
        config.seed + 1 if config.seed else None, config.use_sobol, use_antithetic
    )

    # Cholesky for correlation
    Z_F = Z1
    Z_sigma = sabr.rho * Z1 + np.sqrt(1 - sabr.rho ** 2) * Z2

    # Initialize paths
    F = xp.zeros((num_paths, num_steps + 1))
    sigma = xp.zeros((num_paths, num_steps + 1))
    F[:, 0] = F0
    sigma[:, 0] = sabr.nu

    # Simulate paths
    for t in range(num_steps):
        F_curr = F[:, t]
        sigma_curr = sigma[:, t]

        # Ensure positive values
        F_curr = xp.maximum(F_curr, 1e-10)
        sigma_curr = xp.maximum(sigma_curr, 1e-10)

        # Forward dynamics: dF = σ * F^β * dW1
        F_beta = xp.power(F_curr, sabr.beta)
        dF = sigma_curr * F_beta * Z_F[:, t] * sqrt_dt
        F[:, t + 1] = xp.maximum(F_curr + dF, 1e-10)

        # Volatility dynamics: dσ = α * σ * dW2
        d_sigma = sabr.alpha * sigma_curr * Z_sigma[:, t] * sqrt_dt
        sigma[:, t + 1] = xp.maximum(sigma_curr + d_sigma, 1e-10)

    return F, sigma


def price_sabr_option(
    F0: float,
    K: float,
    T: float,
    r: float,
    sabr: SABRParams,
    is_call: bool = True,
    config: Optional[SimulationConfig] = None,
    use_hagan: bool = False
) -> PricingResult:
    """
    Price European option under SABR model.

    Args:
        F0: Initial forward price
        K: Strike price
        T: Time to maturity
        r: Risk-free rate (for discounting)
        sabr: SABR parameters
        is_call: True for call, False for put
        config: Simulation configuration
        use_hagan: If True, use Hagan's analytical approximation instead of MC

    Returns:
        PricingResult with option price
    """
    import time
    start_time = time.time()

    if use_hagan and SCIPY_AVAILABLE:
        # Use Hagan's approximation
        impl_vol = sabr.implied_vol_hagan(F0, K, T)
        if is_call:
            price = black_scholes_call(F0, K, r, impl_vol, T)
        else:
            price = black_scholes_put(F0, K, r, impl_vol, T)

        return PricingResult(
            price=price,
            std_error=0.0,  # Analytical
            paths_used=0,
            variance_reduction="analytical_hagan",
            elapsed_time=time.time() - start_time
        )

    # Monte Carlo pricing
    if config is None:
        config = SimulationConfig()

    xp = get_array_module(config.backend)

    # Simulate SABR paths
    F_paths, sigma_paths = simulate_sabr_paths(F0, sabr, T, config)

    # Terminal forward price
    F_T = F_paths[:, -1]

    # Compute payoffs
    if is_call:
        payoffs = xp.maximum(F_T - K, 0.0)
    else:
        payoffs = xp.maximum(K - F_T, 0.0)

    # Discount
    discount = np.exp(-r * T)
    discounted_payoffs = payoffs * discount

    # Statistics
    price = float(to_numpy(xp.mean(discounted_payoffs), config.backend))
    std_error = float(to_numpy(xp.std(discounted_payoffs), config.backend)) / np.sqrt(config.num_paths)

    return PricingResult(
        price=price,
        std_error=std_error,
        paths_used=config.num_paths,
        variance_reduction=config.variance_reduction.value,
        elapsed_time=time.time() - start_time
    )


# =============================================================================
# Multi-Asset / Basket Options
# =============================================================================

def cholesky_decomposition(corr_matrix: np.ndarray) -> np.ndarray:
    """
    Compute Cholesky decomposition for correlated asset simulation.

    Returns lower triangular matrix L such that L @ L.T = corr_matrix
    """
    return np.linalg.cholesky(corr_matrix)


def simulate_multi_asset_gbm(
    S0: np.ndarray,
    r: float,
    sigmas: np.ndarray,
    T: float,
    config: SimulationConfig,
    corr_matrix: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Simulate correlated GBM paths for multiple assets.

    Args:
        S0: Initial prices for each asset (n_assets,)
        r: Risk-free rate
        sigmas: Volatilities for each asset (n_assets,)
        T: Time to maturity
        config: Simulation configuration
        corr_matrix: Correlation matrix (n_assets x n_assets), default: identity

    Returns:
        Array of shape (n_assets, num_paths, num_steps + 1)
    """
    xp = get_array_module(config.backend)
    n_assets = len(S0)
    num_paths = config.num_paths
    num_steps = config.num_steps
    dt = T / num_steps
    sqrt_dt = np.sqrt(dt)

    # Default correlation: independent assets
    if corr_matrix is None:
        corr_matrix = np.eye(n_assets)

    # Cholesky decomposition for correlated draws
    L = cholesky_decomposition(corr_matrix)

    # Generate independent standard normals
    # Shape: (n_assets, num_paths, num_steps)
    Z_indep = np.zeros((n_assets, num_paths, num_steps))
    for i in range(n_assets):
        seed_i = config.seed + i if config.seed else None
        Z_indep[i] = generate_random_numbers(
            (num_paths, num_steps), Backend.NUMPY, seed_i, config.use_sobol, False
        )

    # Apply Cholesky to get correlated draws
    # Z_corr[i, :, t] = sum_j L[i,j] * Z_indep[j, :, t]
    Z_corr = np.zeros_like(Z_indep)
    for t in range(num_steps):
        Z_t = Z_indep[:, :, t]  # (n_assets, num_paths)
        Z_corr[:, :, t] = L @ Z_t

    # Initialize paths
    S = np.zeros((n_assets, num_paths, num_steps + 1))
    for i in range(n_assets):
        S[i, :, 0] = S0[i]

    # Simulate each asset
    for i in range(n_assets):
        drift = (r - 0.5 * sigmas[i] ** 2) * dt
        for t in range(num_steps):
            S[i, :, t + 1] = S[i, :, t] * np.exp(
                drift + sigmas[i] * sqrt_dt * Z_corr[i, :, t]
            )

    if config.backend == Backend.CUPY:
        S = cp.asarray(S)

    return S


def basket_call_payoff(S_all: np.ndarray, K: float, weights: np.ndarray, xp=np) -> np.ndarray:
    """Basket call: max(weighted_sum - K, 0)."""
    # S_all shape: (n_assets, num_paths, num_steps + 1)
    S_T = S_all[:, :, -1]  # (n_assets, num_paths)
    basket_value = xp.sum(weights[:, None] * S_T, axis=0)
    return xp.maximum(basket_value - K, 0.0)


def basket_put_payoff(S_all: np.ndarray, K: float, weights: np.ndarray, xp=np) -> np.ndarray:
    """Basket put: max(K - weighted_sum, 0)."""
    S_T = S_all[:, :, -1]
    basket_value = xp.sum(weights[:, None] * S_T, axis=0)
    return xp.maximum(K - basket_value, 0.0)


def best_of_call_payoff(S_all: np.ndarray, K: float, xp=np) -> np.ndarray:
    """Rainbow call on best performer: max(max(S_T) - K, 0)."""
    S_T = S_all[:, :, -1]
    best = xp.max(S_T, axis=0)
    return xp.maximum(best - K, 0.0)


def worst_of_call_payoff(S_all: np.ndarray, K: float, xp=np) -> np.ndarray:
    """Rainbow call on worst performer: max(min(S_T) - K, 0)."""
    S_T = S_all[:, :, -1]
    worst = xp.min(S_T, axis=0)
    return xp.maximum(worst - K, 0.0)


def best_of_put_payoff(S_all: np.ndarray, K: float, xp=np) -> np.ndarray:
    """Rainbow put on best performer: max(K - max(S_T), 0)."""
    S_T = S_all[:, :, -1]
    best = xp.max(S_T, axis=0)
    return xp.maximum(K - best, 0.0)


def worst_of_put_payoff(S_all: np.ndarray, K: float, xp=np) -> np.ndarray:
    """Rainbow put on worst performer: max(K - min(S_T), 0)."""
    S_T = S_all[:, :, -1]
    worst = xp.min(S_T, axis=0)
    return xp.maximum(K - worst, 0.0)


def spread_call_payoff(S_all: np.ndarray, K: float, xp=np) -> np.ndarray:
    """Spread call: max(S1_T - S2_T - K, 0)."""
    S_T = S_all[:, :, -1]
    spread = S_T[0] - S_T[1]
    return xp.maximum(spread - K, 0.0)


def spread_put_payoff(S_all: np.ndarray, K: float, xp=np) -> np.ndarray:
    """Spread put: max(K - (S1_T - S2_T), 0)."""
    S_T = S_all[:, :, -1]
    spread = S_T[0] - S_T[1]
    return xp.maximum(K - spread, 0.0)


def price_basket_option(
    S0: np.ndarray,
    K: float,
    T: float,
    r: float,
    sigmas: np.ndarray,
    payoff_type: MultiAssetPayoffType,
    config: Optional[SimulationConfig] = None,
    weights: Optional[np.ndarray] = None,
    corr_matrix: Optional[np.ndarray] = None
) -> PricingResult:
    """
    Price multi-asset / basket option using Monte Carlo.

    Args:
        S0: Initial prices (n_assets,)
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        sigmas: Volatilities (n_assets,)
        payoff_type: Type of multi-asset payoff
        config: Simulation configuration
        weights: Asset weights for basket (default: equal)
        corr_matrix: Correlation matrix (default: independent)

    Returns:
        PricingResult with basket option price
    """
    import time
    start_time = time.time()

    S0 = np.asarray(S0)
    sigmas = np.asarray(sigmas)
    n_assets = len(S0)

    if config is None:
        config = SimulationConfig()

    if weights is None:
        weights = np.ones(n_assets) / n_assets
    else:
        weights = np.asarray(weights)
        weights = weights / weights.sum()  # Normalize

    xp = get_array_module(config.backend)

    # Simulate correlated paths
    S_paths = simulate_multi_asset_gbm(S0, r, sigmas, T, config, corr_matrix)

    # Compute payoffs
    payoff_funcs = {
        MultiAssetPayoffType.BASKET_CALL: lambda S, K: basket_call_payoff(S, K, weights, xp),
        MultiAssetPayoffType.BASKET_PUT: lambda S, K: basket_put_payoff(S, K, weights, xp),
        MultiAssetPayoffType.BEST_OF_CALL: lambda S, K: best_of_call_payoff(S, K, xp),
        MultiAssetPayoffType.WORST_OF_CALL: lambda S, K: worst_of_call_payoff(S, K, xp),
        MultiAssetPayoffType.BEST_OF_PUT: lambda S, K: best_of_put_payoff(S, K, xp),
        MultiAssetPayoffType.WORST_OF_PUT: lambda S, K: worst_of_put_payoff(S, K, xp),
        MultiAssetPayoffType.SPREAD_CALL: lambda S, K: spread_call_payoff(S, K, xp),
        MultiAssetPayoffType.SPREAD_PUT: lambda S, K: spread_put_payoff(S, K, xp),
    }

    payoff_func = payoff_funcs.get(payoff_type)
    if payoff_func is None:
        raise ValueError(f"Unknown payoff type: {payoff_type}")

    payoffs = payoff_func(S_paths, K)

    # Discount
    discount = np.exp(-r * T)
    discounted_payoffs = payoffs * discount

    # Statistics
    price = float(to_numpy(xp.mean(discounted_payoffs), config.backend))
    std_error = float(to_numpy(xp.std(discounted_payoffs), config.backend)) / np.sqrt(config.num_paths)

    return PricingResult(
        price=price,
        std_error=std_error,
        paths_used=config.num_paths,
        variance_reduction=config.variance_reduction.value,
        elapsed_time=time.time() - start_time
    )


# =============================================================================
# Local Volatility (Dupire)
# =============================================================================

@dataclass
class LocalVolSurface:
    """
    Local volatility surface for Dupire model.

    The local volatility function σ_loc(S, t) is derived from the
    Dupire formula using market implied volatilities.
    """
    strikes: np.ndarray  # Strike grid
    maturities: np.ndarray  # Maturity grid
    local_vols: np.ndarray  # Local vol matrix (len(maturities) x len(strikes))

    def interpolate(self, S: np.ndarray, t: float, xp=np) -> np.ndarray:
        """
        Interpolate local volatility at given spot and time.

        Uses bilinear interpolation on the vol surface.
        """
        # Find maturity index
        t_idx = np.searchsorted(self.maturities, t)
        t_idx = np.clip(t_idx, 1, len(self.maturities) - 1)

        t_lo, t_hi = self.maturities[t_idx - 1], self.maturities[t_idx]
        t_weight = (t - t_lo) / (t_hi - t_lo + 1e-10)

        # Handle array of spots
        S_np = to_numpy(S, Backend.CUPY if hasattr(xp, 'asnumpy') else Backend.NUMPY)

        result = np.zeros(len(S_np))
        for i, s in enumerate(S_np):
            s_idx = np.searchsorted(self.strikes, s)
            s_idx = np.clip(s_idx, 1, len(self.strikes) - 1)

            s_lo, s_hi = self.strikes[s_idx - 1], self.strikes[s_idx]
            s_weight = (s - s_lo) / (s_hi - s_lo + 1e-10)

            # Bilinear interpolation
            vol_ll = self.local_vols[t_idx - 1, s_idx - 1]
            vol_lh = self.local_vols[t_idx - 1, s_idx]
            vol_hl = self.local_vols[t_idx, s_idx - 1]
            vol_hh = self.local_vols[t_idx, s_idx]

            vol_lo = vol_ll * (1 - s_weight) + vol_lh * s_weight
            vol_hi = vol_hl * (1 - s_weight) + vol_hh * s_weight

            result[i] = vol_lo * (1 - t_weight) + vol_hi * t_weight

        return xp.asarray(result) if hasattr(xp, 'asarray') else result


def dupire_local_vol(
    impl_vols: np.ndarray,
    strikes: np.ndarray,
    maturities: np.ndarray,
    S0: float,
    r: float,
    q: float = 0.0
) -> LocalVolSurface:
    """
    Extract local volatility surface using Dupire's formula.

    Dupire's formula:
        σ_loc²(K,T) = (∂C/∂T + (r-q)K ∂C/∂K + qC) / (0.5 K² ∂²C/∂K²)

    Using the relationship with implied vol:
        σ_loc²(K,T) = (∂σ_impl/∂T + ...) / (...)

    Args:
        impl_vols: Implied vol surface (len(maturities) x len(strikes))
        strikes: Strike grid
        maturities: Maturity grid
        S0: Spot price
        r: Risk-free rate
        q: Dividend yield

    Returns:
        LocalVolSurface object
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("SciPy required for Dupire local vol calculation")

    impl_vols = np.asarray(impl_vols)
    expected_shape = (len(maturities), len(strikes))
    if impl_vols.shape != expected_shape:
        raise ValueError(
            f"impl_vols shape {impl_vols.shape} does not match "
            f"(len(maturities), len(strikes)) = {expected_shape}. "
            "Provide a surface shaped as [maturity, strike]."
        )

    n_T = len(maturities)
    n_K = len(strikes)
    local_vols = np.zeros((n_T, n_K))

    # Compute option prices from implied vols
    prices = np.zeros((n_T, n_K))
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            sigma = impl_vols[i, j]
            prices[i, j] = black_scholes_call(S0, K, r - q, sigma, T)

    # Compute local vols using finite differences
    for i in range(1, n_T - 1):
        for j in range(1, n_K - 1):
            T = maturities[i]
            K = strikes[j]

            # Time derivative (forward difference)
            dT = maturities[i + 1] - maturities[i - 1]
            dC_dT = (prices[i + 1, j] - prices[i - 1, j]) / dT

            # Strike derivatives
            dK = strikes[j + 1] - strikes[j - 1]
            dC_dK = (prices[i, j + 1] - prices[i, j - 1]) / dK

            d2K = (strikes[j + 1] - strikes[j]) * (strikes[j] - strikes[j - 1])
            d2C_dK2 = (prices[i, j + 1] - 2 * prices[i, j] + prices[i, j - 1]) / (d2K + 1e-10)

            # Dupire formula
            numerator = dC_dT + (r - q) * K * dC_dK + q * prices[i, j]
            denominator = 0.5 * K ** 2 * d2C_dK2

            if denominator > 1e-10:
                local_vols[i, j] = np.sqrt(max(numerator / denominator, 0.0))
            else:
                local_vols[i, j] = impl_vols[i, j]  # Fallback

    # Fill boundary values
    local_vols[0, :] = local_vols[1, :]
    local_vols[-1, :] = local_vols[-2, :]
    local_vols[:, 0] = local_vols[:, 1]
    local_vols[:, -1] = local_vols[:, -2]

    return LocalVolSurface(strikes=strikes, maturities=maturities, local_vols=local_vols)


def simulate_local_vol_paths(
    market: MarketData,
    local_vol: LocalVolSurface,
    T: float,
    config: SimulationConfig
) -> np.ndarray:
    """
    Simulate paths under local volatility model.

    dS = (r - q) S dt + σ_loc(S, t) S dW
    """
    xp = get_array_module(config.backend)
    num_paths = config.num_paths
    num_steps = config.num_steps
    dt = T / num_steps
    sqrt_dt = np.sqrt(dt)

    use_antithetic = config.variance_reduction in [
        VarianceReduction.ANTITHETIC,
        VarianceReduction.ANTITHETIC_CV
    ]

    Z = generate_random_numbers(
        (num_paths, num_steps), config.backend, config.seed, config.use_sobol, use_antithetic
    )

    # Initialize paths
    S = xp.zeros((num_paths, num_steps + 1))
    S[:, 0] = market.S0

    # Simulate with local volatility
    for t_idx in range(num_steps):
        t = t_idx * dt
        S_curr = S[:, t_idx]

        # Get local vol for each path
        sigma_loc = local_vol.interpolate(S_curr, t, xp)

        # Euler step
        drift = (market.r - market.q) * S_curr * dt
        diffusion = sigma_loc * S_curr * Z[:, t_idx] * sqrt_dt

        S[:, t_idx + 1] = xp.maximum(S_curr + drift + diffusion, 1e-10)

    return S


def price_local_vol_option(
    market: MarketData,
    K: float,
    T: float,
    local_vol: LocalVolSurface,
    payoff_type: PayoffType,
    config: Optional[SimulationConfig] = None
) -> PricingResult:
    """
    Price option under local volatility model.
    """
    import time
    start_time = time.time()

    if config is None:
        config = SimulationConfig()

    xp = get_array_module(config.backend)

    # Simulate paths
    S_paths = simulate_local_vol_paths(market, local_vol, T, config)

    # Get payoff function
    payoff_func = PAYOFF_FUNCTIONS.get(payoff_type)
    if payoff_func is None:
        raise ValueError(f"Unknown payoff type: {payoff_type}")

    payoffs = payoff_func(S_paths, K, xp)

    # Discount
    discount = np.exp(-market.r * T)
    discounted_payoffs = payoffs * discount

    price = float(to_numpy(xp.mean(discounted_payoffs), config.backend))
    std_error = float(to_numpy(xp.std(discounted_payoffs), config.backend)) / np.sqrt(config.num_paths)

    return PricingResult(
        price=price,
        std_error=std_error,
        paths_used=config.num_paths,
        variance_reduction=config.variance_reduction.value,
        elapsed_time=time.time() - start_time
    )


# =============================================================================
# Rough Volatility (Rough Heston)
# =============================================================================

@dataclass
class RoughHestonParams:
    """
    Rough Heston model parameters.

    The rough Heston model replaces the standard Brownian motion driving
    variance with a fractional Brownian motion with Hurst parameter H < 0.5.

    Key characteristics:
    - H < 0.5: Rougher paths than standard Brownian motion
    - Better fit to short-dated implied volatility smiles
    - Captures the term structure of ATM skew
    """
    v0: float  # Initial variance
    theta: float  # Long-term variance (mean level under physical measure)
    lambda_: float  # Mean reversion speed (equivalent to kappa)
    nu: float  # Vol of vol (equivalent to xi)
    rho: float  # Correlation
    H: float  # Hurst parameter (typically 0.05-0.2 for rough vol)

    def __post_init__(self):
        if not 0 < self.H < 0.5:
            raise ValueError(f"H must be in (0, 0.5) for rough volatility, got {self.H}")
        if self.v0 < 0:
            raise ValueError(f"v0 must be non-negative, got {self.v0}")
        if self.theta < 0:
            raise ValueError(f"theta must be non-negative, got {self.theta}")
        if not -1 <= self.rho <= 1:
            raise ValueError(f"rho must be in [-1, 1], got {self.rho}")


def generate_fbm_cholesky(
    H: float,
    n: int,
    dt: float,
    num_paths: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate fractional Brownian motion using Cholesky decomposition.

    This is the exact method but O(n²) in memory and O(n³) in computation.
    For large n, use the hybrid scheme instead.

    Args:
        H: Hurst parameter
        n: Number of time steps
        dt: Time step size
        num_paths: Number of paths
        seed: Random seed

    Returns:
        fBm increments of shape (num_paths, n)
    """
    import math

    # Compute covariance matrix of fBm increments
    # Cov(B^H_s, B^H_t) = 0.5 * (|s|^{2H} + |t|^{2H} - |s-t|^{2H})
    times = np.arange(1, n + 1) * dt
    cov = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            ti, tj = times[i], times[j]
            cov[i, j] = 0.5 * (ti ** (2*H) + tj ** (2*H) - abs(ti - tj) ** (2*H))

    # Cholesky decomposition
    L = np.linalg.cholesky(cov + 1e-10 * np.eye(n))

    # Generate independent normals and correlate
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((num_paths, n))

    # fBm values at grid points
    fBm = Z @ L.T

    # Return increments
    increments = np.zeros_like(fBm)
    increments[:, 0] = fBm[:, 0]
    increments[:, 1:] = np.diff(fBm, axis=1)

    return increments


def generate_fbm_hybrid(
    H: float,
    n: int,
    dt: float,
    num_paths: int,
    seed: Optional[int] = None,
    n_approx: int = 100
) -> np.ndarray:
    """
    Generate fractional Brownian motion using the hybrid scheme.

    This is more efficient for large n, using:
    - Exact simulation for short-range correlations
    - Power-law approximation for long-range correlations

    Based on Bennedsen, Lunde & Pakkanen (2017).
    """
    rng = np.random.default_rng(seed)

    # For rough paths (H < 0.5), memory effects decay quickly
    # Use truncated kernel approximation

    # Kernel weights for Volterra representation
    # V_t = int_0^t K(t-s) dW_s
    # K(t) = c_H * t^{H-0.5}

    import math
    c_H = np.sqrt(2 * H) / math.gamma(H + 0.5)

    # Generate standard BM increments
    dW = rng.standard_normal((num_paths, n)) * np.sqrt(dt)

    # Approximate Volterra integral using Riemann-Liouville discretization
    kernel_weights = np.zeros(min(n, n_approx))
    for k in range(len(kernel_weights)):
        if k == 0:
            kernel_weights[k] = dt ** (H + 0.5) / (H + 0.5)
        else:
            kernel_weights[k] = c_H * ((k * dt) ** (H - 0.5)) * dt

    # Convolve with BM increments (truncated)
    fBm_increments = np.zeros((num_paths, n))

    for t in range(n):
        k_max = min(t + 1, len(kernel_weights))
        for k in range(k_max):
            fBm_increments[:, t] += kernel_weights[k] * dW[:, t - k]

    return fBm_increments


def simulate_rough_heston_paths(
    market: MarketData,
    params: RoughHestonParams,
    T: float,
    config: SimulationConfig,
    use_hybrid: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate rough Heston model paths.

    The variance process follows:
        V_t = V_0 + (1/Γ(H+0.5)) ∫_0^t (t-s)^{H-0.5} λ(θ - V_s) ds
              + (ν/Γ(H+0.5)) ∫_0^t (t-s)^{H-0.5} √V_s dW^V_s

    We use the Euler scheme with fBm increments.

    Args:
        market: Market data
        params: Rough Heston parameters
        T: Time to maturity
        config: Simulation configuration
        use_hybrid: Use hybrid scheme for fBm (faster)

    Returns:
        Tuple of (S_paths, V_paths)
    """
    num_paths = config.num_paths
    num_steps = config.num_steps
    dt = T / num_steps
    sqrt_dt = np.sqrt(dt)

    # Generate standard and fractional BM
    rng = np.random.default_rng(config.seed)
    dW_S = rng.standard_normal((num_paths, num_steps)) * sqrt_dt

    if use_hybrid:
        dB_H = generate_fbm_hybrid(
            params.H, num_steps, dt, num_paths,
            config.seed + 1 if config.seed else None
        )
    else:
        dB_H = generate_fbm_cholesky(
            params.H, num_steps, dt, num_paths,
            config.seed + 1 if config.seed else None
        )

    # Correlate the increments
    dW_V = params.rho * dW_S + np.sqrt(1 - params.rho ** 2) * dB_H

    # Initialize paths
    S = np.zeros((num_paths, num_steps + 1))
    V = np.zeros((num_paths, num_steps + 1))
    S[:, 0] = market.S0
    V[:, 0] = params.v0

    # Gamma function for fractional calculus
    import math
    gamma_H = math.gamma(params.H + 0.5)

    # Simulate paths
    for t in range(num_steps):
        V_curr = np.maximum(V[:, t], 0.0)
        sqrt_V = np.sqrt(V_curr)

        # Variance dynamics (simplified Euler for rough Heston)
        # Note: Full rough Heston requires Volterra integral which is expensive
        # This is an approximation using fractional noise
        dV = params.lambda_ * (params.theta - V_curr) * dt + params.nu * sqrt_V * dW_V[:, t]
        V[:, t + 1] = np.maximum(V_curr + dV, 0.0)

        # Stock dynamics
        dS = (market.r - market.q - 0.5 * V_curr) * dt + sqrt_V * dW_S[:, t]
        S[:, t + 1] = S[:, t] * np.exp(dS)

    return S, V


def price_rough_heston_option(
    market: MarketData,
    K: float,
    T: float,
    params: RoughHestonParams,
    payoff_type: PayoffType,
    config: Optional[SimulationConfig] = None
) -> PricingResult:
    """
    Price option under rough Heston model.
    """
    import time
    start_time = time.time()

    if config is None:
        config = SimulationConfig()

    # Simulate paths
    S_paths, V_paths = simulate_rough_heston_paths(market, params, T, config)

    # Get payoff
    payoff_func = PAYOFF_FUNCTIONS.get(payoff_type)
    if payoff_func is None:
        raise ValueError(f"Unknown payoff type: {payoff_type}")

    payoffs = payoff_func(S_paths, K, np)

    # Discount
    discount = np.exp(-market.r * T)
    discounted_payoffs = payoffs * discount

    price = float(np.mean(discounted_payoffs))
    std_error = float(np.std(discounted_payoffs)) / np.sqrt(config.num_paths)

    return PricingResult(
        price=price,
        std_error=std_error,
        paths_used=config.num_paths,
        variance_reduction=config.variance_reduction.value,
        elapsed_time=time.time() - start_time
    )


# =============================================================================
# Advanced Variance Reduction
# =============================================================================

def importance_sampling_optimal_drift(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    is_call: bool = True
) -> float:
    """
    Compute optimal drift for importance sampling.

    For OTM options, shift the drift to make ITM more likely,
    then adjust with likelihood ratio.

    The optimal drift shift μ* minimizes variance of the IS estimator.
    For calls: μ* ≈ (log(K/S0) + rT) / T
    """
    log_moneyness = np.log(K / S0)

    if is_call:
        # For OTM calls (K > S0), shift drift up
        if log_moneyness > 0:
            return (log_moneyness + 0.5 * sigma ** 2 * T) / T
        else:
            return 0.0
    else:
        # For OTM puts (K < S0), shift drift down
        if log_moneyness < 0:
            return (log_moneyness - 0.5 * sigma ** 2 * T) / T
        else:
            return 0.0


def simulate_gbm_importance_sampling(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    num_paths: int,
    num_steps: int,
    drift_shift: float,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate GBM paths with importance sampling.

    Returns:
        Tuple of (S_paths, likelihood_ratios)
    """
    rng = np.random.default_rng(seed)
    dt = T / num_steps
    sqrt_dt = np.sqrt(dt)

    Z = rng.standard_normal((num_paths, num_steps))

    # Shifted drift
    drift_original = (r - 0.5 * sigma ** 2) * dt
    drift_shifted = (r - 0.5 * sigma ** 2 + drift_shift) * dt

    # Simulate paths under shifted measure
    log_S = np.zeros((num_paths, num_steps + 1))
    log_S[:, 0] = np.log(S0)

    for t in range(num_steps):
        log_S[:, t + 1] = log_S[:, t] + drift_shifted + sigma * sqrt_dt * Z[:, t]

    S_paths = np.exp(log_S)

    # Compute likelihood ratio (Radon-Nikodym derivative)
    # L = exp(-μ* ∫ dW - 0.5 μ*² T)
    # For discrete: L = exp(-μ*/σ * Σ Z_t * sqrt_dt - 0.5 (μ*/σ)² * T)

    sum_Z = np.sum(Z, axis=1) * sqrt_dt
    mu_over_sigma = drift_shift / sigma

    likelihood_ratios = np.exp(
        -mu_over_sigma * sum_Z - 0.5 * mu_over_sigma ** 2 * T
    )

    return S_paths, likelihood_ratios


def price_with_importance_sampling(
    market: MarketData,
    K: float,
    T: float,
    sigma: float,
    is_call: bool = True,
    num_paths: int = 100000,
    num_steps: int = 252,
    seed: Optional[int] = None
) -> PricingResult:
    """
    Price European option using importance sampling.
    """
    import time
    start_time = time.time()

    # Compute optimal drift
    drift_shift = importance_sampling_optimal_drift(
        market.S0, K, market.r, sigma, T, is_call
    )

    # Simulate with IS
    S_paths, likelihood_ratios = simulate_gbm_importance_sampling(
        market.S0, market.r, sigma, T, num_paths, num_steps, drift_shift, seed
    )

    # Compute payoffs
    if is_call:
        payoffs = np.maximum(S_paths[:, -1] - K, 0.0)
    else:
        payoffs = np.maximum(K - S_paths[:, -1], 0.0)

    # Apply likelihood ratio and discount
    discount = np.exp(-market.r * T)
    adjusted_payoffs = payoffs * likelihood_ratios * discount

    price = float(np.mean(adjusted_payoffs))
    std_error = float(np.std(adjusted_payoffs)) / np.sqrt(num_paths)

    return PricingResult(
        price=price,
        std_error=std_error,
        paths_used=num_paths,
        variance_reduction="importance_sampling",
        elapsed_time=time.time() - start_time
    )


def stratified_sampling_uniform(
    num_samples: int,
    num_strata: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate stratified uniform samples.

    Divides [0,1] into equal strata and samples uniformly within each.
    """
    rng = np.random.default_rng(seed)

    samples_per_stratum = num_samples // num_strata
    samples = np.zeros(num_samples)

    for i in range(num_strata):
        lo = i / num_strata
        hi = (i + 1) / num_strata
        n = samples_per_stratum if i < num_strata - 1 else num_samples - i * samples_per_stratum
        samples[i * samples_per_stratum:i * samples_per_stratum + n] = \
            rng.uniform(lo, hi, n)

    # Shuffle to avoid systematic patterns
    rng.shuffle(samples)

    return samples


def latin_hypercube_sampling(
    num_samples: int,
    num_dimensions: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate Latin Hypercube Samples.

    Each dimension is divided into n equal strata, with exactly
    one sample in each stratum per dimension.

    Returns:
        Array of shape (num_samples, num_dimensions) with values in [0, 1]
    """
    rng = np.random.default_rng(seed)

    samples = np.zeros((num_samples, num_dimensions))

    for d in range(num_dimensions):
        # Generate one sample per stratum
        perm = rng.permutation(num_samples)
        for i in range(num_samples):
            lo = perm[i] / num_samples
            hi = (perm[i] + 1) / num_samples
            samples[i, d] = rng.uniform(lo, hi)

    return samples


def simulate_gbm_stratified(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    num_paths: int,
    num_steps: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Simulate GBM paths using Latin Hypercube Sampling.
    """
    if not SCIPY_AVAILABLE:
        # Fallback to standard sampling
        return simulate_gbm_paths(
            S0, r, sigma, T, num_paths, num_steps,
            SimulationConfig(num_paths=num_paths, num_steps=num_steps, seed=seed)
        )

    # Generate LHS samples
    U = latin_hypercube_sampling(num_paths, num_steps, seed)

    # Convert to standard normal
    Z = norm.ppf(np.clip(U, 1e-10, 1 - 1e-10))

    # Simulate paths
    dt = T / num_steps
    sqrt_dt = np.sqrt(dt)
    drift = (r - 0.5 * sigma ** 2) * dt

    log_S = np.zeros((num_paths, num_steps + 1))
    log_S[:, 0] = np.log(S0)
    log_S[:, 1:] = np.cumsum(drift + sigma * sqrt_dt * Z, axis=1) + np.log(S0)

    return np.exp(log_S)


def price_with_stratified_sampling(
    market: MarketData,
    K: float,
    T: float,
    sigma: float,
    payoff_type: PayoffType,
    num_paths: int = 100000,
    num_steps: int = 252,
    seed: Optional[int] = None
) -> PricingResult:
    """
    Price option using stratified/LHS sampling.
    """
    import time
    start_time = time.time()

    # Simulate with LHS
    S_paths = simulate_gbm_stratified(
        market.S0, market.r, sigma, T, num_paths, num_steps, seed
    )

    # Compute payoffs
    payoff_func = PAYOFF_FUNCTIONS.get(payoff_type)
    if payoff_func is None:
        raise ValueError(f"Unknown payoff type: {payoff_type}")

    payoffs = payoff_func(S_paths, K, np)

    # Discount
    discount = np.exp(-market.r * T)
    discounted_payoffs = payoffs * discount

    price = float(np.mean(discounted_payoffs))
    std_error = float(np.std(discounted_payoffs)) / np.sqrt(num_paths)

    return PricingResult(
        price=price,
        std_error=std_error,
        paths_used=num_paths,
        variance_reduction="stratified_lhs",
        elapsed_time=time.time() - start_time
    )


def moment_matching_adjustment(
    S_paths: np.ndarray,
    S0: float,
    r: float,
    T: float
) -> np.ndarray:
    """
    Apply moment matching to ensure simulated paths have correct mean.

    Adjusts terminal values so that E[S_T] = S0 * exp(rT).
    """
    expected_terminal = S0 * np.exp(r * T)
    actual_mean = np.mean(S_paths[:, -1])

    # Scale terminal values
    adjustment = expected_terminal / actual_mean

    S_adjusted = S_paths.copy()
    S_adjusted[:, -1] *= adjustment

    return S_adjusted


# =============================================================================
# Pathwise Greeks (AAD-style)
# =============================================================================

def pathwise_delta(
    S_paths: np.ndarray,
    K: float,
    r: float,
    T: float,
    S0: float,
    sigma: float,
    is_call: bool = True
) -> float:
    """
    Compute Delta using pathwise differentiation.

    For European call: Δ = E[exp(-rT) * 1{S_T > K} * S_T / S0]
    This is more efficient than bump-and-reprice.

    Args:
        S_paths: Simulated paths (num_paths, num_steps + 1)
        K: Strike
        r: Risk-free rate
        T: Maturity
        S0: Initial spot
        sigma: Volatility
        is_call: True for call, False for put

    Returns:
        Pathwise Delta estimate
    """
    S_T = S_paths[:, -1]
    discount = np.exp(-r * T)

    if is_call:
        # dPayoff/dS0 = 1{S_T > K} * S_T / S0
        indicator = S_T > K
        delta_paths = discount * indicator * S_T / S0
    else:
        # dPayoff/dS0 = -1{S_T < K} * S_T / S0  (for put it's the negative)
        indicator = S_T < K
        delta_paths = -discount * indicator * S_T / S0

    return float(np.mean(delta_paths))


def pathwise_vega(
    S_paths: np.ndarray,
    Z: np.ndarray,
    K: float,
    r: float,
    T: float,
    S0: float,
    sigma: float,
    is_call: bool = True
) -> float:
    """
    Compute Vega using pathwise differentiation.

    For GBM: dS_T/dσ = S_T * (-σT + √T * Z_total)
    where Z_total is the cumulative normal random variable.

    Args:
        S_paths: Simulated paths
        Z: Random numbers used for simulation (num_paths, num_steps)
        K: Strike
        r: Risk-free rate
        T: Maturity
        S0: Initial spot
        sigma: Volatility
        is_call: True for call

    Returns:
        Pathwise Vega estimate (per 1% vol change)
    """
    S_T = S_paths[:, -1]
    num_steps = Z.shape[1]
    dt = T / num_steps
    sqrt_dt = np.sqrt(dt)

    # Cumulative Z contribution
    Z_total = np.sum(Z, axis=1) * sqrt_dt / np.sqrt(T)  # Normalized

    # dS_T/dσ = S_T * (-σT + √T * Z_total)
    dS_dsigma = S_T * (-sigma * T + np.sqrt(T) * Z_total)

    discount = np.exp(-r * T)

    if is_call:
        indicator = S_T > K
    else:
        indicator = S_T < K

    # dPayoff/dσ = indicator * dS_dsigma (times -1 for put)
    sign = 1 if is_call else -1
    vega_paths = sign * discount * indicator * dS_dsigma

    # Scale to per 1% vol change
    return float(np.mean(vega_paths)) * 0.01


def pathwise_gamma(
    S_paths: np.ndarray,
    Z: np.ndarray,
    K: float,
    r: float,
    T: float,
    S0: float,
    sigma: float,
    is_call: bool = True
) -> float:
    """
    Compute Gamma using likelihood ratio method (not pure pathwise).

    Gamma is difficult with pure pathwise since it involves Dirac delta.
    We use the likelihood ratio representation instead.

    Γ = E[exp(-rT) * Payoff * (Z² - 1) / (S0² σ² T)]

    where Z is the standard normal driving the terminal price.
    """
    S_T = S_paths[:, -1]
    num_steps = Z.shape[1]
    dt = T / num_steps
    sqrt_dt = np.sqrt(dt)

    # Total Z (approximately standard normal for small dt)
    Z_total = np.sum(Z, axis=1) * sqrt_dt / np.sqrt(T)

    discount = np.exp(-r * T)

    if is_call:
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)

    # Likelihood ratio weight for Gamma
    lr_weight = (Z_total ** 2 - 1) / (S0 ** 2 * sigma ** 2 * T)

    gamma_paths = discount * payoffs * lr_weight

    return float(np.mean(gamma_paths))


def calculate_pathwise_greeks(
    market: MarketData,
    K: float,
    T: float,
    sigma: float,
    is_call: bool = True,
    num_paths: int = 100000,
    num_steps: int = 252,
    seed: Optional[int] = None
) -> Dict[str, float]:
    """
    Calculate Delta, Gamma, and Vega using pathwise/LR methods.

    More efficient than bump-and-reprice for large portfolios.
    """
    rng = np.random.default_rng(seed)
    dt = T / num_steps
    sqrt_dt = np.sqrt(dt)

    # Generate standard normals once. Keep an explicit copy for Greeks:
    # on newer CPython/Numpy combinations chained arithmetic can reuse/mutate
    # temporaries, so path construction should not alias the Greeks input.
    Z = rng.standard_normal((num_paths, num_steps))
    Z_for_greeks = Z.copy()

    drift = (market.r - 0.5 * sigma ** 2) * dt
    log_S = np.zeros((num_paths, num_steps + 1))
    log_S[:, 0] = np.log(market.S0)
    log_S[:, 1:] = np.cumsum(drift + sigma * sqrt_dt * Z, axis=1) + np.log(market.S0)
    S_paths = np.exp(log_S)

    # Calculate Greeks
    delta = pathwise_delta(S_paths, K, market.r, T, market.S0, sigma, is_call)
    vega = pathwise_vega(S_paths, Z_for_greeks, K, market.r, T, market.S0, sigma, is_call)
    gamma = pathwise_gamma(S_paths, Z_for_greeks, K, market.r, T, market.S0, sigma, is_call)

    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega
    }


# =============================================================================
# Batch Pricing
# =============================================================================

@dataclass
class BatchOption:
    """Specification for a single option in a batch."""
    K: float  # Strike
    T: float  # Maturity
    payoff_type: PayoffType
    sigma: Optional[float] = None  # For GBM
    heston: Optional[HestonParams] = None
    jump: Optional[JumpParams] = None
    barrier: Optional[BarrierParams] = None


@dataclass
class BatchPricingResult:
    """Result of batch pricing."""
    prices: np.ndarray
    std_errors: np.ndarray
    elapsed_time: float
    num_options: int


def price_batch_options(
    market: MarketData,
    options: List[BatchOption],
    config: Optional[SimulationConfig] = None,
    parallel: bool = True
) -> BatchPricingResult:
    """
    Price multiple options in a batch for efficiency.

    When options share the same underlying model (GBM/Heston/Bates),
    paths are simulated once and reused for all payoffs.

    Args:
        market: Market data (common for all options)
        options: List of option specifications
        config: Simulation configuration
        parallel: Use parallel path reuse (more efficient)

    Returns:
        BatchPricingResult with arrays of prices and errors
    """
    import time
    start_time = time.time()

    if config is None:
        config = SimulationConfig()

    n_options = len(options)
    prices = np.zeros(n_options)
    std_errors = np.zeros(n_options)

    if not parallel or n_options == 1:
        # Simple sequential pricing
        for i, opt in enumerate(options):
            result = price_option(
                market=market, K=opt.K, T=opt.T,
                payoff_type=opt.payoff_type,
                heston=opt.heston, jump=opt.jump,
                barrier=opt.barrier, config=config,
                sigma=opt.sigma
            )
            prices[i] = result.price
            std_errors[i] = result.std_error
    else:
        # Group options by model and maturity for path reuse
        groups = {}
        for i, opt in enumerate(options):
            # Create group key based on model parameters
            if opt.heston is not None:
                model_key = ('heston', opt.T,
                             opt.heston.v0, opt.heston.kappa,
                             opt.heston.theta, opt.heston.xi, opt.heston.rho)
                if opt.jump is not None and opt.jump.lambda_j > 0:
                    model_key = model_key + (
                        opt.jump.lambda_j,
                        opt.jump.mu_j,
                        opt.jump.sigma_j,
                    )
            else:
                model_key = ('gbm', opt.T, opt.sigma)

            if model_key not in groups:
                groups[model_key] = []
            groups[model_key].append(i)

        xp = get_array_module(config.backend)

        # Process each group
        for model_key, indices in groups.items():
            first_opt = options[indices[0]]

            # Simulate paths once for the group
            if first_opt.heston is not None:
                if first_opt.jump is not None and first_opt.jump.lambda_j > 0:
                    S_paths, _ = simulate_bates_paths(
                        market, first_opt.heston, first_opt.jump, first_opt.T, config
                    )
                else:
                    S_paths, _ = simulate_heston_paths(
                        market, first_opt.heston, first_opt.T, config
                    )
            else:
                S_paths = simulate_gbm_paths(
                    market.S0, market.r, first_opt.sigma,
                    first_opt.T, config.num_paths, config.num_steps, config
                )

            # Compute payoffs for all options in group
            for idx in indices:
                opt = options[idx]
                payoff_func = PAYOFF_FUNCTIONS.get(opt.payoff_type)

                if payoff_func is None:
                    raise ValueError(f"Unknown payoff type: {opt.payoff_type}")

                if opt.payoff_type in BARRIER_PAYOFF_TYPES:
                    if opt.barrier is None:
                        raise ValueError(
                            f"Barrier parameters are required for payoff type '{opt.payoff_type.value}'"
                        )
                    payoffs = payoff_func(S_paths, opt.K, opt.barrier.barrier,
                                          opt.barrier.rebate, xp)
                else:
                    payoffs = payoff_func(S_paths, opt.K, xp)

                discount = np.exp(-market.r * opt.T)
                discounted_payoffs = payoffs * discount

                prices[idx] = float(to_numpy(xp.mean(discounted_payoffs), config.backend))
                std_errors[idx] = float(to_numpy(xp.std(discounted_payoffs), config.backend)) / \
                                  np.sqrt(config.num_paths)

    elapsed = time.time() - start_time

    return BatchPricingResult(
        prices=prices,
        std_errors=std_errors,
        elapsed_time=elapsed,
        num_options=n_options
    )


def price_portfolio_greeks(
    market: MarketData,
    options: List[BatchOption],
    positions: List[float],
    config: Optional[SimulationConfig] = None
) -> Dict[str, float]:
    """
    Calculate portfolio-level Greeks.

    Args:
        market: Market data
        options: List of option specifications
        positions: Position sizes (positive = long, negative = short)
        config: Simulation configuration

    Returns:
        Dictionary with portfolio delta, gamma, vega
    """
    if config is None:
        config = SimulationConfig()

    portfolio_delta = 0.0
    portfolio_gamma = 0.0
    portfolio_vega = 0.0

    for opt, position in zip(options, positions):
        if opt.sigma is None:
            continue  # Skip stochastic vol for now

        is_call = 'call' in opt.payoff_type.value.lower()

        greeks = calculate_pathwise_greeks(
            market=market, K=opt.K, T=opt.T, sigma=opt.sigma,
            is_call=is_call, num_paths=config.num_paths // 10,  # Fewer paths for Greeks
            num_steps=config.num_steps
        )

        portfolio_delta += position * greeks['delta']
        portfolio_gamma += position * greeks['gamma']
        portfolio_vega += position * greeks['vega']

    return {
        'delta': portfolio_delta,
        'gamma': portfolio_gamma,
        'vega': portfolio_vega
    }


# =============================================================================
# Notebook-Friendly Wrapper Functions
# =============================================================================

def price_european(
    market: MarketData,
    K: float,
    T: float,
    sigma: float,
    payoff_type: PayoffType,
    config: SimulationConfig,
    heston: Optional[HestonParams] = None,
    jump: Optional[JumpParams] = None
) -> PricingResult:
    """Notebook-friendly wrapper for European option pricing."""
    return price_option(
        market=market,
        K=K,
        T=T,
        payoff_type=payoff_type,
        config=config,
        sigma=sigma,
        heston=heston,
        jump=jump
    )


def sabr_implied_volatility(S0: float, K: float, T: float, sabr: SABRParams) -> float:
    """Get SABR implied volatility using Hagan's approximation."""
    return sabr.implied_vol_hagan(S0, K, T)


def black_scholes_price(S0: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
    """Combined Black-Scholes pricing for call or put."""
    if option_type.lower() == 'call':
        return black_scholes_call(S0, K, r, sigma, T)
    else:
        return black_scholes_put(S0, K, r, sigma, T)


def black_scholes_greeks(S0: float, K: float, T: float, r: float, q: float, sigma: float) -> Dict[str, float]:
    """Calculate Black-Scholes Greeks for a call option."""
    from scipy.stats import norm

    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Call price
    price = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    # Greeks
    delta = np.exp(-q * T) * norm.cdf(d1)
    gamma = np.exp(-q * T) * norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
    vega = S0 * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    theta = (-S0 * sigma * np.exp(-q * T) * norm.pdf(d1) / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * norm.cdf(d2)
             + q * S0 * np.exp(-q * T) * norm.cdf(d1))
    rho = K * T * np.exp(-r * T) * norm.cdf(d2)

    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta / 365,  # Daily theta
        'rho': rho
    }


@dataclass
class MultiAssetParams:
    """Parameters for multi-asset options (notebook-friendly interface)."""
    spots: np.ndarray  # Initial spot prices
    vols: np.ndarray  # Volatilities
    correlations: np.ndarray  # Correlation matrix
    weights: np.ndarray  # Portfolio weights
    r: float  # Risk-free rate
    q: np.ndarray  # Dividend yields


def simulate_correlated_assets(
    params: MultiAssetParams,
    T: float,
    config: SimulationConfig
) -> np.ndarray:
    """Simulate correlated asset paths."""
    num_assets = len(params.spots)
    num_paths = config.num_paths
    num_steps = config.num_steps
    dt = T / num_steps

    # Cholesky decomposition for correlation
    L = np.linalg.cholesky(params.correlations)

    # Generate correlated random numbers
    np.random.seed(config.seed)
    Z = np.random.standard_normal((num_paths, num_steps, num_assets))
    corr_Z = np.einsum('ijk,lk->ijl', Z, L)

    # Simulate paths
    paths = np.zeros((num_paths, num_steps + 1, num_assets))
    paths[:, 0, :] = params.spots

    for t in range(num_steps):
        for i in range(num_assets):
            drift = (params.r - params.q[i] - 0.5 * params.vols[i]**2) * dt
            diffusion = params.vols[i] * np.sqrt(dt) * corr_Z[:, t, i]
            paths[:, t + 1, i] = paths[:, t, i] * np.exp(drift + diffusion)

    return paths


def price_multi_asset_option(
    params: MultiAssetParams,
    K: float,
    T: float,
    payoff_type: MultiAssetPayoffType,
    config: SimulationConfig
) -> PricingResult:
    """Price multi-asset/basket options."""
    import time
    start_time = time.time()

    # Simulate correlated paths
    paths = simulate_correlated_assets(params, T, config)
    S_T = paths[:, -1, :]  # Terminal values

    # Calculate payoffs based on type
    if payoff_type == MultiAssetPayoffType.BASKET_CALL:
        basket = np.sum(S_T * params.weights, axis=1)
        payoffs = np.maximum(basket - K, 0)
    elif payoff_type == MultiAssetPayoffType.BASKET_PUT:
        basket = np.sum(S_T * params.weights, axis=1)
        payoffs = np.maximum(K - basket, 0)
    elif payoff_type == MultiAssetPayoffType.BEST_OF_CALL:
        best = np.max(S_T, axis=1)
        payoffs = np.maximum(best - K, 0)
    elif payoff_type == MultiAssetPayoffType.WORST_OF_PUT:
        worst = np.min(S_T, axis=1)
        payoffs = np.maximum(K - worst, 0)
    else:
        raise ValueError(f"Unsupported payoff type: {payoff_type}")

    # Discount and compute statistics
    discount = np.exp(-params.r * T)
    discounted_payoffs = discount * payoffs

    price = float(np.mean(discounted_payoffs))
    std_error = float(np.std(discounted_payoffs)) / np.sqrt(config.num_paths)

    return PricingResult(
        price=price,
        std_error=std_error,
        paths_used=config.num_paths,
        variance_reduction="none",
        elapsed_time=time.time() - start_time
    )


# =============================================================================
# Example Usage and Tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Monte Carlo Option Pricing Library - Demo")
    print("=" * 60)

    # Test 1: GBM European Option (validate against BS)
    print("\n1. European Call (GBM) - Validation against Black-Scholes")
    print("-" * 60)

    market = MarketData(S0=100.0, r=0.05)
    config = SimulationConfig(
        num_paths=100000,
        num_steps=252,
        variance_reduction=VarianceReduction.ANTITHETIC
    )

    result = price_option(
        market=market,
        K=100.0,
        T=1.0,
        payoff_type=PayoffType.EUROPEAN_CALL,
        config=config,
        sigma=0.2
    )

    if SCIPY_AVAILABLE:
        bs_price = black_scholes_call(100.0, 100.0, 0.05, 0.2, 1.0)
        print(f"MC Price:  ${result.price:.4f} (SE: {result.std_error:.4f})")
        print(f"BS Price:  ${bs_price:.4f}")
        print(f"Error:     ${abs(result.price - bs_price):.4f}")
    else:
        print(f"MC Price:  ${result.price:.4f} (SE: {result.std_error:.4f})")

    # Test 2: Asian Put with Control Variates
    print("\n2. Asian Put with Control Variates")
    print("-" * 60)

    result_cv = price_asian_put(
        S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0,
        num_paths=100000
    )

    print(f"Price:     ${result_cv.price:.4f} (SE: {result_cv.std_error:.4f})")
    if result_cv.control_variate_beta:
        print(f"CV Beta:   {result_cv.control_variate_beta:.4f}")
    print(f"Time:      {result_cv.elapsed_time:.3f}s")

    # Test 3: Barrier Option
    print("\n3. Down-and-Out Put")
    print("-" * 60)

    result_barrier = price_barrier_option(
        S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0,
        barrier=80.0, barrier_type="down_out_put"
    )

    print(f"Price:     ${result_barrier.price:.4f} (SE: {result_barrier.std_error:.4f})")

    # Test 4: Bates Model (Heston + Jumps)
    print("\n4. Asian Put under Bates Model")
    print("-" * 60)

    heston = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
    jump = JumpParams(lambda_j=0.1, mu_j=-0.05, sigma_j=0.1)

    result_bates = price_asian_put(
        S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0,
        num_paths=50000,
        heston=heston,
        jump=jump
    )

    print(f"Price:     ${result_bates.price:.4f} (SE: {result_bates.std_error:.4f})")
    print(f"Feller:    {'Satisfied' if heston.feller_condition else 'Violated'}")

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
