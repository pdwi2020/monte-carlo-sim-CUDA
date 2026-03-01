"""
Model Calibration Module

This module implements calibration routines for:
- Heston stochastic volatility model
- SABR model
- Local volatility surface

Calibration minimizes the difference between model-implied and market-observed
option prices or implied volatilities.

References:
    - Gatheral, J. (2006). The Volatility Surface.
    - Andersen, L. & Piterbarg, V. (2010). Interest Rate Modeling.
"""

import numpy as np
from typing import Optional, List, Dict, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings

try:
    from scipy.optimize import minimize, differential_evolution, basinhopping
    from scipy.interpolate import RectBivariateSpline, interp2d
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    minimize = None
    differential_evolution = None
    basinhopping = None
    RectBivariateSpline = None
    interp2d = None
    stats = None
    SCIPY_AVAILABLE = False


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MarketOption:
    """Market option data for calibration."""
    strike: float
    maturity: float  # Time to expiry in years
    market_price: Optional[float] = None
    market_iv: Optional[float] = None  # Implied volatility
    option_type: str = "call"  # "call" or "put"
    bid: Optional[float] = None
    ask: Optional[float] = None

    @property
    def mid_price(self) -> Optional[float]:
        if self.bid is not None and self.ask is not None:
            return 0.5 * (self.bid + self.ask)
        return self.market_price


@dataclass
class VolSurface:
    """Implied volatility surface."""
    strikes: np.ndarray
    maturities: np.ndarray
    ivs: np.ndarray  # Shape: (len(maturities), len(strikes))
    spot: float
    rate: float

    def interpolate(self, K: float, T: float) -> float:
        """Interpolate implied vol at given strike and maturity."""
        if SCIPY_AVAILABLE:
            spline = RectBivariateSpline(
                self.maturities, self.strikes, self.ivs
            )
            return float(spline(T, K)[0, 0])
        else:
            # Simple bilinear interpolation fallback
            T_idx = np.searchsorted(self.maturities, T)
            K_idx = np.searchsorted(self.strikes, K)
            T_idx = np.clip(T_idx, 1, len(self.maturities) - 1)
            K_idx = np.clip(K_idx, 1, len(self.strikes) - 1)
            return self.ivs[T_idx, K_idx]


@dataclass
class CalibrationResult:
    """Result of model calibration."""
    success: bool
    parameters: Dict[str, float]
    objective_value: float
    rmse: float  # Root mean squared error
    num_iterations: int
    calibration_time: float
    residuals: Optional[np.ndarray] = None
    message: str = ""


# =============================================================================
# Black-Scholes Helpers
# =============================================================================

def black_scholes_call(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Black-Scholes call price."""
    if not SCIPY_AVAILABLE:
        raise ImportError("SciPy required")

    if T <= 0 or sigma <= 0:
        return max(S - K * np.exp(-r * T), 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)


def black_scholes_put(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Black-Scholes put price."""
    if not SCIPY_AVAILABLE:
        raise ImportError("SciPy required")

    if T <= 0 or sigma <= 0:
        return max(K * np.exp(-r * T) - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    option_type: str = "call",
    tol: float = 1e-6,
    max_iter: int = 100
) -> float:
    """
    Calculate implied volatility using Newton-Raphson.
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("SciPy required")

    # Initial guess using Brenner-Subrahmanyam approximation
    sigma = np.sqrt(2 * np.pi / T) * market_price / S
    sigma = max(0.01, min(sigma, 5.0))

    price_func = black_scholes_call if option_type == "call" else black_scholes_put

    for _ in range(max_iter):
        price = price_func(S, K, r, sigma, T)

        # Vega
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        vega = S * np.sqrt(T) * stats.norm.pdf(d1)

        if vega < 1e-10:
            break

        diff = price - market_price
        if abs(diff) < tol:
            return sigma

        sigma = sigma - diff / vega
        sigma = max(0.001, min(sigma, 10.0))

    return sigma


# =============================================================================
# Heston Model Pricing (Semi-Analytical)
# =============================================================================

def heston_characteristic_function(
    u: complex,
    S: float,
    K: float,
    r: float,
    T: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float
) -> complex:
    """
    Heston model characteristic function for option pricing.

    Uses the formulation from Albrecher et al. (2007) for numerical stability.
    """
    i = complex(0, 1)

    # Parameters
    a = kappa * theta

    # Modified parameters for numerical stability
    u_adj = u - 0.5 * i

    d = np.sqrt((rho * xi * i * u - kappa) ** 2 + xi ** 2 * (i * u + u ** 2))

    g = (kappa - rho * xi * i * u - d) / (kappa - rho * xi * i * u + d)

    C = kappa * (
        (kappa - rho * xi * i * u - d) * T -
        2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))
    ) / (xi ** 2)

    D = (kappa - rho * xi * i * u - d) * (1 - np.exp(-d * T)) / (
        xi ** 2 * (1 - g * np.exp(-d * T))
    )

    return np.exp(i * u * (np.log(S) + r * T) + C * theta + D * v0)


def heston_call_price(
    S: float,
    K: float,
    r: float,
    T: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    num_points: int = 100
) -> float:
    """
    Calculate Heston call price using Fourier inversion.

    Uses the Carr-Madan formula with numerical integration.
    """
    if T <= 0:
        return max(S - K, 0)

    # Integration over characteristic function
    du = 0.01
    u_max = num_points * du

    integral = 0.0
    for j in range(1, num_points):
        u = j * du

        phi1 = heston_characteristic_function(
            u - 1j, S, K, r, T, v0, kappa, theta, xi, rho
        )
        phi2 = heston_characteristic_function(
            u, S, K, r, T, v0, kappa, theta, xi, rho
        )

        integrand1 = np.real(np.exp(-1j * u * np.log(K)) * phi1 / (1j * u * S * np.exp(r * T)))
        integrand2 = np.real(np.exp(-1j * u * np.log(K)) * phi2 / (1j * u))

        integral += (integrand1 - K * np.exp(-r * T) * integrand2) * du

    price = 0.5 * (S - K * np.exp(-r * T)) + integral / np.pi

    return max(price, 0.0)


# =============================================================================
# Heston Calibration
# =============================================================================

def calibrate_heston(
    market_options: List[MarketOption],
    spot: float,
    rate: float,
    initial_params: Optional[Dict[str, float]] = None,
    method: str = "L-BFGS-B",
    use_iv: bool = True,
    max_iter: int = 1000,
    tol: float = 1e-6
) -> CalibrationResult:
    """
    Calibrate Heston model to market options.

    Args:
        market_options: List of market option data
        spot: Current spot price
        rate: Risk-free rate
        initial_params: Initial parameter guess
        method: Optimization method ("L-BFGS-B", "differential_evolution", "basinhopping")
        use_iv: If True, calibrate to implied vols; if False, to prices
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        CalibrationResult with calibrated parameters
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("SciPy required for calibration")

    import time
    start_time = time.time()

    # Default initial parameters
    if initial_params is None:
        initial_params = {
            'v0': 0.04,      # Initial variance
            'kappa': 2.0,    # Mean reversion speed
            'theta': 0.04,   # Long-term variance
            'xi': 0.3,       # Vol of vol
            'rho': -0.7      # Correlation
        }

    # Parameter bounds
    bounds = [
        (0.001, 1.0),   # v0
        (0.01, 10.0),   # kappa
        (0.001, 1.0),   # theta
        (0.01, 2.0),    # xi
        (-0.99, 0.99)   # rho
    ]

    x0 = [
        initial_params['v0'],
        initial_params['kappa'],
        initial_params['theta'],
        initial_params['xi'],
        initial_params['rho']
    ]

    # Market data
    market_values = []
    weights = []

    for opt in market_options:
        if use_iv and opt.market_iv is not None:
            market_values.append(opt.market_iv)
        elif opt.market_price is not None:
            market_values.append(opt.market_price)
        else:
            continue

        # Weight by vega (ATM options more important)
        moneyness = opt.strike / spot
        weight = np.exp(-0.5 * (np.log(moneyness)) ** 2 / 0.1)
        weights.append(weight)

    market_values = np.array(market_values)
    weights = np.array(weights)
    weights = weights / weights.sum()

    def objective(params):
        v0, kappa, theta, xi, rho = params

        # Feller condition penalty
        feller_penalty = 0.0
        if 2 * kappa * theta < xi ** 2:
            feller_penalty = 100 * (xi ** 2 - 2 * kappa * theta) ** 2

        model_values = []
        for opt in market_options:
            try:
                price = heston_call_price(
                    spot, opt.strike, rate, opt.maturity,
                    v0, kappa, theta, xi, rho
                )

                if opt.option_type == "put":
                    # Put-call parity
                    price = price - spot + opt.strike * np.exp(-rate * opt.maturity)

                if use_iv and opt.market_iv is not None:
                    # Convert to implied vol
                    try:
                        model_iv = implied_volatility(
                            price, spot, opt.strike, rate, opt.maturity, opt.option_type
                        )
                        model_values.append(model_iv)
                    except:
                        model_values.append(np.sqrt(v0))  # Fallback
                else:
                    model_values.append(price)
            except:
                model_values.append(np.nan)

        model_values = np.array(model_values)

        # Handle NaN
        valid = ~np.isnan(model_values)
        if valid.sum() == 0:
            return 1e10

        # Weighted SSE
        residuals = (model_values[valid] - market_values[valid]) ** 2
        sse = np.sum(weights[valid] * residuals)

        return sse + feller_penalty

    # Optimization
    if method == "differential_evolution":
        result = differential_evolution(
            objective, bounds, maxiter=max_iter, tol=tol,
            seed=42, polish=True
        )
    elif method == "basinhopping":
        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}
        result = basinhopping(
            objective, x0, minimizer_kwargs=minimizer_kwargs,
            niter=max_iter // 10
        )
    else:
        result = minimize(
            objective, x0, method=method, bounds=bounds,
            options={'maxiter': max_iter, 'ftol': tol}
        )

    # Extract results
    v0, kappa, theta, xi, rho = result.x

    # Calculate RMSE
    model_values = []
    for opt in market_options:
        price = heston_call_price(
            spot, opt.strike, rate, opt.maturity,
            v0, kappa, theta, xi, rho
        )
        if opt.option_type == "put":
            price = price - spot + opt.strike * np.exp(-rate * opt.maturity)

        if use_iv and opt.market_iv is not None:
            try:
                model_iv = implied_volatility(
                    price, spot, opt.strike, rate, opt.maturity, opt.option_type
                )
                model_values.append(model_iv)
            except:
                model_values.append(np.nan)
        else:
            model_values.append(price)

    model_values = np.array(model_values)
    valid = ~np.isnan(model_values)
    residuals = model_values[valid] - market_values[valid]
    rmse = np.sqrt(np.mean(residuals ** 2))

    elapsed_time = time.time() - start_time

    return CalibrationResult(
        success=result.success if hasattr(result, 'success') else True,
        parameters={
            'v0': v0,
            'kappa': kappa,
            'theta': theta,
            'xi': xi,
            'rho': rho
        },
        objective_value=result.fun,
        rmse=rmse,
        num_iterations=result.nit if hasattr(result, 'nit') else 0,
        calibration_time=elapsed_time,
        residuals=residuals,
        message=result.message if hasattr(result, 'message') else ""
    )


# =============================================================================
# SABR Calibration
# =============================================================================

def sabr_implied_vol_hagan(
    F: float,
    K: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float
) -> float:
    """
    Hagan's SABR implied volatility formula.
    """
    if T <= 0:
        return 0.0

    if abs(F - K) < 1e-10:
        # ATM
        FK = F
        vol = alpha / (FK ** (1 - beta)) * (
            1 + (
                (1 - beta) ** 2 / 24 * alpha ** 2 / FK ** (2 - 2 * beta) +
                rho * beta * nu * alpha / (4 * FK ** (1 - beta)) +
                (2 - 3 * rho ** 2) / 24 * nu ** 2
            ) * T
        )
        return vol

    # OTM/ITM
    log_FK = np.log(F / K)
    FK_mid = (F * K) ** 0.5

    z = nu / alpha * FK_mid ** (1 - beta) * log_FK

    # Handle small z
    if abs(z) < 1e-10:
        x_z = 1.0
    else:
        x_z = z / np.log(
            (np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho)
        )

    # Correction terms
    A = alpha / (
        FK_mid ** (1 - beta) * (
            1 + (1 - beta) ** 2 / 24 * log_FK ** 2 +
            (1 - beta) ** 4 / 1920 * log_FK ** 4
        )
    )

    B = 1 + (
        (1 - beta) ** 2 / 24 * alpha ** 2 / FK_mid ** (2 - 2 * beta) +
        rho * beta * nu * alpha / (4 * FK_mid ** (1 - beta)) +
        (2 - 3 * rho ** 2) / 24 * nu ** 2
    ) * T

    return A * x_z * B


def calibrate_sabr_slice(
    market_options: List[MarketOption],
    forward: float,
    maturity: float,
    beta: float = 0.5,
    initial_params: Optional[Dict[str, float]] = None
) -> CalibrationResult:
    """
    Calibrate SABR model to a single maturity slice.

    Args:
        market_options: Options at a single maturity
        forward: Forward price
        maturity: Time to expiry
        beta: Fixed beta parameter (often fixed to 0 or 1)
        initial_params: Initial guess for alpha, rho, nu

    Returns:
        CalibrationResult with SABR parameters
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("SciPy required for calibration")

    import time
    start_time = time.time()

    # Default initial parameters
    if initial_params is None:
        # ATM vol approximation for alpha
        atm_opt = min(market_options, key=lambda o: abs(o.strike - forward))
        atm_vol = atm_opt.market_iv if atm_opt.market_iv else 0.2
        alpha_init = atm_vol * forward ** (1 - beta)

        initial_params = {
            'alpha': alpha_init,
            'rho': -0.3,
            'nu': 0.4
        }

    # Bounds
    bounds = [
        (0.001, 5.0),    # alpha
        (-0.999, 0.999), # rho
        (0.001, 5.0)     # nu
    ]

    x0 = [initial_params['alpha'], initial_params['rho'], initial_params['nu']]

    # Market IVs
    market_ivs = np.array([opt.market_iv for opt in market_options])
    strikes = np.array([opt.strike for opt in market_options])

    def objective(params):
        alpha, rho, nu = params

        model_ivs = []
        for K in strikes:
            try:
                iv = sabr_implied_vol_hagan(forward, K, maturity, alpha, beta, rho, nu)
                model_ivs.append(iv)
            except:
                model_ivs.append(np.nan)

        model_ivs = np.array(model_ivs)
        valid = ~np.isnan(model_ivs)

        if valid.sum() == 0:
            return 1e10

        sse = np.sum((model_ivs[valid] - market_ivs[valid]) ** 2)
        return sse

    result = minimize(
        objective, x0, method='L-BFGS-B', bounds=bounds,
        options={'maxiter': 1000}
    )

    alpha, rho, nu = result.x

    # Calculate RMSE
    model_ivs = [sabr_implied_vol_hagan(forward, K, maturity, alpha, beta, rho, nu)
                 for K in strikes]
    model_ivs = np.array(model_ivs)
    residuals = model_ivs - market_ivs
    rmse = np.sqrt(np.mean(residuals ** 2))

    elapsed_time = time.time() - start_time

    return CalibrationResult(
        success=result.success,
        parameters={
            'alpha': alpha,
            'beta': beta,
            'rho': rho,
            'nu': nu
        },
        objective_value=result.fun,
        rmse=rmse,
        num_iterations=result.nit,
        calibration_time=elapsed_time,
        residuals=residuals,
        message=result.message
    )


def calibrate_sabr_surface(
    market_options: List[MarketOption],
    spot: float,
    rate: float,
    beta: float = 0.5
) -> Dict[float, CalibrationResult]:
    """
    Calibrate SABR parameters for each maturity slice.

    Args:
        market_options: All market options
        spot: Current spot price
        rate: Risk-free rate
        beta: Fixed beta for all slices

    Returns:
        Dictionary mapping maturity to CalibrationResult
    """
    # Group by maturity
    maturities = sorted(set(opt.maturity for opt in market_options))

    results = {}
    for T in maturities:
        slice_options = [opt for opt in market_options if opt.maturity == T]

        if len(slice_options) < 3:
            continue

        # Forward price
        forward = spot * np.exp(rate * T)

        result = calibrate_sabr_slice(slice_options, forward, T, beta)
        results[T] = result

    return results


# =============================================================================
# Local Volatility Extraction
# =============================================================================

def dupire_local_vol(
    vol_surface: VolSurface,
    K: float,
    T: float,
    dk: float = 0.01,
    dt: float = 0.01
) -> float:
    """
    Extract local volatility using Dupire's formula.

    σ_loc²(K,T) = [∂C/∂T + rK∂C/∂K + qC] / [0.5 * K² * ∂²C/∂K²]

    For implied volatility surface:
    Uses finite differences on the call price surface.
    """
    S = vol_surface.spot
    r = vol_surface.rate

    # Get implied vols for numerical derivatives
    iv = vol_surface.interpolate(K, T)
    iv_K_up = vol_surface.interpolate(K + dk * K, T)
    iv_K_down = vol_surface.interpolate(K - dk * K, T)
    iv_T_up = vol_surface.interpolate(K, T + dt) if T + dt <= vol_surface.maturities[-1] else iv

    # Call prices
    C = black_scholes_call(S, K, r, iv, T)
    C_K_up = black_scholes_call(S, K + dk * K, r, iv_K_up, T)
    C_K_down = black_scholes_call(S, K - dk * K, r, iv_K_down, T)
    C_T_up = black_scholes_call(S, K, r, iv_T_up, T + dt) if T + dt <= vol_surface.maturities[-1] else C

    # Numerical derivatives
    dC_dT = (C_T_up - C) / dt if T + dt <= vol_surface.maturities[-1] else 0
    dC_dK = (C_K_up - C_K_down) / (2 * dk * K)
    d2C_dK2 = (C_K_up - 2 * C + C_K_down) / ((dk * K) ** 2)

    # Dupire formula
    numerator = dC_dT + r * K * dC_dK
    denominator = 0.5 * K ** 2 * d2C_dK2

    if denominator <= 1e-10:
        return iv  # Fallback to implied vol

    local_var = numerator / denominator

    if local_var <= 0:
        return iv  # Fallback

    return np.sqrt(local_var)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Model Calibration Module - Demo")
    print("=" * 60)

    if not SCIPY_AVAILABLE:
        print("SciPy not available. Skipping demo.")
        exit()

    # Create synthetic market data
    spot = 100.0
    rate = 0.05

    # True Heston parameters (we'll try to recover these)
    true_params = {
        'v0': 0.04,
        'kappa': 2.0,
        'theta': 0.04,
        'xi': 0.3,
        'rho': -0.7
    }

    # Generate synthetic option prices
    strikes = [90, 95, 100, 105, 110]
    maturities = [0.25, 0.5, 1.0]

    market_options = []
    for T in maturities:
        for K in strikes:
            price = heston_call_price(
                spot, K, rate, T,
                **true_params
            )
            # Add some noise
            price *= (1 + np.random.normal(0, 0.01))

            iv = implied_volatility(price, spot, K, rate, T)

            market_options.append(MarketOption(
                strike=K,
                maturity=T,
                market_price=price,
                market_iv=iv
            ))

    # Calibrate Heston
    print("\n1. Heston Model Calibration")
    print("-" * 40)
    print(f"   True parameters: {true_params}")

    result = calibrate_heston(
        market_options, spot, rate,
        method="L-BFGS-B",
        use_iv=True
    )

    print(f"   Calibrated:      {result.parameters}")
    print(f"   RMSE:            {result.rmse:.6f}")
    print(f"   Time:            {result.calibration_time:.2f}s")
    print(f"   Success:         {result.success}")

    # SABR calibration for single slice
    print("\n2. SABR Calibration (1Y slice)")
    print("-" * 40)

    slice_options = [opt for opt in market_options if opt.maturity == 1.0]
    forward = spot * np.exp(rate * 1.0)

    sabr_result = calibrate_sabr_slice(slice_options, forward, 1.0, beta=0.5)

    print(f"   Parameters:      {sabr_result.parameters}")
    print(f"   RMSE:            {sabr_result.rmse:.6f}")

    print("\n" + "=" * 60)
    print("Demo Complete!")
