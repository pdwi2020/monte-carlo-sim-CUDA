"""
Risk Metrics Module - VaR, CVaR, and Portfolio Risk Calculations

This module provides comprehensive risk measurement tools:
- Historical Simulation VaR
- Monte Carlo VaR
- Parametric VaR (Delta-Normal)
- Expected Shortfall (CVaR / Conditional VaR)
- Portfolio risk aggregation

References:
    - Jorion, P. (2007). Value at Risk.
    - McNeil, A., Frey, R., Embrechts, P. (2015). Quantitative Risk Management.
"""

import numpy as np
from typing import Optional, List, Dict, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings

try:
    from scipy import stats
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    stats = None
    minimize = None
    SCIPY_AVAILABLE = False


class VaRMethod(Enum):
    """VaR calculation methodology."""
    HISTORICAL = "historical"
    MONTE_CARLO = "monte_carlo"
    PARAMETRIC = "parametric"  # Delta-Normal


@dataclass
class VaRResult:
    """Result of VaR calculation."""
    var: float  # Value at Risk (positive number = loss)
    confidence_level: float
    method: str
    horizon_days: int
    cvar: Optional[float] = None  # Expected Shortfall
    num_observations: Optional[int] = None
    num_simulations: Optional[int] = None


@dataclass
class PortfolioPosition:
    """Single position in a portfolio."""
    asset_id: str
    quantity: float
    current_price: float
    volatility: Optional[float] = None  # Annual volatility

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price


@dataclass
class Portfolio:
    """Portfolio of positions for risk calculation."""
    positions: List[PortfolioPosition]
    correlation_matrix: Optional[np.ndarray] = None

    @property
    def total_value(self) -> float:
        return sum(p.market_value for p in self.positions)

    @property
    def weights(self) -> np.ndarray:
        total = self.total_value
        return np.array([p.market_value / total for p in self.positions])

    @property
    def volatilities(self) -> np.ndarray:
        return np.array([p.volatility or 0.2 for p in self.positions])


# =============================================================================
# Historical Simulation VaR
# =============================================================================

def historical_var(
    returns: np.ndarray,
    confidence_level: float = 0.95,
    horizon_days: int = 1,
    portfolio_value: float = 1.0
) -> VaRResult:
    """
    Calculate VaR using historical simulation.

    Args:
        returns: Array of historical returns (daily)
        confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
        horizon_days: Holding period in days
        portfolio_value: Current portfolio value

    Returns:
        VaRResult with VaR and CVaR
    """
    returns = np.asarray(returns)

    # Scale returns for horizon
    if horizon_days > 1:
        # Use overlapping windows or square-root scaling
        scaled_returns = returns * np.sqrt(horizon_days)
    else:
        scaled_returns = returns

    # VaR is the quantile of losses (negative returns)
    alpha = 1 - confidence_level
    var_return = np.percentile(scaled_returns, alpha * 100)
    var = -var_return * portfolio_value  # Convert to loss

    # CVaR (Expected Shortfall) = E[Loss | Loss > VaR]
    tail_returns = scaled_returns[scaled_returns <= var_return]
    if len(tail_returns) > 0:
        cvar = -np.mean(tail_returns) * portfolio_value
    else:
        cvar = var

    return VaRResult(
        var=max(var, 0),
        confidence_level=confidence_level,
        method="historical",
        horizon_days=horizon_days,
        cvar=max(cvar, 0),
        num_observations=len(returns)
    )


def historical_var_weighted(
    returns: np.ndarray,
    confidence_level: float = 0.95,
    horizon_days: int = 1,
    portfolio_value: float = 1.0,
    decay_factor: float = 0.94
) -> VaRResult:
    """
    Calculate VaR using age-weighted historical simulation.

    Recent observations receive higher weights using exponential decay.

    Args:
        returns: Historical returns (most recent last)
        confidence_level: Confidence level
        horizon_days: Holding period
        portfolio_value: Current portfolio value
        decay_factor: Exponential decay (0.94 = ~30 day half-life)
    """
    returns = np.asarray(returns)
    n = len(returns)

    # Exponential weights (more recent = higher weight)
    weights = decay_factor ** np.arange(n - 1, -1, -1)
    weights = weights / weights.sum()

    # Sort returns and cumulate weights
    sorted_idx = np.argsort(returns)
    sorted_returns = returns[sorted_idx]
    sorted_weights = weights[sorted_idx]
    cumulative_weights = np.cumsum(sorted_weights)

    # Find VaR quantile
    alpha = 1 - confidence_level
    var_idx = np.searchsorted(cumulative_weights, alpha)
    var_idx = min(var_idx, n - 1)

    var_return = sorted_returns[var_idx]

    # Scale for horizon
    if horizon_days > 1:
        var_return = var_return * np.sqrt(horizon_days)

    var = -var_return * portfolio_value

    # Weighted CVaR
    tail_mask = sorted_returns <= var_return
    if tail_mask.sum() > 0:
        tail_weights = sorted_weights[tail_mask]
        tail_returns = sorted_returns[tail_mask]
        cvar = -np.average(tail_returns, weights=tail_weights) * portfolio_value
        if horizon_days > 1:
            cvar = cvar * np.sqrt(horizon_days)
    else:
        cvar = var

    return VaRResult(
        var=max(var, 0),
        confidence_level=confidence_level,
        method="historical_weighted",
        horizon_days=horizon_days,
        cvar=max(cvar, 0),
        num_observations=n
    )


# =============================================================================
# Parametric (Delta-Normal) VaR
# =============================================================================

def parametric_var(
    portfolio_value: float,
    volatility: float,
    confidence_level: float = 0.95,
    horizon_days: int = 1,
    mean_return: float = 0.0
) -> VaRResult:
    """
    Calculate VaR using parametric (Delta-Normal) method.

    Assumes returns are normally distributed.

    Args:
        portfolio_value: Current portfolio value
        volatility: Annual volatility (standard deviation of returns)
        confidence_level: Confidence level
        horizon_days: Holding period in days
        mean_return: Expected daily return (usually assumed 0)

    Returns:
        VaRResult with VaR and CVaR
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("SciPy required for parametric VaR")

    # Daily volatility
    daily_vol = volatility / np.sqrt(252)

    # Scale for horizon
    horizon_vol = daily_vol * np.sqrt(horizon_days)
    horizon_mean = mean_return * horizon_days

    # Z-score for confidence level
    z = stats.norm.ppf(confidence_level)

    # VaR = -μ + σ * z
    var = portfolio_value * (-horizon_mean + horizon_vol * z)

    # CVaR for normal distribution
    # E[X | X > VaR] = μ + σ * φ(z) / (1-Φ(z))
    phi_z = stats.norm.pdf(z)
    cvar_factor = phi_z / (1 - confidence_level)
    cvar = portfolio_value * (-horizon_mean + horizon_vol * cvar_factor)

    return VaRResult(
        var=max(var, 0),
        confidence_level=confidence_level,
        method="parametric",
        horizon_days=horizon_days,
        cvar=max(cvar, 0)
    )


def parametric_var_portfolio(
    portfolio: Portfolio,
    confidence_level: float = 0.95,
    horizon_days: int = 1
) -> VaRResult:
    """
    Calculate portfolio VaR using parametric method with correlations.

    Uses the variance-covariance approach:
    σ_p² = w' Σ w

    Args:
        portfolio: Portfolio with positions and correlation matrix
        confidence_level: Confidence level
        horizon_days: Holding period
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("SciPy required for parametric VaR")

    weights = portfolio.weights
    vols = portfolio.volatilities
    n = len(weights)

    # Correlation matrix (default: identity = independent)
    if portfolio.correlation_matrix is not None:
        corr = portfolio.correlation_matrix
    else:
        corr = np.eye(n)

    # Covariance matrix: Σ = diag(σ) @ ρ @ diag(σ)
    cov = np.outer(vols, vols) * corr

    # Portfolio variance: σ_p² = w' Σ w
    portfolio_var = weights @ cov @ weights
    portfolio_vol = np.sqrt(portfolio_var)

    # Daily volatility
    daily_vol = portfolio_vol / np.sqrt(252)
    horizon_vol = daily_vol * np.sqrt(horizon_days)

    # Z-score
    z = stats.norm.ppf(confidence_level)

    # VaR
    var = portfolio.total_value * horizon_vol * z

    # CVaR
    phi_z = stats.norm.pdf(z)
    cvar_factor = phi_z / (1 - confidence_level)
    cvar = portfolio.total_value * horizon_vol * cvar_factor

    return VaRResult(
        var=max(var, 0),
        confidence_level=confidence_level,
        method="parametric_portfolio",
        horizon_days=horizon_days,
        cvar=max(cvar, 0)
    )


# =============================================================================
# Monte Carlo VaR
# =============================================================================

def monte_carlo_var(
    portfolio_value: float,
    volatility: float,
    confidence_level: float = 0.95,
    horizon_days: int = 1,
    num_simulations: int = 100000,
    mean_return: float = 0.0,
    distribution: str = "normal",
    seed: Optional[int] = None
) -> VaRResult:
    """
    Calculate VaR using Monte Carlo simulation.

    Supports different return distributions.

    Args:
        portfolio_value: Current portfolio value
        volatility: Annual volatility
        confidence_level: Confidence level
        horizon_days: Holding period
        num_simulations: Number of simulations
        mean_return: Expected daily return
        distribution: "normal", "t" (Student-t), or "mixture"
        seed: Random seed
    """
    rng = np.random.default_rng(seed)

    daily_vol = volatility / np.sqrt(252)
    horizon_vol = daily_vol * np.sqrt(horizon_days)
    horizon_mean = mean_return * horizon_days

    # Generate random returns
    if distribution == "normal":
        returns = rng.normal(horizon_mean, horizon_vol, num_simulations)
    elif distribution == "t":
        # Student-t with 5 degrees of freedom (fatter tails)
        if SCIPY_AVAILABLE:
            returns = stats.t.rvs(df=5, loc=horizon_mean,
                                  scale=horizon_vol * np.sqrt(3/5),  # Adjust scale for same variance
                                  size=num_simulations, random_state=seed)
        else:
            returns = rng.standard_t(5, num_simulations) * horizon_vol * np.sqrt(3/5) + horizon_mean
    elif distribution == "mixture":
        # Normal mixture (normal + fat-tailed component)
        normal_weight = 0.95
        n_normal = int(num_simulations * normal_weight)
        n_fat = num_simulations - n_normal

        normal_returns = rng.normal(horizon_mean, horizon_vol, n_normal)
        fat_returns = rng.normal(horizon_mean, horizon_vol * 2, n_fat)  # 2x vol for fat tails
        returns = np.concatenate([normal_returns, fat_returns])
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    # Portfolio P&L
    pnl = portfolio_value * returns

    # VaR (percentile of losses)
    alpha = 1 - confidence_level
    var = -np.percentile(pnl, alpha * 100)

    # CVaR
    tail_pnl = pnl[pnl <= -var]
    if len(tail_pnl) > 0:
        cvar = -np.mean(tail_pnl)
    else:
        cvar = var

    return VaRResult(
        var=max(var, 0),
        confidence_level=confidence_level,
        method=f"monte_carlo_{distribution}",
        horizon_days=horizon_days,
        cvar=max(cvar, 0),
        num_simulations=num_simulations
    )


def monte_carlo_var_paths(
    portfolio_value: float,
    volatility: float,
    confidence_level: float = 0.95,
    horizon_days: int = 10,
    num_paths: int = 10000,
    steps_per_day: int = 1,
    mean_return: float = 0.0,
    seed: Optional[int] = None
) -> Tuple[VaRResult, np.ndarray]:
    """
    Calculate VaR with full path simulation.

    Returns VaR result and simulated price paths.

    Args:
        portfolio_value: Initial portfolio value
        volatility: Annual volatility
        confidence_level: Confidence level
        horizon_days: Holding period
        num_paths: Number of simulation paths
        steps_per_day: Discretization steps per day
        mean_return: Expected daily return
        seed: Random seed

    Returns:
        Tuple of (VaRResult, price_paths array of shape (num_paths, num_steps+1))
    """
    rng = np.random.default_rng(seed)

    num_steps = horizon_days * steps_per_day
    dt = 1.0 / (252 * steps_per_day)  # Time step in years

    daily_drift = mean_return / 252
    annual_vol = volatility

    # Simulate GBM paths
    Z = rng.standard_normal((num_paths, num_steps))

    drift = (daily_drift * 252 - 0.5 * annual_vol ** 2) * dt
    diffusion = annual_vol * np.sqrt(dt) * Z

    log_returns = np.cumsum(drift + diffusion, axis=1)

    paths = np.zeros((num_paths, num_steps + 1))
    paths[:, 0] = portfolio_value
    paths[:, 1:] = portfolio_value * np.exp(log_returns)

    # Terminal P&L
    terminal_pnl = paths[:, -1] - portfolio_value

    # VaR
    alpha = 1 - confidence_level
    var = -np.percentile(terminal_pnl, alpha * 100)

    # CVaR
    tail_pnl = terminal_pnl[terminal_pnl <= -var]
    cvar = -np.mean(tail_pnl) if len(tail_pnl) > 0 else var

    result = VaRResult(
        var=max(var, 0),
        confidence_level=confidence_level,
        method="monte_carlo_paths",
        horizon_days=horizon_days,
        cvar=max(cvar, 0),
        num_simulations=num_paths
    )

    return result, paths


# =============================================================================
# Stress Testing
# =============================================================================

@dataclass
class StressScenario:
    """Definition of a stress test scenario."""
    name: str
    description: str
    market_move: float  # Percentage move (e.g., -0.20 = -20%)
    volatility_shock: float = 0.0  # Additive vol shock
    correlation_shock: float = 0.0  # Correlation shift towards 1


@dataclass
class StressTestResult:
    """Result of stress testing."""
    scenario_name: str
    portfolio_loss: float
    loss_percentage: float
    stressed_var: Optional[float] = None


def run_stress_test(
    portfolio_value: float,
    volatility: float,
    scenarios: List[StressScenario],
    var_confidence: float = 0.95
) -> List[StressTestResult]:
    """
    Run stress tests on a portfolio.

    Args:
        portfolio_value: Current portfolio value
        volatility: Annual volatility
        scenarios: List of stress scenarios
        var_confidence: Confidence level for stressed VaR

    Returns:
        List of stress test results
    """
    results = []

    for scenario in scenarios:
        # Direct loss from market move
        loss = -portfolio_value * scenario.market_move
        loss_pct = -scenario.market_move

        # Stressed VaR with increased volatility
        stressed_vol = volatility + scenario.volatility_shock
        stressed_var_result = parametric_var(
            portfolio_value=portfolio_value,
            volatility=stressed_vol,
            confidence_level=var_confidence,
            horizon_days=1
        )

        results.append(StressTestResult(
            scenario_name=scenario.name,
            portfolio_loss=loss,
            loss_percentage=loss_pct,
            stressed_var=stressed_var_result.var
        ))

    return results


# =============================================================================
# Predefined Stress Scenarios
# =============================================================================

STANDARD_SCENARIOS = [
    StressScenario(
        name="2008 Financial Crisis",
        description="Lehman collapse, -35% equities, vol spike",
        market_move=-0.35,
        volatility_shock=0.40,
        correlation_shock=0.3
    ),
    StressScenario(
        name="COVID-19 Crash",
        description="March 2020, -30% equities, vol spike",
        market_move=-0.30,
        volatility_shock=0.60,
        correlation_shock=0.4
    ),
    StressScenario(
        name="1987 Black Monday",
        description="Single-day 22% drop",
        market_move=-0.22,
        volatility_shock=0.50,
        correlation_shock=0.5
    ),
    StressScenario(
        name="Moderate Correction",
        description="10% market pullback",
        market_move=-0.10,
        volatility_shock=0.10
    ),
    StressScenario(
        name="Severe Recession",
        description="50% bear market",
        market_move=-0.50,
        volatility_shock=0.30,
        correlation_shock=0.4
    ),
]


# =============================================================================
# Backtesting VaR
# =============================================================================

@dataclass
class BacktestResult:
    """Result of VaR backtesting."""
    num_observations: int
    num_breaches: int
    breach_rate: float
    expected_breaches: float
    p_value: Optional[float] = None  # Kupiec test p-value
    is_model_rejected: bool = False


def backtest_var(
    returns: np.ndarray,
    var_values: np.ndarray,
    confidence_level: float = 0.95
) -> BacktestResult:
    """
    Backtest VaR model using Kupiec's POF test.

    Tests whether the number of breaches is consistent with the confidence level.

    Args:
        returns: Realized returns
        var_values: Predicted VaR values (positive = loss)
        confidence_level: VaR confidence level

    Returns:
        BacktestResult with statistics
    """
    returns = np.asarray(returns)
    var_values = np.asarray(var_values)

    n = len(returns)

    # A breach occurs when realized loss exceeds VaR
    # Loss = -return, breach if -return > var
    breaches = (-returns) > var_values
    num_breaches = np.sum(breaches)
    breach_rate = num_breaches / n

    expected_rate = 1 - confidence_level
    expected_breaches = n * expected_rate

    # Kupiec's POF (Proportion of Failures) test
    # LR = -2 * log((1-p)^(n-x) * p^x / ((1-x/n)^(n-x) * (x/n)^x))
    p_value = None
    is_rejected = False

    if SCIPY_AVAILABLE and num_breaches > 0 and num_breaches < n:
        x = num_breaches
        p = expected_rate
        p_hat = x / n

        # Likelihood ratio
        lr = -2 * (
            (n - x) * np.log((1 - p) / (1 - p_hat)) +
            x * np.log(p / p_hat)
        )

        # Under H0, LR ~ chi-squared with 1 df
        p_value = 1 - stats.chi2.cdf(lr, 1)
        is_rejected = p_value < 0.05

    return BacktestResult(
        num_observations=n,
        num_breaches=int(num_breaches),
        breach_rate=breach_rate,
        expected_breaches=expected_breaches,
        p_value=p_value,
        is_model_rejected=is_rejected
    )


# =============================================================================
# Notebook-Friendly Wrapper Functions
# =============================================================================

def calculate_var_cvar(
    returns: np.ndarray,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Simple VaR and CVaR calculation from returns array.

    Args:
        returns: Array of returns (can be simulated or historical)
        confidence: Confidence level (e.g., 0.95 for 95%)

    Returns:
        Tuple of (VaR, CVaR) as positive numbers representing loss
    """
    returns = np.asarray(returns)

    # VaR is the quantile of losses (negative returns)
    var = -np.percentile(returns, (1 - confidence) * 100)

    # CVaR is the expected loss given loss exceeds VaR
    losses = -returns
    cvar = np.mean(losses[losses >= var])

    return float(var), float(cvar)


def simulate_portfolio_returns(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    volatilities: np.ndarray,
    correlation: float,
    num_sims: int = 10000,
    horizon_days: int = 1,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Simulate portfolio returns using correlated normal distributions.

    Args:
        weights: Portfolio weights (must sum to 1)
        expected_returns: Annual expected returns per asset
        volatilities: Annual volatilities per asset
        correlation: Uniform correlation between all assets
        num_sims: Number of simulations
        horizon_days: Holding period in days
        seed: Random seed

    Returns:
        Array of simulated portfolio returns
    """
    if seed is not None:
        np.random.seed(seed)

    n_assets = len(weights)

    # Build correlation matrix
    corr_matrix = np.full((n_assets, n_assets), correlation)
    np.fill_diagonal(corr_matrix, 1.0)

    # Convert annual to daily
    daily_returns = expected_returns / 252
    daily_vols = volatilities / np.sqrt(252)

    # Scale for horizon
    horizon_returns = daily_returns * horizon_days
    horizon_vols = daily_vols * np.sqrt(horizon_days)

    # Cholesky decomposition
    L = np.linalg.cholesky(corr_matrix)

    # Generate correlated random numbers
    Z = np.random.standard_normal((num_sims, n_assets))
    corr_Z = Z @ L.T

    # Asset returns
    asset_returns = horizon_returns + horizon_vols * corr_Z

    # Portfolio returns
    portfolio_returns = asset_returns @ weights

    return portfolio_returns


def marginal_var(
    portfolio_var: float,
    weights: np.ndarray,
    volatilities: np.ndarray,
    correlation_matrix: np.ndarray
) -> np.ndarray:
    """Calculate marginal VaR contribution of each asset."""
    cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
    portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
    marginal = (cov_matrix @ weights) / portfolio_vol
    return marginal * (portfolio_var / portfolio_vol)


def component_var(
    portfolio_var: float,
    weights: np.ndarray,
    volatilities: np.ndarray,
    correlation_matrix: np.ndarray
) -> np.ndarray:
    """Calculate component VaR (marginal VaR × weight)."""
    m_var = marginal_var(portfolio_var, weights, volatilities, correlation_matrix)
    return m_var * weights


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Risk Metrics Module - Demo")
    print("=" * 60)

    # Generate synthetic returns
    np.random.seed(42)
    daily_returns = np.random.normal(0.0005, 0.02, 252)  # 1 year of daily returns

    portfolio_value = 1_000_000  # $1M portfolio
    annual_vol = 0.20  # 20% annual volatility

    # Historical VaR
    print("\n1. Historical Simulation VaR (95%, 1-day)")
    hist_var = historical_var(daily_returns, 0.95, 1, portfolio_value)
    print(f"   VaR:  ${hist_var.var:,.0f}")
    print(f"   CVaR: ${hist_var.cvar:,.0f}")

    # Parametric VaR
    print("\n2. Parametric (Delta-Normal) VaR (95%, 1-day)")
    param_var = parametric_var(portfolio_value, annual_vol, 0.95, 1)
    print(f"   VaR:  ${param_var.var:,.0f}")
    print(f"   CVaR: ${param_var.cvar:,.0f}")

    # Monte Carlo VaR
    print("\n3. Monte Carlo VaR (95%, 10-day)")
    mc_var = monte_carlo_var(portfolio_value, annual_vol, 0.95, 10, 100000)
    print(f"   VaR:  ${mc_var.var:,.0f}")
    print(f"   CVaR: ${mc_var.cvar:,.0f}")

    # Monte Carlo with Student-t
    print("\n4. Monte Carlo VaR with Fat Tails (Student-t)")
    mc_var_t = monte_carlo_var(portfolio_value, annual_vol, 0.95, 1, 100000, distribution="t")
    print(f"   VaR:  ${mc_var_t.var:,.0f}")
    print(f"   CVaR: ${mc_var_t.cvar:,.0f}")

    # Stress Testing
    print("\n5. Stress Testing")
    stress_results = run_stress_test(portfolio_value, annual_vol, STANDARD_SCENARIOS[:3])
    for r in stress_results:
        print(f"   {r.scenario_name}: Loss ${r.portfolio_loss:,.0f} ({r.loss_percentage:.1%})")

    print("\n" + "=" * 60)
    print("Demo Complete!")
