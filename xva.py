"""
XVA Module - Valuation Adjustments for Derivatives

This module implements:
- CVA (Credit Valuation Adjustment) - counterparty credit risk
- DVA (Debit Valuation Adjustment) - own credit risk
- FVA (Funding Valuation Adjustment) - funding costs
- MVA (Margin Valuation Adjustment) - initial margin costs
- KVA (Capital Valuation Adjustment) - regulatory capital costs

References:
    - Gregory, J. (2015). The xVA Challenge.
    - Green, A. (2015). XVA: Credit, Funding and Capital Valuation Adjustments.
"""

import numpy as np
from typing import Optional, List, Dict, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum

try:
    from scipy import stats
    from scipy.interpolate import interp1d
    SCIPY_AVAILABLE = True
except ImportError:
    stats = None
    interp1d = None
    SCIPY_AVAILABLE = False


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CreditCurve:
    """Credit spread curve for CVA/DVA calculations."""
    tenors: np.ndarray  # Time points in years
    spreads: np.ndarray  # Credit spreads (annual, e.g., 0.01 = 100bps)
    recovery_rate: float = 0.4  # Loss given default

    def __post_init__(self):
        self.tenors = np.asarray(self.tenors)
        self.spreads = np.asarray(self.spreads)
        if len(self.tenors) != len(self.spreads):
            raise ValueError("tenors and spreads must have same length")

    def survival_probability(self, t: float) -> float:
        """
        Calculate survival probability to time t.

        P(τ > t) = exp(-∫₀ᵗ h(s) ds)
        where h(s) ≈ spread(s) / (1 - R)
        """
        if t <= 0:
            return 1.0

        # Interpolate spread
        if SCIPY_AVAILABLE:
            spread_interp = interp1d(
                self.tenors, self.spreads,
                kind='linear', fill_value='extrapolate'
            )
            spread_t = float(spread_interp(t))
        else:
            # Linear interpolation fallback
            spread_t = np.interp(t, self.tenors, self.spreads)

        # Hazard rate approximation
        hazard_rate = spread_t / (1 - self.recovery_rate)

        return np.exp(-hazard_rate * t)

    def default_probability(self, t1: float, t2: float) -> float:
        """
        Probability of default between t1 and t2.

        P(t1 < τ ≤ t2) = P(τ > t1) - P(τ > t2)
        """
        return self.survival_probability(t1) - self.survival_probability(t2)


@dataclass
class FundingCurve:
    """Funding spread curve for FVA calculations."""
    tenors: np.ndarray  # Time points in years
    borrowing_spreads: np.ndarray  # Spread over risk-free for borrowing
    lending_spreads: np.ndarray  # Spread over risk-free for lending

    def __post_init__(self):
        self.tenors = np.asarray(self.tenors)
        self.borrowing_spreads = np.asarray(self.borrowing_spreads)
        self.lending_spreads = np.asarray(self.lending_spreads)

    def borrowing_spread(self, t: float) -> float:
        """Interpolate borrowing spread at time t."""
        return float(np.interp(t, self.tenors, self.borrowing_spreads))

    def lending_spread(self, t: float) -> float:
        """Interpolate lending spread at time t."""
        return float(np.interp(t, self.tenors, self.lending_spreads))


@dataclass
class ExposureProfile:
    """Expected exposure profile for a derivative."""
    times: np.ndarray  # Observation times
    expected_exposure: np.ndarray  # EE at each time (positive = we're owed)
    expected_negative_exposure: np.ndarray  # ENE at each time (negative = we owe)
    potential_future_exposure: Optional[np.ndarray] = None  # PFE (97.5% quantile)

    def __post_init__(self):
        self.times = np.asarray(self.times)
        self.expected_exposure = np.asarray(self.expected_exposure)
        self.expected_negative_exposure = np.asarray(self.expected_negative_exposure)
        if self.potential_future_exposure is not None:
            self.potential_future_exposure = np.asarray(self.potential_future_exposure)


@dataclass
class XVAResult:
    """Result of XVA calculation."""
    cva: float = 0.0  # Credit Valuation Adjustment (cost of counterparty default)
    dva: float = 0.0  # Debit Valuation Adjustment (benefit from own default)
    fva: float = 0.0  # Funding Valuation Adjustment
    colva: float = 0.0  # Collateral Valuation Adjustment
    mva: float = 0.0  # Margin Valuation Adjustment
    kva: float = 0.0  # Capital Valuation Adjustment

    @property
    def total_xva(self) -> float:
        """Total XVA = CVA - DVA + FVA + ColVA + MVA + KVA."""
        return self.cva - self.dva + self.fva + self.colva + self.mva + self.kva

    @property
    def bilateral_cva(self) -> float:
        """Bilateral CVA = CVA - DVA."""
        return self.cva - self.dva


# =============================================================================
# CVA Calculation
# =============================================================================

def calculate_cva(
    exposure_profile: ExposureProfile,
    counterparty_credit: CreditCurve,
    discount_curve: Optional[Callable[[float], float]] = None,
    wrong_way_risk_factor: float = 1.0
) -> float:
    """
    Calculate Credit Valuation Adjustment (CVA).

    CVA = (1 - R) * Σᵢ DF(tᵢ) * EE(tᵢ) * PD(tᵢ₋₁, tᵢ)

    Args:
        exposure_profile: Expected exposure profile
        counterparty_credit: Counterparty credit curve
        discount_curve: Function returning discount factor for time t
        wrong_way_risk_factor: Multiplier for wrong-way risk (>1 increases CVA)

    Returns:
        CVA as a positive number (cost)
    """
    if discount_curve is None:
        # Default: flat 5% discount rate
        discount_curve = lambda t: np.exp(-0.05 * t)

    times = exposure_profile.times
    ee = exposure_profile.expected_exposure
    lgd = 1 - counterparty_credit.recovery_rate

    cva = 0.0
    for i in range(1, len(times)):
        t_prev = times[i - 1]
        t_curr = times[i]

        # Average exposure in interval
        avg_ee = 0.5 * (ee[i - 1] + ee[i])

        # Discount factor at midpoint
        t_mid = 0.5 * (t_prev + t_curr)
        df = discount_curve(t_mid)

        # Default probability in interval
        pd = counterparty_credit.default_probability(t_prev, t_curr)

        # CVA contribution
        cva += df * avg_ee * pd * lgd * wrong_way_risk_factor

    return max(cva, 0.0)


def calculate_cva_monte_carlo(
    mtm_paths: np.ndarray,
    times: np.ndarray,
    counterparty_credit: CreditCurve,
    discount_curve: Optional[Callable[[float], float]] = None,
    num_default_sims: int = 10000,
    seed: Optional[int] = None
) -> float:
    """
    Calculate CVA using Monte Carlo simulation with default time sampling.

    Args:
        mtm_paths: Mark-to-market paths, shape (num_paths, num_times)
        times: Time points
        counterparty_credit: Counterparty credit curve
        discount_curve: Discount factor function
        num_default_sims: Number of default time simulations
        seed: Random seed

    Returns:
        CVA value
    """
    if discount_curve is None:
        discount_curve = lambda t: np.exp(-0.05 * t)

    rng = np.random.default_rng(seed)
    lgd = 1 - counterparty_credit.recovery_rate
    T = times[-1]

    # Sample default times
    U = rng.uniform(0, 1, num_default_sims)

    # Inverse of survival probability to get default time
    # P(τ > t) = S(t), so τ = S^(-1)(U)
    # Approximate by finding where survival prob crosses U

    cva_sims = []

    for u in U:
        # Find default time
        default_time = None
        for t in times:
            if counterparty_credit.survival_probability(t) < u:
                default_time = t
                break

        if default_time is None or default_time > T:
            # No default before maturity
            cva_sims.append(0.0)
            continue

        # Find closest time index
        idx = np.searchsorted(times, default_time)
        idx = min(idx, len(times) - 1)

        # Exposure at default (average across paths)
        exposure_at_default = np.mean(np.maximum(mtm_paths[:, idx], 0))

        # Discounted loss
        df = discount_curve(default_time)
        loss = df * exposure_at_default * lgd
        cva_sims.append(loss)

    return np.mean(cva_sims)


# =============================================================================
# DVA Calculation
# =============================================================================

def calculate_dva(
    exposure_profile: ExposureProfile,
    own_credit: CreditCurve,
    discount_curve: Optional[Callable[[float], float]] = None
) -> float:
    """
    Calculate Debit Valuation Adjustment (DVA).

    DVA = (1 - R) * Σᵢ DF(tᵢ) * ENE(tᵢ) * PD_own(tᵢ₋₁, tᵢ)

    DVA represents the benefit from our own potential default.

    Args:
        exposure_profile: Expected negative exposure profile
        own_credit: Own credit curve
        discount_curve: Discount factor function

    Returns:
        DVA as a positive number (benefit)
    """
    if discount_curve is None:
        discount_curve = lambda t: np.exp(-0.05 * t)

    times = exposure_profile.times
    ene = exposure_profile.expected_negative_exposure
    lgd = 1 - own_credit.recovery_rate

    dva = 0.0
    for i in range(1, len(times)):
        t_prev = times[i - 1]
        t_curr = times[i]

        # Average ENE in interval (negative values)
        avg_ene = 0.5 * (abs(ene[i - 1]) + abs(ene[i]))

        # Discount factor at midpoint
        t_mid = 0.5 * (t_prev + t_curr)
        df = discount_curve(t_mid)

        # Own default probability in interval
        pd = own_credit.default_probability(t_prev, t_curr)

        # DVA contribution
        dva += df * avg_ene * pd * lgd

    return max(dva, 0.0)


# =============================================================================
# FVA Calculation
# =============================================================================

def calculate_fva(
    exposure_profile: ExposureProfile,
    funding_curve: FundingCurve,
    discount_curve: Optional[Callable[[float], float]] = None
) -> float:
    """
    Calculate Funding Valuation Adjustment (FVA).

    FVA accounts for the cost/benefit of funding uncollateralized derivatives.

    FVA = FCA - FBA
    where:
        FCA (Funding Cost Adjustment) = borrowing cost for positive exposure
        FBA (Funding Benefit Adjustment) = lending benefit for negative exposure

    Args:
        exposure_profile: Expected exposure profile
        funding_curve: Funding spreads curve
        discount_curve: Discount factor function

    Returns:
        FVA (positive = cost, negative = benefit)
    """
    if discount_curve is None:
        discount_curve = lambda t: np.exp(-0.05 * t)

    times = exposure_profile.times
    ee = exposure_profile.expected_exposure
    ene = exposure_profile.expected_negative_exposure

    fca = 0.0  # Funding Cost Adjustment
    fba = 0.0  # Funding Benefit Adjustment

    for i in range(1, len(times)):
        t_prev = times[i - 1]
        t_curr = times[i]
        dt = t_curr - t_prev
        t_mid = 0.5 * (t_prev + t_curr)

        df = discount_curve(t_mid)

        # Funding cost (positive exposure - we need to fund)
        avg_ee = 0.5 * (ee[i - 1] + ee[i])
        borrow_spread = funding_curve.borrowing_spread(t_mid)
        fca += df * avg_ee * borrow_spread * dt

        # Funding benefit (negative exposure - we receive funding)
        avg_ene = 0.5 * (abs(ene[i - 1]) + abs(ene[i]))
        lend_spread = funding_curve.lending_spread(t_mid)
        fba += df * avg_ene * lend_spread * dt

    return fca - fba


# =============================================================================
# ColVA Calculation
# =============================================================================

def calculate_colva(
    exposure_profile: ExposureProfile,
    collateral_rate: float,
    risk_free_rate: float,
    discount_curve: Optional[Callable[[float], float]] = None
) -> float:
    """
    Calculate Collateral Valuation Adjustment (ColVA).

    ColVA accounts for the cost of posting collateral at a rate different
    from the risk-free rate.

    Args:
        exposure_profile: Expected exposure profile
        collateral_rate: Rate earned on posted collateral (e.g., OIS)
        risk_free_rate: Risk-free rate
        discount_curve: Discount factor function

    Returns:
        ColVA value
    """
    if discount_curve is None:
        discount_curve = lambda t: np.exp(-risk_free_rate * t)

    times = exposure_profile.times
    ee = exposure_profile.expected_exposure

    spread = risk_free_rate - collateral_rate

    colva = 0.0
    for i in range(1, len(times)):
        t_prev = times[i - 1]
        t_curr = times[i]
        dt = t_curr - t_prev
        t_mid = 0.5 * (t_prev + t_curr)

        df = discount_curve(t_mid)
        avg_ee = 0.5 * (ee[i - 1] + ee[i])

        colva += df * avg_ee * spread * dt

    return colva


# =============================================================================
# MVA Calculation
# =============================================================================

def calculate_mva(
    initial_margin_profile: np.ndarray,
    times: np.ndarray,
    funding_spread: float,
    discount_curve: Optional[Callable[[float], float]] = None
) -> float:
    """
    Calculate Margin Valuation Adjustment (MVA).

    MVA is the cost of funding initial margin over the life of the trade.

    Args:
        initial_margin_profile: Initial margin required at each time point
        times: Time points
        funding_spread: Funding spread for margin
        discount_curve: Discount factor function

    Returns:
        MVA value
    """
    if discount_curve is None:
        discount_curve = lambda t: np.exp(-0.05 * t)

    mva = 0.0
    for i in range(1, len(times)):
        t_prev = times[i - 1]
        t_curr = times[i]
        dt = t_curr - t_prev
        t_mid = 0.5 * (t_prev + t_curr)

        df = discount_curve(t_mid)
        avg_im = 0.5 * (initial_margin_profile[i - 1] + initial_margin_profile[i])

        mva += df * avg_im * funding_spread * dt

    return mva


# =============================================================================
# KVA Calculation
# =============================================================================

def calculate_kva(
    capital_profile: np.ndarray,
    times: np.ndarray,
    cost_of_capital: float = 0.10,
    discount_curve: Optional[Callable[[float], float]] = None
) -> float:
    """
    Calculate Capital Valuation Adjustment (KVA).

    KVA is the cost of holding regulatory capital over the life of the trade.

    Args:
        capital_profile: Regulatory capital required at each time point
        times: Time points
        cost_of_capital: Hurdle rate for capital (e.g., 10%)
        discount_curve: Discount factor function

    Returns:
        KVA value
    """
    if discount_curve is None:
        discount_curve = lambda t: np.exp(-0.05 * t)

    kva = 0.0
    for i in range(1, len(times)):
        t_prev = times[i - 1]
        t_curr = times[i]
        dt = t_curr - t_prev
        t_mid = 0.5 * (t_prev + t_curr)

        df = discount_curve(t_mid)
        avg_capital = 0.5 * (capital_profile[i - 1] + capital_profile[i])

        kva += df * avg_capital * cost_of_capital * dt

    return kva


# =============================================================================
# Full XVA Calculation
# =============================================================================

def calculate_full_xva(
    exposure_profile: ExposureProfile,
    counterparty_credit: CreditCurve,
    own_credit: CreditCurve,
    funding_curve: Optional[FundingCurve] = None,
    initial_margin_profile: Optional[np.ndarray] = None,
    capital_profile: Optional[np.ndarray] = None,
    collateral_rate: float = 0.0,
    risk_free_rate: float = 0.05,
    cost_of_capital: float = 0.10,
    funding_spread: float = 0.01,
    discount_curve: Optional[Callable[[float], float]] = None
) -> XVAResult:
    """
    Calculate all XVA components.

    Args:
        exposure_profile: Expected exposure profile
        counterparty_credit: Counterparty credit curve
        own_credit: Own credit curve
        funding_curve: Funding spreads curve
        initial_margin_profile: Initial margin profile (optional)
        capital_profile: Regulatory capital profile (optional)
        collateral_rate: Rate on posted collateral
        risk_free_rate: Risk-free rate
        cost_of_capital: Hurdle rate for capital
        funding_spread: Funding spread for margin
        discount_curve: Discount factor function

    Returns:
        XVAResult with all components
    """
    if discount_curve is None:
        discount_curve = lambda t: np.exp(-risk_free_rate * t)

    # CVA
    cva = calculate_cva(exposure_profile, counterparty_credit, discount_curve)

    # DVA
    dva = calculate_dva(exposure_profile, own_credit, discount_curve)

    # FVA
    fva = 0.0
    if funding_curve is not None:
        fva = calculate_fva(exposure_profile, funding_curve, discount_curve)

    # ColVA
    colva = 0.0
    if collateral_rate != risk_free_rate:
        colva = calculate_colva(
            exposure_profile, collateral_rate, risk_free_rate, discount_curve
        )

    # MVA
    mva = 0.0
    if initial_margin_profile is not None:
        mva = calculate_mva(
            initial_margin_profile, exposure_profile.times,
            funding_spread, discount_curve
        )

    # KVA
    kva = 0.0
    if capital_profile is not None:
        kva = calculate_kva(
            capital_profile, exposure_profile.times,
            cost_of_capital, discount_curve
        )

    return XVAResult(
        cva=cva,
        dva=dva,
        fva=fva,
        colva=colva,
        mva=mva,
        kva=kva
    )


# =============================================================================
# Exposure Generation
# =============================================================================

def generate_exposure_profile_from_paths(
    mtm_paths: np.ndarray,
    times: np.ndarray,
    confidence_level: float = 0.975
) -> ExposureProfile:
    """
    Generate exposure profile from mark-to-market paths.

    Args:
        mtm_paths: MTM values, shape (num_paths, num_times)
        times: Time points
        confidence_level: Confidence level for PFE

    Returns:
        ExposureProfile
    """
    # Expected Exposure (positive part)
    positive_mtm = np.maximum(mtm_paths, 0)
    ee = np.mean(positive_mtm, axis=0)

    # Expected Negative Exposure
    negative_mtm = np.minimum(mtm_paths, 0)
    ene = np.mean(negative_mtm, axis=0)

    # Potential Future Exposure (quantile)
    pfe = np.percentile(mtm_paths, confidence_level * 100, axis=0)

    return ExposureProfile(
        times=times,
        expected_exposure=ee,
        expected_negative_exposure=ene,
        potential_future_exposure=pfe
    )


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("XVA Module - Demo")
    print("=" * 60)

    # Create exposure profile for a 5-year interest rate swap
    times = np.array([0, 0.5, 1, 2, 3, 4, 5])

    # Hump-shaped exposure profile typical of IRS
    ee = np.array([0, 50000, 100000, 150000, 125000, 75000, 0])
    ene = np.array([0, -30000, -60000, -90000, -75000, -45000, 0])

    exposure_profile = ExposureProfile(
        times=times,
        expected_exposure=ee,
        expected_negative_exposure=ene
    )

    # Credit curves
    credit_tenors = np.array([1, 2, 3, 5, 7, 10])

    counterparty_credit = CreditCurve(
        tenors=credit_tenors,
        spreads=np.array([0.005, 0.006, 0.007, 0.008, 0.009, 0.01]),  # 50-100bps
        recovery_rate=0.4
    )

    own_credit = CreditCurve(
        tenors=credit_tenors,
        spreads=np.array([0.003, 0.004, 0.004, 0.005, 0.005, 0.006]),  # 30-60bps
        recovery_rate=0.4
    )

    # Funding curve
    funding_curve = FundingCurve(
        tenors=credit_tenors,
        borrowing_spreads=np.array([0.004, 0.005, 0.005, 0.006, 0.006, 0.007]),
        lending_spreads=np.array([0.001, 0.001, 0.002, 0.002, 0.002, 0.003])
    )

    # Calculate XVA
    print("\n1. Individual XVA Components")
    print("-" * 40)

    cva = calculate_cva(exposure_profile, counterparty_credit)
    print(f"   CVA: ${cva:,.0f}")

    dva = calculate_dva(exposure_profile, own_credit)
    print(f"   DVA: ${dva:,.0f}")

    fva = calculate_fva(exposure_profile, funding_curve)
    print(f"   FVA: ${fva:,.0f}")

    # Full XVA
    print("\n2. Full XVA Calculation")
    print("-" * 40)

    result = calculate_full_xva(
        exposure_profile=exposure_profile,
        counterparty_credit=counterparty_credit,
        own_credit=own_credit,
        funding_curve=funding_curve
    )

    print(f"   CVA:       ${result.cva:,.0f}")
    print(f"   DVA:       ${result.dva:,.0f}")
    print(f"   FVA:       ${result.fva:,.0f}")
    print(f"   Bilateral: ${result.bilateral_cva:,.0f}")
    print(f"   Total XVA: ${result.total_xva:,.0f}")

    print("\n" + "=" * 60)
    print("Demo Complete!")
