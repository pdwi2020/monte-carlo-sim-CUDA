"""Error decomposition for pricing experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np

from mc_pricer import (
    Backend,
    DiscretizationScheme,
    MarketData,
    PayoffType,
    SimulationConfig,
    VarianceReduction,
    price_option,
)
from .benchmark import BenchmarkCase
from .model_risk import price_model_ensemble, summarize_model_risk


@dataclass
class ErrorDecomposition:
    """Estimated error components for one pricing scenario."""

    discretization_bias: float
    mc_standard_error: float
    model_spread: float
    total_rss_error: float


def estimate_discretization_bias(
    case: BenchmarkCase,
    *,
    coarse_steps: int = 50,
    fine_steps: int = 200,
    num_paths: int = 12000,
    seed: int = 2024,
) -> float:
    """Estimate discretization bias via fine/coarse resolution difference."""

    coarse_cfg = SimulationConfig(
        num_paths=num_paths,
        num_steps=coarse_steps,
        backend=Backend.NUMPY,
        scheme=DiscretizationScheme.EULER,
        variance_reduction=VarianceReduction.ANTITHETIC,
        seed=seed,
    )
    fine_cfg = SimulationConfig(
        num_paths=num_paths,
        num_steps=fine_steps,
        backend=Backend.NUMPY,
        scheme=DiscretizationScheme.QE,
        variance_reduction=VarianceReduction.ANTITHETIC,
        seed=seed,
    )

    coarse = price_option(
        market=case.market,
        K=case.strike,
        T=case.maturity,
        payoff_type=case.payoff_type,
        heston=case.heston,
        jump=case.jump,
        barrier=case.barrier,
        sigma=case.sigma,
        config=coarse_cfg,
    )
    fine = price_option(
        market=case.market,
        K=case.strike,
        T=case.maturity,
        payoff_type=case.payoff_type,
        heston=case.heston,
        jump=case.jump,
        barrier=case.barrier,
        sigma=case.sigma,
        config=fine_cfg,
    )
    return abs(fine.price - coarse.price)


def estimate_error_decomposition(
    case: BenchmarkCase,
    *,
    seed: int = 2024,
    reference_num_paths: int = 20000,
    bias_num_paths: int = 12000,
) -> ErrorDecomposition:
    """Estimate total error as RSS of discretization, sampling, and model spread."""

    reference_cfg = SimulationConfig(
        num_paths=reference_num_paths,
        num_steps=120,
        backend=Backend.NUMPY,
        scheme=DiscretizationScheme.QE,
        variance_reduction=VarianceReduction.ANTITHETIC,
        seed=seed,
    )

    ref = price_option(
        market=case.market,
        K=case.strike,
        T=case.maturity,
        payoff_type=case.payoff_type,
        heston=case.heston,
        jump=case.jump,
        barrier=case.barrier,
        sigma=case.sigma,
        config=reference_cfg,
    )

    disc_bias = estimate_discretization_bias(case, seed=seed, num_paths=bias_num_paths)

    ensemble = price_model_ensemble(
        spot=case.market.S0,
        strike=case.strike,
        rate=case.market.r,
        maturity=case.maturity,
        payoff_type=case.payoff_type,
        sigma=case.sigma or 0.2,
        heston=case.heston,
        jump=case.jump,
        config=reference_cfg,
    )
    model_risk = summarize_model_risk(ensemble).spread

    total = float(np.sqrt(disc_bias ** 2 + ref.std_error ** 2 + model_risk ** 2))

    return ErrorDecomposition(
        discretization_bias=float(disc_bias),
        mc_standard_error=float(ref.std_error),
        model_spread=float(model_risk),
        total_rss_error=total,
    )


def decomposition_to_dict(decomposition: ErrorDecomposition) -> Dict[str, float]:
    """Serialize decomposition."""

    return asdict(decomposition)
