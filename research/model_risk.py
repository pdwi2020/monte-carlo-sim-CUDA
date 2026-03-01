"""Model risk analysis utilities using multi-model pricing ensembles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np

from mc_pricer import (
    Backend,
    DiscretizationScheme,
    HestonParams,
    JumpParams,
    MarketData,
    PayoffType,
    RoughHestonParams,
    SABRParams,
    SimulationConfig,
    VarianceReduction,
    price_option,
    price_rough_heston_option,
    price_sabr_option,
)


@dataclass
class ModelSnapshot:
    """One model pricing snapshot in the ensemble."""

    model: str
    price: float
    std_error: float
    runtime_seconds: float


@dataclass
class ModelRiskSummary:
    """Model dispersion summary."""

    min_price: float
    max_price: float
    median_price: float
    spread: float
    relative_spread: float


def default_simulation_config(
    seed: Optional[int] = 42,
    num_paths: int = 12000,
    num_steps: int = 100,
) -> SimulationConfig:
    """Balanced simulation config for model comparisons."""

    return SimulationConfig(
        num_paths=num_paths,
        num_steps=num_steps,
        backend=Backend.NUMPY,
        scheme=DiscretizationScheme.QE,
        variance_reduction=VarianceReduction.ANTITHETIC,
        seed=seed,
    )


def price_model_ensemble(
    *,
    spot: float,
    strike: float,
    rate: float,
    maturity: float,
    payoff_type: PayoffType = PayoffType.EUROPEAN_CALL,
    sigma: float = 0.2,
    heston: Optional[HestonParams] = None,
    jump: Optional[JumpParams] = None,
    sabr: Optional[SABRParams] = None,
    rough: Optional[RoughHestonParams] = None,
    config: Optional[SimulationConfig] = None,
) -> List[ModelSnapshot]:
    """Price the same instrument under multiple models for model-risk spread."""

    if config is None:
        config = default_simulation_config()

    market = MarketData(S0=spot, r=rate, q=0.0)
    heston_params = heston or HestonParams(v0=sigma ** 2, kappa=2.0, theta=sigma ** 2, xi=0.4, rho=-0.6)
    jump_params = jump or JumpParams(lambda_j=0.3, mu_j=-0.08, sigma_j=0.15)
    sabr_params = sabr or SABRParams(alpha=0.3, beta=0.6, rho=-0.3, nu=0.2)
    rough_params = rough or RoughHestonParams(v0=sigma ** 2, theta=sigma ** 2, lambda_=2.0, nu=0.3, rho=-0.5, H=0.15)

    snapshots: List[ModelSnapshot] = []

    gbm = price_option(
        market=market,
        K=strike,
        T=maturity,
        payoff_type=payoff_type,
        sigma=sigma,
        config=config,
    )
    snapshots.append(ModelSnapshot("gbm", gbm.price, gbm.std_error, float(gbm.elapsed_time or 0.0)))

    heston_res = price_option(
        market=market,
        K=strike,
        T=maturity,
        payoff_type=payoff_type,
        heston=heston_params,
        config=config,
    )
    snapshots.append(ModelSnapshot("heston", heston_res.price, heston_res.std_error, float(heston_res.elapsed_time or 0.0)))

    bates_res = price_option(
        market=market,
        K=strike,
        T=maturity,
        payoff_type=payoff_type,
        heston=heston_params,
        jump=jump_params,
        config=config,
    )
    snapshots.append(ModelSnapshot("bates", bates_res.price, bates_res.std_error, float(bates_res.elapsed_time or 0.0)))

    # SABR implementation supports European call/put through its own routine.
    if payoff_type in (PayoffType.EUROPEAN_CALL, PayoffType.EUROPEAN_PUT):
        sabr_res = price_sabr_option(
            F0=spot,
            K=strike,
            T=maturity,
            r=rate,
            sabr=sabr_params,
            is_call=payoff_type == PayoffType.EUROPEAN_CALL,
            config=config,
        )
        snapshots.append(ModelSnapshot("sabr", sabr_res.price, sabr_res.std_error, float(sabr_res.elapsed_time or 0.0)))

        rough_res = price_rough_heston_option(
            market=market,
            K=strike,
            T=maturity,
            params=rough_params,
            payoff_type=payoff_type,
            config=config,
        )
        snapshots.append(ModelSnapshot("rough_heston", rough_res.price, rough_res.std_error, float(rough_res.elapsed_time or 0.0)))

    return snapshots


def summarize_model_risk(snapshots: List[ModelSnapshot]) -> ModelRiskSummary:
    """Summarize ensemble dispersion as a model risk indicator."""

    if not snapshots:
        raise ValueError("snapshots cannot be empty")

    prices = np.array([s.price for s in snapshots], dtype=float)
    min_p = float(np.min(prices))
    max_p = float(np.max(prices))
    median = float(np.median(prices))
    spread = max_p - min_p
    rel = spread / max(abs(median), 1e-12)

    return ModelRiskSummary(
        min_price=min_p,
        max_price=max_p,
        median_price=median,
        spread=spread,
        relative_spread=rel,
    )


def stress_model_ensemble(
    *,
    spot: float,
    strike: float,
    rate: float,
    maturity: float,
    payoff_type: PayoffType = PayoffType.EUROPEAN_CALL,
    sigma: float = 0.2,
    spot_shocks: Optional[List[float]] = None,
    config: Optional[SimulationConfig] = None,
) -> List[Dict[str, object]]:
    """Run shocked spot scenarios and return model-risk spread under stress."""

    if spot_shocks is None:
        spot_shocks = [-0.2, -0.1, 0.0, 0.1, 0.2]

    outputs: List[Dict[str, object]] = []
    for shock in spot_shocks:
        shocked_spot = spot * (1.0 + shock)
        snaps = price_model_ensemble(
            spot=shocked_spot,
            strike=strike,
            rate=rate,
            maturity=maturity,
            payoff_type=payoff_type,
            sigma=sigma,
            config=config,
        )
        summary = summarize_model_risk(snaps)
        outputs.append(
            {
                "spot_shock": shock,
                "spot": shocked_spot,
                "summary": asdict(summary),
                "snapshots": [asdict(s) for s in snaps],
            }
        )

    return outputs
