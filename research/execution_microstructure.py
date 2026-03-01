"""Microstructure-aware execution stress tests for hedging strategies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np

from mc_pricer import HestonParams

from .hedging import _delta_hedge_pnl  # internal path-level engine reuse


@dataclass
class MicrostructureModel:
    """Execution/microstructure model parameters."""

    spread_bps: float = 8.0
    latency_steps: int = 1
    queue_fill_base: float = 0.92
    queue_fill_slope: float = 0.12
    temp_impact_bps: float = 4.0
    perm_impact_bps: float = 1.5


def _summarize_pnl(pnl: np.ndarray) -> Dict[str, float]:
    losses = -pnl
    var95 = float(np.quantile(losses, 0.95))
    tail = losses[losses >= var95]
    cvar95 = float(np.mean(tail)) if tail.size > 0 else var95
    return {
        "mean_pnl": float(np.mean(pnl)),
        "std_pnl": float(np.std(pnl, ddof=1)),
        "rmse_pnl": float(np.sqrt(np.mean(pnl**2))),
        "var95_loss": var95,
        "cvar95_loss": cvar95,
    }


def run_microstructure_hedging_study(
    *,
    true_model: str = "heston",
    s0: float = 100.0,
    strike: float = 100.0,
    r: float = 0.03,
    maturity: float = 1.0,
    sigma_hedger: float = 0.2,
    sigma_true: float = 0.2,
    heston_true: Optional[HestonParams] = None,
    num_paths: int = 2500,
    num_steps: int = 52,
    rebalance_every: int = 1,
    transaction_cost: float = 0.0005,
    microstructure: Optional[MicrostructureModel] = None,
    seed: int = 42,
) -> Dict[str, object]:
    """Stress delta-hedging PnL with latency/queue/impact microstructure effects."""

    model = microstructure or MicrostructureModel()
    base_pnl = _delta_hedge_pnl(
        true_model=true_model,
        s0=s0,
        strike=strike,
        r=r,
        maturity=maturity,
        sigma_hedger=sigma_hedger,
        sigma_true=sigma_true,
        heston_true=heston_true,
        is_call=True,
        num_paths=num_paths,
        num_steps=num_steps,
        rebalance_every=rebalance_every,
        transaction_cost=transaction_cost,
        execution_model=None,
        seed=seed,
    )

    rng = np.random.default_rng(seed + 9000)
    latency_penalty = max(model.latency_steps, 0) / max(num_steps, 1)
    trade_intensity = np.abs(rng.normal(1.0, 0.35, size=num_paths))
    spread_cost = model.spread_bps * 1e-4 * s0 * trade_intensity * (num_steps / max(rebalance_every, 1)) * 0.02
    temp_impact = model.temp_impact_bps * 1e-4 * s0 * (trade_intensity**1.3) * 0.02
    perm_impact = model.perm_impact_bps * 1e-4 * s0 * trade_intensity * latency_penalty

    fill_prob = np.clip(model.queue_fill_base - model.queue_fill_slope * latency_penalty, 0.4, 1.0)
    partial_fill = rng.binomial(1, fill_prob, size=num_paths).astype(float)
    fill_drag = (1.0 - partial_fill) * np.abs(rng.normal(0.0, 0.65, size=num_paths))

    micro_pnl = base_pnl - spread_cost - temp_impact - perm_impact - fill_drag

    base = _summarize_pnl(base_pnl)
    stressed = _summarize_pnl(micro_pnl)
    return {
        "frictionless": base,
        "microstructure_stressed": stressed,
        "microstructure_model": asdict(model),
        "effective_fill_probability": float(fill_prob),
        "cvar95_degradation_ratio": float(stressed["cvar95_loss"] / max(base["cvar95_loss"], 1e-12)),
        "var95_degradation_ratio": float(stressed["var95_loss"] / max(base["var95_loss"], 1e-12)),
    }
