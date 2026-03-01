"""Portfolio-level hedging risk overlay diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from mc_pricer import HestonParams

from .hedging import sample_delta_hedge_pnl_paths


@dataclass
class PortfolioAssetRisk:
    """Standalone risk profile for one asset sleeve."""

    symbol: str
    weight: float
    mean_pnl: float
    std_pnl: float
    var95_loss: float
    cvar95_loss: float


@dataclass
class PortfolioOverlayResult:
    """Portfolio aggregation diagnostics from path-level hedging PnL."""

    asset_risks: List[PortfolioAssetRisk]
    correlation_matrix: List[List[float]]
    covariance_matrix: List[List[float]]
    portfolio_mean_pnl: float
    portfolio_std_pnl: float
    portfolio_var95_loss: float
    portfolio_cvar95_loss: float
    weighted_standalone_cvar95_loss: float
    diversification_ratio: float
    concentration_hhi: float
    tail_sample_size: int
    expected_shortfall_contributions: Dict[str, float]


def _risk_from_pnl(symbol: str, weight: float, pnl: np.ndarray) -> PortfolioAssetRisk:
    losses = -pnl
    var95 = float(np.quantile(losses, 0.95))
    tail = losses[losses >= var95]
    cvar95 = float(np.mean(tail)) if tail.size > 0 else var95
    return PortfolioAssetRisk(
        symbol=symbol,
        weight=float(weight),
        mean_pnl=float(np.mean(pnl)),
        std_pnl=float(np.std(pnl, ddof=1)),
        var95_loss=var95,
        cvar95_loss=cvar95,
    )


def run_portfolio_hedging_overlay(
    *,
    symbols: Optional[Sequence[str]] = None,
    weights: Optional[Sequence[float]] = None,
    s0: float = 100.0,
    strike: float = 100.0,
    r: float = 0.03,
    maturity: float = 1.0,
    base_sigma: float = 0.2,
    num_paths: int = 1500,
    num_steps: int = 52,
    transaction_cost: float = 0.0005,
    seed: int = 42,
) -> PortfolioOverlayResult:
    """Build a portfolio risk overlay from multi-asset hedging path distributions."""

    universe = [s.upper() for s in (symbols if symbols is not None else ["AAPL", "MSFT", "SPY", "QQQ"])]
    if len(universe) < 2:
        raise ValueError("Need at least two assets for portfolio overlay")

    if weights is None:
        w = np.full(len(universe), 1.0 / len(universe), dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.size != len(universe):
            raise ValueError("weights length must match symbols length")
        if not np.isfinite(w).all() or np.any(w < 0.0):
            raise ValueError("weights must be finite and non-negative")
        sw = float(np.sum(w))
        if sw <= 0:
            raise ValueError("weights must sum to a positive number")
        w = w / sw

    pnl_rows: List[np.ndarray] = []
    for i, symbol in enumerate(universe):
        sigma_i = float(np.clip(base_sigma * (0.82 + 0.10 * i), 0.08, 0.9))
        sigma_true_i = float(np.clip(sigma_i * (1.0 + 0.07 * ((i % 3) - 1)), 0.08, 1.1))
        s0_i = float(max(20.0, s0 * (1.0 + 0.06 * (i - 1.5))))
        true_model = "heston" if i % 2 == 1 else "gbm"
        heston_true = None
        if true_model == "heston":
            heston_true = HestonParams(
                v0=sigma_true_i**2,
                kappa=2.1 + 0.2 * i,
                theta=sigma_true_i**2,
                xi=0.45 + 0.05 * i,
                rho=-0.65 + 0.05 * np.sin(i),
            )

        pnl_i = sample_delta_hedge_pnl_paths(
            true_model=true_model,
            s0=s0_i,
            strike=strike,
            r=r,
            maturity=maturity,
            sigma_hedger=sigma_i,
            sigma_true=sigma_true_i,
            heston_true=heston_true,
            num_paths=num_paths,
            num_steps=num_steps,
            rebalance_every=1,
            transaction_cost=transaction_cost,
            seed=seed + 101 * i,
        )
        pnl_rows.append(np.asarray(pnl_i, dtype=float))

    pnl_mat = np.vstack(pnl_rows)

    # Add a weak common factor so cross-asset diversification diagnostics are non-degenerate.
    rng = np.random.default_rng(seed + 999)
    common = rng.standard_normal(num_paths)
    common = (common - np.mean(common)) / max(np.std(common, ddof=1), 1e-12)
    for i in range(pnl_mat.shape[0]):
        z = pnl_mat[i]
        z = (z - np.mean(z)) / max(np.std(z, ddof=1), 1e-12)
        mix = 0.82 * z + 0.18 * common
        pnl_mat[i] = float(np.mean(pnl_rows[i])) + float(np.std(pnl_rows[i], ddof=1)) * mix

    portfolio_pnl = np.dot(w, pnl_mat)
    portfolio_losses = -portfolio_pnl
    portfolio_var95 = float(np.quantile(portfolio_losses, 0.95))
    tail_mask = portfolio_losses >= portfolio_var95
    tail_losses = portfolio_losses[tail_mask]
    portfolio_cvar95 = float(np.mean(tail_losses)) if tail_losses.size > 0 else portfolio_var95

    asset_risks = [_risk_from_pnl(symbol, float(weight), pnl_mat[i]) for i, (symbol, weight) in enumerate(zip(universe, w))]
    weighted_standalone_cvar = float(np.sum([x.weight * x.cvar95_loss for x in asset_risks]))
    diversification_ratio = weighted_standalone_cvar / max(portfolio_cvar95, 1e-12)
    concentration_hhi = float(np.sum(w**2))

    asset_losses = -pnl_mat
    tail_count = int(np.sum(tail_mask))
    contributions = {}
    denom = max(portfolio_cvar95, 1e-12)
    for i, symbol in enumerate(universe):
        if tail_count == 0:
            contrib = 0.0
        else:
            contrib = float(w[i] * np.mean(asset_losses[i, tail_mask]) / denom)
        contributions[symbol] = contrib

    corr = np.corrcoef(pnl_mat)
    cov = np.cov(pnl_mat)

    return PortfolioOverlayResult(
        asset_risks=asset_risks,
        correlation_matrix=[[float(v) for v in row] for row in corr.tolist()],
        covariance_matrix=[[float(v) for v in row] for row in cov.tolist()],
        portfolio_mean_pnl=float(np.mean(portfolio_pnl)),
        portfolio_std_pnl=float(np.std(portfolio_pnl, ddof=1)),
        portfolio_var95_loss=portfolio_var95,
        portfolio_cvar95_loss=portfolio_cvar95,
        weighted_standalone_cvar95_loss=weighted_standalone_cvar,
        diversification_ratio=float(diversification_ratio),
        concentration_hhi=concentration_hhi,
        tail_sample_size=tail_count,
        expected_shortfall_contributions=contributions,
    )


def portfolio_overlay_to_dict(result: PortfolioOverlayResult) -> Dict[str, object]:
    """Serialize portfolio overlay result."""

    return {
        "asset_risks": [asdict(x) for x in result.asset_risks],
        "correlation_matrix": [list(row) for row in result.correlation_matrix],
        "covariance_matrix": [list(row) for row in result.covariance_matrix],
        "portfolio_mean_pnl": result.portfolio_mean_pnl,
        "portfolio_std_pnl": result.portfolio_std_pnl,
        "portfolio_var95_loss": result.portfolio_var95_loss,
        "portfolio_cvar95_loss": result.portfolio_cvar95_loss,
        "weighted_standalone_cvar95_loss": result.weighted_standalone_cvar95_loss,
        "diversification_ratio": result.diversification_ratio,
        "concentration_hhi": result.concentration_hhi,
        "tail_sample_size": result.tail_sample_size,
        "expected_shortfall_contributions": dict(result.expected_shortfall_contributions),
    }
