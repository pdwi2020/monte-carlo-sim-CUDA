"""Lightweight rough-Bergomi baseline for model-class comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from calibration import MarketOption, calibrate_heston, heston_call_price, implied_volatility


@dataclass
class RoughBergomiParams:
    """Approximate rough-Bergomi parameterization for calibration studies."""

    xi0: float
    eta: float
    rho: float
    hurst: float


@dataclass
class RoughBergomiCalibrationResult:
    """Calibration hook output for rough-Bergomi-like model."""

    base_heston_params: Dict[str, float]
    base_rmse: float
    rough_bergomi_rmse: float
    improvement_ratio: float
    best_params: Dict[str, float]
    grid_results: List[Dict[str, float]]


def _targets(options: Sequence[MarketOption], *, use_iv: bool) -> Tuple[List[MarketOption], np.ndarray]:
    rows: List[MarketOption] = []
    y: List[float] = []
    for opt in options:
        if use_iv and opt.market_iv is not None:
            rows.append(opt)
            y.append(float(opt.market_iv))
        elif opt.market_price is not None:
            rows.append(opt)
            y.append(float(opt.market_price))
        elif opt.mid_price is not None:
            rows.append(opt)
            y.append(float(opt.mid_price))
    if len(rows) < 5:
        raise ValueError("Need at least 5 usable options")
    return rows, np.asarray(y, dtype=float)


def _rb_price_like_heston(
    opt: MarketOption,
    *,
    spot: float,
    rate: float,
    xi0: float,
    eta: float,
    rho: float,
    hurst: float,
) -> float:
    # Map rough-Bergomi-like params into an effective Heston vol-of-vol profile.
    t = max(float(opt.maturity), 1e-6)
    v0 = max(float(xi0), 1e-8)
    theta = v0
    kappa = float(np.clip(1.4 + 0.8 * (0.5 - hurst), 0.05, 8.0))
    xi_eff = float(np.clip(eta * (t ** (0.5 - hurst)), 0.01, 2.0))
    call = heston_call_price(
        spot,
        opt.strike,
        rate,
        opt.maturity,
        v0,
        kappa,
        theta,
        xi_eff,
        float(np.clip(rho, -0.99, 0.99)),
    )
    if opt.option_type == "put":
        return float(call - spot + opt.strike * np.exp(-rate * opt.maturity))
    return float(call)


def _rb_panel_rmse(
    options: Sequence[MarketOption],
    *,
    spot: float,
    rate: float,
    params: RoughBergomiParams,
    use_iv: bool,
) -> float:
    rows, y = _targets(options, use_iv=use_iv)
    pred: List[float] = []
    for opt in rows:
        price = _rb_price_like_heston(
            opt,
            spot=spot,
            rate=rate,
            xi0=params.xi0,
            eta=params.eta,
            rho=params.rho,
            hurst=params.hurst,
        )
        if use_iv and opt.market_iv is not None:
            try:
                val = implied_volatility(price, spot, opt.strike, rate, opt.maturity, opt.option_type)
            except Exception:
                val = float(np.sqrt(max(params.xi0, 1e-12)))
        else:
            val = price
        pred.append(float(val))
    return float(np.sqrt(np.mean((np.asarray(pred) - y) ** 2)))


def calibrate_rough_bergomi_hook(
    market_options: Sequence[MarketOption],
    *,
    spot: float,
    rate: float,
    use_iv: bool = False,
    max_iter: int = 120,
    hurst_grid: Optional[Sequence[float]] = None,
    eta_grid: Optional[Sequence[float]] = None,
) -> RoughBergomiCalibrationResult:
    """Calibrate a rough-Bergomi baseline via grid refinement around Heston level."""

    if hurst_grid is None:
        hurst_grid = [0.08, 0.12, 0.20, 0.35, 0.5]
    if eta_grid is None:
        eta_grid = [0.20, 0.35, 0.50, 0.70]

    rows, _ = _targets(list(market_options), use_iv=use_iv)
    base = calibrate_heston(list(rows), spot=spot, rate=rate, use_iv=use_iv, max_iter=max_iter)
    base_rmse = float(base.rmse)
    base_v0 = float(base.parameters["v0"])
    base_rho = float(base.parameters["rho"])

    best_rmse = float("inf")
    best_params: Dict[str, float] = {}
    grid_results: List[Dict[str, float]] = []

    for h in hurst_grid:
        for eta in eta_grid:
            cand = RoughBergomiParams(
                xi0=float(np.clip(base_v0, 1e-5, 1.0)),
                eta=float(np.clip(eta, 0.01, 2.0)),
                rho=float(np.clip(base_rho, -0.99, 0.99)),
                hurst=float(np.clip(h, 0.01, 0.5)),
            )
            rmse = _rb_panel_rmse(rows, spot=spot, rate=rate, params=cand, use_iv=use_iv)
            row = {"xi0": cand.xi0, "eta": cand.eta, "rho": cand.rho, "hurst": cand.hurst, "rmse": rmse}
            grid_results.append(row)
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = {k: float(v) for k, v in row.items() if k != "rmse"}

    return RoughBergomiCalibrationResult(
        base_heston_params={k: float(v) for k, v in base.parameters.items()},
        base_rmse=base_rmse,
        rough_bergomi_rmse=float(best_rmse),
        improvement_ratio=float((base_rmse - best_rmse) / max(base_rmse, 1e-12)),
        best_params=best_params,
        grid_results=sorted(grid_results, key=lambda x: x["rmse"]),
    )


def rough_bergomi_to_dict(result: RoughBergomiCalibrationResult) -> Dict[str, object]:
    """Serialize rough-Bergomi calibration diagnostics."""

    return {
        "base_heston_params": dict(result.base_heston_params),
        "base_rmse": result.base_rmse,
        "rough_bergomi_rmse": result.rough_bergomi_rmse,
        "improvement_ratio": result.improvement_ratio,
        "best_params": dict(result.best_params),
        "grid_results": [dict(x) for x in result.grid_results],
    }
