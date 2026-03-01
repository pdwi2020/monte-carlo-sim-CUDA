"""Rough-Heston simulation and lightweight calibration hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from calibration import CalibrationResult, MarketOption, calibrate_heston, heston_call_price, implied_volatility


@dataclass
class RoughHestonParams:
    """Rough-Heston parameters with Hurst exponent."""

    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float
    hurst: float

    def __post_init__(self) -> None:
        if self.v0 < 0:
            raise ValueError("v0 must be non-negative")
        if self.kappa <= 0:
            raise ValueError("kappa must be positive")
        if self.theta < 0:
            raise ValueError("theta must be non-negative")
        if self.xi <= 0:
            raise ValueError("xi must be positive")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError("rho must be in [-1, 1]")
        if not 0.0 < self.hurst <= 0.5:
            raise ValueError("hurst must be in (0, 0.5]")


@dataclass
class RoughHestonPriceResult:
    """Rough-Heston Monte Carlo pricing summary."""

    price: float
    std_error: float
    num_paths: int
    num_steps: int
    hurst: float


@dataclass
class RoughHestonCalibrationResult:
    """Calibration hook diagnostics for rough-Heston refinement."""

    base_heston: CalibrationResult
    base_rmse: float
    rough_rmse: float
    improvement_ratio: float
    best_params: Dict[str, float]
    grid_results: List[Dict[str, float]]
    num_paths: int
    num_steps: int


def rough_params_from_dict(params: Dict[str, float]) -> RoughHestonParams:
    """Build RoughHestonParams from serialized dict-like payload."""

    return RoughHestonParams(
        v0=float(params["v0"]),
        kappa=float(params["kappa"]),
        theta=float(params["theta"]),
        xi=float(params["xi"]),
        rho=float(params["rho"]),
        hurst=float(params["hurst"]),
    )


def _fractional_gaussian_noise(
    num_paths: int,
    num_steps: int,
    hurst: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate fGn increments with unit marginal variance."""

    if abs(hurst - 0.5) < 1e-12:
        return rng.standard_normal((num_paths, num_steps))

    gamma = np.empty(num_steps, dtype=float)
    for k in range(num_steps):
        gamma[k] = 0.5 * (
            abs(k - 1) ** (2.0 * hurst) - 2.0 * (k ** (2.0 * hurst)) + (k + 1) ** (2.0 * hurst)
        )

    idx = np.arange(num_steps)
    cov = gamma[np.abs(idx[:, None] - idx[None, :])]
    cov += 1e-12 * np.eye(num_steps)

    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(cov)
        vals = np.clip(vals, 1e-12, None)
        chol = vecs @ np.diag(np.sqrt(vals))

    z = rng.standard_normal((num_paths, num_steps))
    return z @ chol.T


def simulate_rough_heston_paths(
    *,
    s0: float,
    r: float,
    maturity: float,
    params: RoughHestonParams,
    num_paths: int,
    num_steps: int,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """Simulate rough-Heston style paths via Euler with fractional volatility shocks."""

    rng = np.random.default_rng(seed)
    dt = maturity / num_steps
    sqrt_dt = np.sqrt(dt)

    s = np.empty((num_paths, num_steps + 1), dtype=float)
    s[:, 0] = s0
    v = np.full(num_paths, params.v0, dtype=float)

    z_v = _fractional_gaussian_noise(num_paths=num_paths, num_steps=num_steps, hurst=params.hurst, rng=rng)
    z_s_orth = rng.standard_normal((num_paths, num_steps))
    rho_comp = np.sqrt(max(1.0 - params.rho ** 2, 0.0))

    for t in range(num_steps):
        v_pos = np.maximum(v, 0.0)
        dWv = sqrt_dt * z_v[:, t]
        dWs = sqrt_dt * (params.rho * z_v[:, t] + rho_comp * z_s_orth[:, t])
        s[:, t + 1] = s[:, t] * np.exp((r - 0.5 * v_pos) * dt + np.sqrt(v_pos) * dWs)
        v_next = v + params.kappa * (params.theta - v_pos) * dt + params.xi * np.sqrt(v_pos) * dWv
        # Numerical guardrail for coarse-grid rough simulations.
        v = np.clip(v_next, 0.0, 4.0)

    return s


def rough_heston_price_option(
    *,
    s0: float,
    strike: float,
    r: float,
    maturity: float,
    params: RoughHestonParams,
    is_call: bool = True,
    payoff_style: str = "european",
    num_paths: int = 4000,
    num_steps: int = 80,
    seed: Optional[int] = 42,
) -> RoughHestonPriceResult:
    """Price vanilla/asian options under rough-Heston simulation."""

    paths = simulate_rough_heston_paths(
        s0=s0,
        r=r,
        maturity=maturity,
        params=params,
        num_paths=num_paths,
        num_steps=num_steps,
        seed=seed,
    )

    if payoff_style == "european":
        underlier = paths[:, -1]
    elif payoff_style == "asian":
        underlier = np.mean(paths[:, 1:], axis=1)
    else:
        raise ValueError("payoff_style must be 'european' or 'asian'")

    if is_call:
        payoff = np.maximum(underlier - strike, 0.0)
    else:
        payoff = np.maximum(strike - underlier, 0.0)
    discounted = np.exp(-r * maturity) * payoff

    return RoughHestonPriceResult(
        price=float(np.mean(discounted)),
        std_error=float(np.std(discounted, ddof=1) / np.sqrt(num_paths)),
        num_paths=num_paths,
        num_steps=num_steps,
        hurst=params.hurst,
    )


def _prepare_targets(options: Sequence[MarketOption], use_iv: bool) -> Tuple[List[MarketOption], np.ndarray]:
    filtered: List[MarketOption] = []
    target_values: List[float] = []
    for opt in options:
        if use_iv and opt.market_iv is not None:
            filtered.append(opt)
            target_values.append(float(opt.market_iv))
        elif opt.market_price is not None:
            filtered.append(opt)
            target_values.append(float(opt.market_price))
        elif opt.mid_price is not None:
            filtered.append(opt)
            target_values.append(float(opt.mid_price))
    if len(filtered) < 5:
        raise ValueError("Need at least 5 options with usable market observations")
    return filtered, np.asarray(target_values, dtype=float)


def _rough_model_values(
    *,
    options: Sequence[MarketOption],
    spot: float,
    rate: float,
    params: RoughHestonParams,
    use_iv: bool,
    num_paths: int,
    num_steps: int,
    seed: int,
) -> np.ndarray:
    """Compute model values using a roughness-adjusted Heston approximation."""

    model_values: List[float] = []
    del num_paths, num_steps, seed

    for opt in options:
        maturity_scale = max(float(opt.maturity), 1e-4)
        xi_eff = float(np.clip(params.xi * (maturity_scale ** (0.5 - params.hurst)), 0.01, 2.0))
        kappa_eff = float(np.clip(params.kappa * (maturity_scale ** (params.hurst - 0.5)), 0.01, 10.0))
        call_price = heston_call_price(
            spot,
            opt.strike,
            rate,
            opt.maturity,
            params.v0,
            kappa_eff,
            params.theta,
            xi_eff,
            params.rho,
        )
        if opt.option_type == "put":
            model_price = call_price - spot + opt.strike * np.exp(-rate * opt.maturity)
        else:
            model_price = call_price

        if use_iv and opt.market_iv is not None:
            try:
                val = implied_volatility(model_price, spot, opt.strike, rate, opt.maturity, opt.option_type)
            except Exception:
                val = np.sqrt(max(params.v0, 1e-10))
        else:
            val = model_price
        model_values.append(float(val))

    return np.asarray(model_values, dtype=float)


def calibrate_rough_heston_hook(
    market_options: Sequence[MarketOption],
    *,
    spot: float,
    rate: float,
    use_iv: bool = False,
    max_iter: int = 180,
    hurst_grid: Optional[Sequence[float]] = None,
    xi_scales: Optional[Sequence[float]] = None,
    num_paths: int = 700,
    num_steps: int = 36,
    seed: int = 42,
) -> RoughHestonCalibrationResult:
    """Start from Heston calibration and refine roughness/vol-of-vol by grid search."""

    if hurst_grid is None:
        hurst_grid = [0.12, 0.20, 0.30, 0.50]
    if xi_scales is None:
        xi_scales = [0.85, 1.0, 1.15]

    options, targets = _prepare_targets(list(market_options), use_iv=use_iv)
    base = calibrate_heston(
        list(options),
        spot=spot,
        rate=rate,
        use_iv=use_iv,
        max_iter=max_iter,
    )

    base_params = dict(base.parameters)
    best_rmse = float("inf")
    best_params: Dict[str, float] = {}
    grid_results: List[Dict[str, float]] = []

    for h in hurst_grid:
        for scale in xi_scales:
            candidate = RoughHestonParams(
                v0=float(base_params["v0"]),
                kappa=float(base_params["kappa"]),
                theta=float(base_params["theta"]),
                xi=float(np.clip(base_params["xi"] * scale, 0.01, 2.0)),
                rho=float(np.clip(base_params["rho"], -0.99, 0.99)),
                hurst=float(h),
            )
            model_values = _rough_model_values(
                options=options,
                spot=spot,
                rate=rate,
                params=candidate,
                use_iv=use_iv,
                num_paths=num_paths,
                num_steps=num_steps,
                seed=seed,
            )
            rmse = float(np.sqrt(np.mean((model_values - targets) ** 2)))
            row = {
                "hurst": candidate.hurst,
                "v0": candidate.v0,
                "kappa": candidate.kappa,
                "theta": candidate.theta,
                "xi": candidate.xi,
                "rho": candidate.rho,
                "rmse": rmse,
            }
            grid_results.append(row)
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = {k: float(v) for k, v in row.items() if k != "rmse"}

    improvement = (float(base.rmse) - best_rmse) / max(float(base.rmse), 1e-12)
    grid_results.sort(key=lambda x: x["rmse"])

    return RoughHestonCalibrationResult(
        base_heston=base,
        base_rmse=float(base.rmse),
        rough_rmse=float(best_rmse),
        improvement_ratio=float(improvement),
        best_params=best_params,
        grid_results=grid_results,
        num_paths=num_paths,
        num_steps=num_steps,
    )


def rough_heston_panel_rmse(
    market_options: Sequence[MarketOption],
    *,
    spot: float,
    rate: float,
    params: RoughHestonParams,
    use_iv: bool = False,
    num_paths: int = 700,
    num_steps: int = 36,
    seed: int = 42,
) -> float:
    """Evaluate rough-Heston panel RMSE for an option set."""

    options, targets = _prepare_targets(list(market_options), use_iv=use_iv)
    model_values = _rough_model_values(
        options=options,
        spot=spot,
        rate=rate,
        params=params,
        use_iv=use_iv,
        num_paths=num_paths,
        num_steps=num_steps,
        seed=seed,
    )
    return float(np.sqrt(np.mean((model_values - targets) ** 2)))


def rough_heston_calibration_to_dict(result: RoughHestonCalibrationResult) -> Dict[str, object]:
    """Serialize rough-Heston calibration hook output."""

    return {
        "base_heston": {
            "success": result.base_heston.success,
            "parameters": dict(result.base_heston.parameters),
            "objective_value": float(result.base_heston.objective_value),
            "rmse": float(result.base_heston.rmse),
            "num_iterations": int(result.base_heston.num_iterations),
            "calibration_time": float(result.base_heston.calibration_time),
            "message": result.base_heston.message,
        },
        "base_rmse": result.base_rmse,
        "rough_rmse": result.rough_rmse,
        "improvement_ratio": result.improvement_ratio,
        "best_params": dict(result.best_params),
        "grid_results": [dict(row) for row in result.grid_results],
        "num_paths": result.num_paths,
        "num_steps": result.num_steps,
    }
