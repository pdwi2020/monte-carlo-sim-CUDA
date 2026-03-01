"""Full rough-Heston calibration engine with Monte Carlo objective."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from calibration import MarketOption, implied_volatility
from .rough_heston import RoughHestonParams, calibrate_rough_heston_hook, simulate_rough_heston_paths


@dataclass
class RoughFullCalibrationResult:
    """Result of full rough-Heston calibration."""

    best_params: Dict[str, float]
    rmse: float
    global_evaluations: int
    local_iterations: int
    objective_trace: List[Dict[str, float]]
    num_paths: int
    num_steps: int


_PARAM_BOUNDS = {
    "v0": (0.001, 0.5),
    "kappa": (0.05, 8.0),
    "theta": (0.001, 0.5),
    "xi": (0.05, 1.5),
    "rho": (-0.99, 0.99),
    "hurst": (0.05, 0.5),
}
_PARAM_ORDER = ["v0", "kappa", "theta", "xi", "rho", "hurst"]


def _panel_model_values(
    options: Sequence[MarketOption],
    *,
    spot: float,
    rate: float,
    params: RoughHestonParams,
    use_iv: bool,
    num_paths: int,
    num_steps: int,
    seed: int,
) -> np.ndarray:
    by_maturity: Dict[float, List[MarketOption]] = {}
    for opt in options:
        by_maturity.setdefault(float(opt.maturity), []).append(opt)

    out: List[float] = []
    for i, maturity in enumerate(sorted(by_maturity.keys())):
        panel = by_maturity[maturity]
        local_steps = max(8, int(np.ceil(num_steps * max(maturity, 0.05))))
        paths = simulate_rough_heston_paths(
            s0=spot,
            r=rate,
            maturity=maturity,
            params=params,
            num_paths=num_paths,
            num_steps=local_steps,
            seed=seed + 100 * i,
        )
        terminal = paths[:, -1]
        disc = np.exp(-rate * maturity)

        strike_cache: Dict[Tuple[float, str], float] = {}
        for opt in panel:
            key = (float(opt.strike), opt.option_type)
            if key in strike_cache:
                out.append(strike_cache[key])
                continue

            call_price = float(disc * np.mean(np.maximum(terminal - opt.strike, 0.0)))
            if opt.option_type == "put":
                price = call_price - spot + opt.strike * np.exp(-rate * opt.maturity)
            else:
                price = call_price

            if use_iv and opt.market_iv is not None:
                try:
                    model_val = implied_volatility(price, spot, opt.strike, rate, opt.maturity, opt.option_type)
                except Exception:
                    model_val = np.sqrt(max(params.v0, 1e-8))
            else:
                model_val = price

            strike_cache[key] = float(model_val)
            out.append(float(model_val))

    return np.asarray(out, dtype=float)


def _prepare_targets(options: Sequence[MarketOption], *, use_iv: bool) -> Tuple[List[MarketOption], np.ndarray]:
    filtered: List[MarketOption] = []
    targets: List[float] = []
    for opt in options:
        if use_iv and opt.market_iv is not None:
            filtered.append(opt)
            targets.append(float(opt.market_iv))
        elif opt.market_price is not None:
            filtered.append(opt)
            targets.append(float(opt.market_price))
        elif opt.mid_price is not None:
            filtered.append(opt)
            targets.append(float(opt.mid_price))
    if len(filtered) < 5:
        raise ValueError("Need at least 5 options for rough full calibration")
    return filtered, np.asarray(targets, dtype=float)


def _clip_params(params: Dict[str, float]) -> Dict[str, float]:
    out = {}
    for k in _PARAM_ORDER:
        lo, hi = _PARAM_BOUNDS[k]
        out[k] = float(np.clip(params[k], lo, hi))
    return out


def _to_params_obj(params: Dict[str, float]) -> RoughHestonParams:
    return RoughHestonParams(
        v0=params["v0"],
        kappa=params["kappa"],
        theta=params["theta"],
        xi=params["xi"],
        rho=params["rho"],
        hurst=params["hurst"],
    )


def _rmse_for_params(
    params: Dict[str, float],
    *,
    options: Sequence[MarketOption],
    targets: np.ndarray,
    spot: float,
    rate: float,
    use_iv: bool,
    num_paths: int,
    num_steps: int,
    seed: int,
) -> float:
    clipped = _clip_params(params)
    model = _panel_model_values(
        options,
        spot=spot,
        rate=rate,
        params=_to_params_obj(clipped),
        use_iv=use_iv,
        num_paths=num_paths,
        num_steps=num_steps,
        seed=seed,
    )
    return float(np.sqrt(np.mean((model - targets) ** 2)))


def calibrate_rough_heston_full(
    market_options: Sequence[MarketOption],
    *,
    spot: float,
    rate: float,
    use_iv: bool = False,
    max_iter: int = 140,
    num_paths: int = 1200,
    num_steps: int = 48,
    n_global_samples: int = 24,
    local_iterations: int = 18,
    seed: int = 42,
) -> RoughFullCalibrationResult:
    """Calibrate rough-Heston with MC objective via global+local search."""

    options, targets = _prepare_targets(market_options, use_iv=use_iv)

    # Warm start from rough hook.
    hook = calibrate_rough_heston_hook(
        options,
        spot=spot,
        rate=rate,
        use_iv=use_iv,
        max_iter=max_iter,
        num_paths=max(250, num_paths // 4),
        num_steps=max(14, num_steps // 2),
        seed=seed,
    )
    best = _clip_params(dict(hook.best_params))
    best_rmse = _rmse_for_params(
        best,
        options=options,
        targets=targets,
        spot=spot,
        rate=rate,
        use_iv=use_iv,
        num_paths=max(300, num_paths // 3),
        num_steps=max(16, num_steps // 2),
        seed=seed + 1,
    )
    trace: List[Dict[str, float]] = [{"stage": "warm_start", "rmse": best_rmse, **best}]

    rng = np.random.default_rng(seed + 2)

    # Global random search.
    scales = {"v0": 0.03, "kappa": 0.6, "theta": 0.03, "xi": 0.12, "rho": 0.08, "hurst": 0.07}
    for i in range(n_global_samples):
        trial = dict(best)
        for k in _PARAM_ORDER:
            trial[k] += float(scales[k] * rng.normal())
        trial = _clip_params(trial)
        rmse = _rmse_for_params(
            trial,
            options=options,
            targets=targets,
            spot=spot,
            rate=rate,
            use_iv=use_iv,
            num_paths=max(300, num_paths // 3),
            num_steps=max(16, num_steps // 2),
            seed=seed + 100 + i,
        )
        trace.append({"stage": "global", "rmse": rmse, **trial})
        if rmse < best_rmse:
            best_rmse = rmse
            best = trial

    # Local coordinate search with shrinking steps.
    local = dict(best)
    step = {"v0": 0.01, "kappa": 0.2, "theta": 0.01, "xi": 0.04, "rho": 0.03, "hurst": 0.02}
    for i in range(local_iterations):
        improved = False
        for k in _PARAM_ORDER:
            for sign in (-1.0, 1.0):
                trial = dict(local)
                trial[k] += sign * step[k]
                trial = _clip_params(trial)
                rmse = _rmse_for_params(
                    trial,
                    options=options,
                    targets=targets,
                    spot=spot,
                    rate=rate,
                    use_iv=use_iv,
                    num_paths=num_paths,
                    num_steps=num_steps,
                    seed=seed + 500 + i,
                )
                trace.append({"stage": "local", "rmse": rmse, **trial})
                if rmse < best_rmse:
                    best_rmse = rmse
                    best = dict(trial)
                    local = dict(trial)
                    improved = True
        if not improved:
            for k in step:
                step[k] *= 0.5

    return RoughFullCalibrationResult(
        best_params=best,
        rmse=best_rmse,
        global_evaluations=n_global_samples,
        local_iterations=local_iterations,
        objective_trace=trace,
        num_paths=num_paths,
        num_steps=num_steps,
    )


def rough_full_calibration_to_dict(result: RoughFullCalibrationResult) -> Dict[str, object]:
    """Serialize rough full-calibration output."""

    return {
        "best_params": dict(result.best_params),
        "rmse": result.rmse,
        "global_evaluations": result.global_evaluations,
        "local_iterations": result.local_iterations,
        "objective_trace": [dict(x) for x in result.objective_trace],
        "num_paths": result.num_paths,
        "num_steps": result.num_steps,
    }

