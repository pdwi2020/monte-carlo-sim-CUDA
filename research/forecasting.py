"""Rolling out-of-sample forecasting diagnostics for volatility models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from calibration import MarketOption, calibrate_heston, heston_call_price, implied_volatility
from .historical_backtest import DatedOptionPanel
from .rough_heston import calibrate_rough_heston_hook, rough_heston_panel_rmse, rough_params_from_dict


@dataclass
class RollingForecastObservation:
    """One model forecast loss from origin date to next date."""

    model: str
    origin_date: str
    target_date: str
    rmse: float
    num_options: int


@dataclass
class RollingForecastStudyResult:
    """Rolling out-of-sample forecasting study output."""

    models: List[str]
    target_dates: List[str]
    observations: List[RollingForecastObservation]
    losses_by_model: Dict[str, List[float]]
    mean_rmse_by_model: Dict[str, float]
    std_rmse_by_model: Dict[str, float]
    best_model: str
    worst_model: str
    num_transitions: int


def _extract_targets(options: Sequence[MarketOption], *, use_iv: bool) -> Tuple[List[MarketOption], np.ndarray]:
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
        raise ValueError("Need at least 5 usable option observations")
    return rows, np.asarray(y, dtype=float)


def _heston_panel_rmse(
    options: Sequence[MarketOption],
    *,
    spot: float,
    rate: float,
    params: Dict[str, float],
    use_iv: bool,
) -> float:
    rows, targets = _extract_targets(options, use_iv=use_iv)
    pred: List[float] = []
    for opt in rows:
        call = heston_call_price(
            spot,
            opt.strike,
            rate,
            opt.maturity,
            params["v0"],
            params["kappa"],
            params["theta"],
            params["xi"],
            params["rho"],
        )
        if opt.option_type == "put":
            model_price = call - spot + opt.strike * np.exp(-rate * opt.maturity)
        else:
            model_price = call
        if use_iv and opt.market_iv is not None:
            try:
                val = implied_volatility(model_price, spot, opt.strike, rate, opt.maturity, opt.option_type)
            except Exception:
                val = float(np.sqrt(max(params["v0"], 1e-10)))
        else:
            val = model_price
        pred.append(float(val))
    p = np.asarray(pred, dtype=float)
    return float(np.sqrt(np.mean((p - targets) ** 2)))


def _opt_target(opt: MarketOption, *, use_iv: bool) -> Optional[float]:
    if use_iv and opt.market_iv is not None:
        return float(opt.market_iv)
    if opt.market_price is not None:
        return float(opt.market_price)
    if opt.mid_price is not None:
        return float(opt.mid_price)
    return None


def _naive_last_surface_rmse(
    origin: Sequence[MarketOption],
    target: Sequence[MarketOption],
    *,
    use_iv: bool,
) -> float:
    """Forecast next surface by carrying forward origin values on matching buckets."""

    origin_map: Dict[Tuple[float, float, str], float] = {}
    by_type_values: Dict[str, List[float]] = {"call": [], "put": []}
    for opt in origin:
        y = _opt_target(opt, use_iv=use_iv)
        if y is None:
            continue
        key = (round(float(opt.strike), 8), round(float(opt.maturity), 8), str(opt.option_type))
        origin_map[key] = y
        by_type_values[str(opt.option_type)].append(y)

    default_val = {
        "call": float(np.mean(by_type_values["call"])) if by_type_values["call"] else 0.0,
        "put": float(np.mean(by_type_values["put"])) if by_type_values["put"] else 0.0,
    }

    pred: List[float] = []
    actual: List[float] = []
    for opt in target:
        y = _opt_target(opt, use_iv=use_iv)
        if y is None:
            continue
        key = (round(float(opt.strike), 8), round(float(opt.maturity), 8), str(opt.option_type))
        pred.append(float(origin_map.get(key, default_val[str(opt.option_type)])))
        actual.append(y)

    if len(actual) < 5:
        raise ValueError("Need at least 5 matched target options for naive forecast")
    p = np.asarray(pred, dtype=float)
    a = np.asarray(actual, dtype=float)
    return float(np.sqrt(np.mean((p - a) ** 2)))


def run_rolling_oos_forecast_study(
    panels: Sequence[DatedOptionPanel],
    *,
    spot: float,
    rate: float,
    use_iv: bool = False,
    heston_max_iter: int = 120,
    rough_max_iter: int = 90,
    rough_num_paths: int = 300,
    rough_num_steps: int = 16,
    max_transitions: Optional[int] = None,
    seed: int = 42,
) -> RollingForecastStudyResult:
    """Run rolling one-step-ahead OOS forecast study for multiple model classes."""

    ordered = sorted(panels, key=lambda p: p.quote_date)
    if len(ordered) < 3:
        raise ValueError("Need at least 3 dated panels for rolling forecasts")

    if max_transitions is not None:
        max_t = int(max(1, max_transitions))
        ordered = ordered[: max_t + 1]
        if len(ordered) < 2:
            raise ValueError("Insufficient panels after max_transitions truncation")

    model_names = ["heston", "rough_heston", "naive_last_surface"]
    losses: Dict[str, List[float]] = {m: [] for m in model_names}
    observations: List[RollingForecastObservation] = []
    target_dates: List[str] = []

    for i in range(len(ordered) - 1):
        origin = ordered[i]
        target = ordered[i + 1]
        target_dates.append(target.quote_date)

        heston_fit = calibrate_heston(
            list(origin.options),
            spot=spot,
            rate=rate,
            use_iv=use_iv,
            max_iter=heston_max_iter,
        )
        heston_rmse = _heston_panel_rmse(
            target.options,
            spot=spot,
            rate=rate,
            params=heston_fit.parameters,
            use_iv=use_iv,
        )
        losses["heston"].append(heston_rmse)
        observations.append(
            RollingForecastObservation(
                model="heston",
                origin_date=origin.quote_date,
                target_date=target.quote_date,
                rmse=heston_rmse,
                num_options=len(target.options),
            )
        )

        rough_fit = calibrate_rough_heston_hook(
            origin.options,
            spot=spot,
            rate=rate,
            use_iv=use_iv,
            max_iter=rough_max_iter,
            num_paths=rough_num_paths,
            num_steps=rough_num_steps,
            seed=seed + 100 + i,
        )
        rough_rmse = rough_heston_panel_rmse(
            target.options,
            spot=spot,
            rate=rate,
            params=rough_params_from_dict(rough_fit.best_params),
            use_iv=use_iv,
            num_paths=rough_num_paths,
            num_steps=rough_num_steps,
            seed=seed + 200 + i,
        )
        losses["rough_heston"].append(rough_rmse)
        observations.append(
            RollingForecastObservation(
                model="rough_heston",
                origin_date=origin.quote_date,
                target_date=target.quote_date,
                rmse=rough_rmse,
                num_options=len(target.options),
            )
        )

        naive_rmse = _naive_last_surface_rmse(origin.options, target.options, use_iv=use_iv)
        losses["naive_last_surface"].append(naive_rmse)
        observations.append(
            RollingForecastObservation(
                model="naive_last_surface",
                origin_date=origin.quote_date,
                target_date=target.quote_date,
                rmse=naive_rmse,
                num_options=len(target.options),
            )
        )

    mean_rmse = {m: float(np.mean(v)) for m, v in losses.items()}
    std_rmse = {m: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for m, v in losses.items()}
    sorted_models = sorted(model_names, key=lambda m: mean_rmse[m])

    return RollingForecastStudyResult(
        models=model_names,
        target_dates=target_dates,
        observations=observations,
        losses_by_model={k: list(v) for k, v in losses.items()},
        mean_rmse_by_model=mean_rmse,
        std_rmse_by_model=std_rmse,
        best_model=sorted_models[0],
        worst_model=sorted_models[-1],
        num_transitions=len(target_dates),
    )


def rolling_forecast_to_dict(result: RollingForecastStudyResult) -> Dict[str, object]:
    """Serialize rolling forecast diagnostics."""

    return {
        "models": list(result.models),
        "target_dates": list(result.target_dates),
        "observations": [asdict(x) for x in result.observations],
        "losses_by_model": {k: list(v) for k, v in result.losses_by_model.items()},
        "mean_rmse_by_model": dict(result.mean_rmse_by_model),
        "std_rmse_by_model": dict(result.std_rmse_by_model),
        "best_model": result.best_model,
        "worst_model": result.worst_model,
        "num_transitions": result.num_transitions,
    }
