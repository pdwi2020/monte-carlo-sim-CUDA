"""Challenger-model baselines for OOS forecasting (SABR + SSVI-style carry)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from calibration import MarketOption, calibrate_sabr_slice, heston_call_price, implied_volatility, sabr_implied_vol_hagan
from .historical_backtest import DatedOptionPanel
from .svi import SVIParams, fit_svi_surface, svi_total_variance


@dataclass
class ChallengerObservation:
    """One challenger forecast observation."""

    model: str
    origin_date: str
    target_date: str
    rmse: float
    num_options: int


@dataclass
class ChallengerStudyResult:
    """Challenger-study payload."""

    models: List[str]
    target_dates: List[str]
    observations: List[ChallengerObservation]
    losses_by_model: Dict[str, List[float]]
    mean_rmse_by_model: Dict[str, float]
    std_rmse_by_model: Dict[str, float]
    best_model: str
    worst_model: str
    num_transitions: int


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


def _slice_market_ivs(options: Sequence[MarketOption], maturity: float) -> List[MarketOption]:
    out = [o for o in options if abs(float(o.maturity) - float(maturity)) < 1e-10 and o.market_iv is not None]
    return out


def _fit_sabr_by_maturity(
    options: Sequence[MarketOption],
    *,
    spot: float,
    rate: float,
    beta: float = 0.6,
) -> Dict[float, Dict[str, float]]:
    mats = sorted({float(o.maturity) for o in options})
    out: Dict[float, Dict[str, float]] = {}
    for t in mats:
        sl = _slice_market_ivs(options, t)
        if len(sl) < 3:
            continue
        fwd = float(spot * np.exp(rate * t))
        res = calibrate_sabr_slice(sl, fwd, t, beta=beta)
        if res.success:
            out[t] = {k: float(v) for k, v in res.parameters.items()}
    return out


def _predict_sabr(
    target: Sequence[MarketOption],
    *,
    sabr_by_maturity: Dict[float, Dict[str, float]],
    spot: float,
    rate: float,
    use_iv: bool,
) -> float:
    rows, y = _targets(target, use_iv=use_iv)
    pred: List[float] = []
    for opt in rows:
        p = sabr_by_maturity.get(float(opt.maturity))
        if p is None:
            # fallback to BS-like smooth anchor from Heston at moderate parameters
            price = heston_call_price(spot, opt.strike, rate, opt.maturity, 0.04, 2.0, 0.04, 0.3, -0.5)
            if opt.option_type == "put":
                price = price - spot + opt.strike * np.exp(-rate * opt.maturity)
            if use_iv and opt.market_iv is not None:
                try:
                    pred.append(float(implied_volatility(price, spot, opt.strike, rate, opt.maturity, opt.option_type)))
                except Exception:
                    pred.append(0.2)
            else:
                pred.append(float(price))
            continue
        fwd = float(spot * np.exp(rate * opt.maturity))
        iv = sabr_implied_vol_hagan(
            fwd,
            float(opt.strike),
            float(opt.maturity),
            float(p["alpha"]),
            float(p["beta"]),
            float(p["rho"]),
            float(p["nu"]),
        )
        if use_iv and opt.market_iv is not None:
            pred.append(float(iv))
        else:
            # price via BS using SABR IV
            sigma = max(float(iv), 1e-6)
            d1 = (np.log(max(spot, 1e-12) / max(opt.strike, 1e-12)) + (rate + 0.5 * sigma * sigma) * opt.maturity) / (
                sigma * np.sqrt(max(opt.maturity, 1e-8))
            )
            d2 = d1 - sigma * np.sqrt(max(opt.maturity, 1e-8))
            nd1 = 0.5 * (1.0 + np.math.erf(d1 / np.sqrt(2.0)))
            nd2 = 0.5 * (1.0 + np.math.erf(d2 / np.sqrt(2.0)))
            call = spot * nd1 - opt.strike * np.exp(-rate * opt.maturity) * nd2
            if opt.option_type == "put":
                pred.append(float(call - spot + opt.strike * np.exp(-rate * opt.maturity)))
            else:
                pred.append(float(call))
    p_arr = np.asarray(pred, dtype=float)
    return float(np.sqrt(np.mean((p_arr - y) ** 2)))


def _predict_ssvi_carry(
    target: Sequence[MarketOption],
    *,
    params_by_maturity: Dict[float, SVIParams],
    spot: float,
    use_iv: bool,
) -> float:
    rows, y = _targets(target, use_iv=use_iv)
    pred: List[float] = []
    for opt in rows:
        p = params_by_maturity.get(float(opt.maturity))
        if p is None:
            pred.append(float(np.mean(y)))
            continue
        k = float(np.log(max(opt.strike, 1e-12) / max(spot, 1e-12)))
        w = float(max(svi_total_variance(k, p), 1e-10))
        iv = float(np.sqrt(w / max(opt.maturity, 1e-10)))
        if use_iv and opt.market_iv is not None:
            pred.append(iv)
        else:
            pred.append(float(opt.market_price if opt.market_price is not None else opt.mid_price if opt.mid_price is not None else iv))
    p_arr = np.asarray(pred, dtype=float)
    return float(np.sqrt(np.mean((p_arr - y) ** 2)))


def run_challenger_baseline_study(
    panels: Sequence[DatedOptionPanel],
    *,
    spot: float,
    rate: float,
    use_iv: bool = True,
    max_transitions: Optional[int] = None,
) -> ChallengerStudyResult:
    """Run one-step OOS challenger forecasts (SABR + SSVI carry)."""

    ordered = sorted(panels, key=lambda p: p.quote_date)
    if len(ordered) < 3:
        raise ValueError("Need at least 3 dated panels")
    if max_transitions is not None:
        tmax = int(max(1, max_transitions))
        ordered = ordered[: tmax + 1]

    models = ["sabr_hagan_surface", "ssvi_carry_surface"]
    losses: Dict[str, List[float]] = {m: [] for m in models}
    obs: List[ChallengerObservation] = []
    target_dates: List[str] = []

    for i in range(len(ordered) - 1):
        origin = ordered[i]
        target = ordered[i + 1]
        target_dates.append(target.quote_date)

        sabr_by_m = _fit_sabr_by_maturity(origin.options, spot=spot, rate=rate, beta=0.6)
        rmse_sabr = _predict_sabr(target.options, sabr_by_maturity=sabr_by_m, spot=spot, rate=rate, use_iv=use_iv)
        losses["sabr_hagan_surface"].append(rmse_sabr)
        obs.append(
            ChallengerObservation(
                model="sabr_hagan_surface",
                origin_date=origin.quote_date,
                target_date=target.quote_date,
                rmse=rmse_sabr,
                num_options=len(target.options),
            )
        )

        svi_by_m = fit_svi_surface(origin.options, spot=spot)
        rmse_ssvi = _predict_ssvi_carry(target.options, params_by_maturity=svi_by_m, spot=spot, use_iv=use_iv)
        losses["ssvi_carry_surface"].append(rmse_ssvi)
        obs.append(
            ChallengerObservation(
                model="ssvi_carry_surface",
                origin_date=origin.quote_date,
                target_date=target.quote_date,
                rmse=rmse_ssvi,
                num_options=len(target.options),
            )
        )

    mean_rmse = {m: float(np.mean(v)) for m, v in losses.items()}
    std_rmse = {m: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for m, v in losses.items()}
    ranked = sorted(models, key=lambda m: mean_rmse[m])
    return ChallengerStudyResult(
        models=models,
        target_dates=target_dates,
        observations=obs,
        losses_by_model={k: list(v) for k, v in losses.items()},
        mean_rmse_by_model=mean_rmse,
        std_rmse_by_model=std_rmse,
        best_model=ranked[0],
        worst_model=ranked[-1],
        num_transitions=len(target_dates),
    )


def challenger_to_dict(result: ChallengerStudyResult) -> Dict[str, object]:
    """Serialize challenger-study output."""

    return {
        "models": result.models,
        "target_dates": result.target_dates,
        "observations": [asdict(x) for x in result.observations],
        "losses_by_model": result.losses_by_model,
        "mean_rmse_by_model": result.mean_rmse_by_model,
        "std_rmse_by_model": result.std_rmse_by_model,
        "best_model": result.best_model,
        "worst_model": result.worst_model,
        "num_transitions": result.num_transitions,
    }
