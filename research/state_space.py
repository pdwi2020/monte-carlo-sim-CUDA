"""Sequential/state-space calibration diagnostics for time-varying Heston params."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence

import numpy as np

from calibration import MarketOption, calibrate_heston, heston_call_price, implied_volatility
from .historical_backtest import DatedOptionPanel

_PARAM_ORDER = ["v0", "kappa", "theta", "xi", "rho"]


@dataclass
class StateEstimate:
    """Filtered latent-parameter estimate for one quote date."""

    quote_date: str
    parameters: Dict[str, float]
    raw_calibration_parameters: Dict[str, float]
    raw_panel_rmse: float
    innovation_l2: float
    panel_rmse: float


@dataclass
class StateSpaceCalibrationResult:
    """Output of sequential Heston state-space filtering."""

    estimates: List[StateEstimate]
    mean_innovation_l2: float
    max_innovation_l2: float
    mean_panel_rmse: float
    num_dates: int


def _targets(options: Sequence[MarketOption], *, use_iv: bool) -> tuple[List[MarketOption], np.ndarray]:
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


def _panel_rmse(
    options: Sequence[MarketOption],
    *,
    spot: float,
    rate: float,
    params: Dict[str, float],
    use_iv: bool,
) -> float:
    rows, y = _targets(options, use_iv=use_iv)
    pred = []
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
            price = call - spot + opt.strike * np.exp(-rate * opt.maturity)
        else:
            price = call

        if use_iv and opt.market_iv is not None:
            try:
                value = implied_volatility(price, spot, opt.strike, rate, opt.maturity, opt.option_type)
            except Exception:
                value = float(np.sqrt(max(params["v0"], 1e-12)))
        else:
            value = price
        pred.append(float(value))
    return float(np.sqrt(np.mean((np.asarray(pred, dtype=float) - y) ** 2)))


def run_heston_state_space_filter(
    panels: Sequence[DatedOptionPanel],
    *,
    spot: float,
    rate: float,
    use_iv: bool = False,
    max_iter: int = 80,
    process_noise: float = 0.03,
    measurement_noise: float = 0.08,
) -> StateSpaceCalibrationResult:
    """Run a diagonal Kalman-style filter on date-wise Heston calibrations."""

    ordered = sorted(list(panels), key=lambda p: p.quote_date)
    if len(ordered) < 2:
        raise ValueError("Need at least 2 panels for state-space filtering")

    init = calibrate_heston(list(ordered[0].options), spot=spot, rate=rate, use_iv=use_iv, max_iter=max_iter)
    state = np.asarray([float(init.parameters[k]) for k in _PARAM_ORDER], dtype=float)
    p_diag = np.full(5, 0.05, dtype=float)
    q_diag = np.full(5, float(max(process_noise, 1e-6)), dtype=float)
    r_diag = np.full(5, float(max(measurement_noise, 1e-6)), dtype=float)

    rows: List[StateEstimate] = []
    innovations: List[float] = []
    rmses: List[float] = []

    for i, panel in enumerate(ordered):
        raw = calibrate_heston(
            list(panel.options),
            spot=spot,
            rate=rate,
            use_iv=use_iv,
            max_iter=max(25, max_iter // 2),
            initial_params={k: float(v) for k, v in zip(_PARAM_ORDER, state)},
        )
        z = np.asarray([float(raw.parameters[k]) for k in _PARAM_ORDER], dtype=float)

        # Predict and update (diagonal state covariance for robustness/speed).
        p_prior = p_diag + q_diag
        k_gain = p_prior / np.maximum(p_prior + r_diag, 1e-10)
        innovation = z - state
        state = state + k_gain * innovation
        p_diag = (1.0 - k_gain) * p_prior

        # Clamp to admissible parameter region.
        state[0] = float(np.clip(state[0], 0.001, 1.0))    # v0
        state[1] = float(np.clip(state[1], 0.01, 10.0))    # kappa
        state[2] = float(np.clip(state[2], 0.001, 1.0))    # theta
        state[3] = float(np.clip(state[3], 0.01, 2.0))     # xi
        state[4] = float(np.clip(state[4], -0.99, 0.99))   # rho

        params = {k: float(v) for k, v in zip(_PARAM_ORDER, state)}
        raw_rmse = _panel_rmse(panel.options, spot=spot, rate=rate, params=raw.parameters, use_iv=use_iv)
        rmse = _panel_rmse(panel.options, spot=spot, rate=rate, params=params, use_iv=use_iv)
        innov_norm = float(np.linalg.norm(innovation, ord=2))

        innovations.append(innov_norm)
        rmses.append(rmse)
        rows.append(
            StateEstimate(
                quote_date=panel.quote_date,
                parameters=params,
                raw_calibration_parameters={k: float(v) for k, v in raw.parameters.items()},
                raw_panel_rmse=raw_rmse,
                innovation_l2=innov_norm,
                panel_rmse=rmse,
            )
        )

    return StateSpaceCalibrationResult(
        estimates=rows,
        mean_innovation_l2=float(np.mean(innovations)),
        max_innovation_l2=float(np.max(innovations)),
        mean_panel_rmse=float(np.mean(rmses)),
        num_dates=len(rows),
    )


def state_space_to_dict(result: StateSpaceCalibrationResult) -> Dict[str, object]:
    """Serialize state-space calibration result."""

    return {
        "estimates": [asdict(x) for x in result.estimates],
        "mean_innovation_l2": result.mean_innovation_l2,
        "max_innovation_l2": result.max_innovation_l2,
        "mean_panel_rmse": result.mean_panel_rmse,
        "num_dates": result.num_dates,
    }
