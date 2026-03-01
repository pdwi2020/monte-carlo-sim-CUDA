"""Date-sliced rough-Heston calibration backtesting for research pipelines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import Dict, Iterable, List, Sequence

import numpy as np

from calibration import MarketOption, heston_call_price, implied_volatility
from .rough_heston import (
    RoughHestonParams,
    calibrate_rough_heston_hook,
    rough_heston_panel_rmse,
    rough_params_from_dict,
)


@dataclass
class DatedOptionPanel:
    """One full option-chain panel for a quote date."""

    quote_date: str
    options: List[MarketOption]
    true_params: Dict[str, float]


@dataclass
class HistoricalPanelEvaluation:
    """Out-of-sample evaluation for one quote date."""

    quote_date: str
    rmse: float
    num_options: int


@dataclass
class ParameterDriftSummary:
    """Parameter-drift diagnostics across date calibrations."""

    mean_l2_drift: float
    std_l2_drift: float
    max_l2_drift: float
    num_transitions: int


@dataclass
class HistoricalRoughBacktestResult:
    """Historical calibration/backtest output payload."""

    train_dates: List[str]
    validate_dates: List[str]
    test_dates: List[str]
    train_num_options: int
    reference_rough_params: Dict[str, float]
    train_rmse: float
    validate_mean_rmse: float
    test_mean_rmse: float
    validate_evaluations: List[HistoricalPanelEvaluation]
    test_evaluations: List[HistoricalPanelEvaluation]
    parameter_drift: ParameterDriftSummary
    date_calibrations: List[Dict[str, float]]


def generate_synthetic_market_option_timeseries(
    *,
    start_date: str = "2024-01-05",
    num_dates: int = 10,
    step_days: int = 7,
    spot: float = 100.0,
    rate: float = 0.03,
    seed: int = 42,
) -> List[DatedOptionPanel]:
    """Generate synthetic dated option panels with smooth parameter drift."""

    if num_dates < 4:
        raise ValueError("num_dates must be at least 4")

    rng = np.random.default_rng(seed)
    d0 = date.fromisoformat(start_date)
    strikes = [85, 95, 100, 105, 115]
    maturities = [0.25, 0.5, 1.0]

    panels: List[DatedOptionPanel] = []
    for i in range(num_dates):
        t = i / max(num_dates - 1, 1)
        # Smooth, bounded parameter drift over time.
        true = {
            "v0": float(np.clip(0.04 * (1.0 + 0.15 * np.sin(2.0 * np.pi * t)), 0.01, 0.2)),
            "kappa": float(np.clip(2.2 * (1.0 + 0.10 * np.cos(2.0 * np.pi * t)), 0.5, 5.0)),
            "theta": float(np.clip(0.04 * (1.0 + 0.12 * np.sin(np.pi * t + 0.4)), 0.01, 0.2)),
            "xi": float(np.clip(0.35 * (1.0 + 0.12 * np.cos(np.pi * t + 0.3)), 0.1, 1.0)),
            "rho": float(np.clip(-0.62 + 0.08 * np.sin(np.pi * t), -0.95, -0.2)),
        }

        options: List[MarketOption] = []
        for maturity in maturities:
            for strike in strikes:
                clean = heston_call_price(spot, strike, rate, maturity, **true)
                noisy = max(clean * (1.0 + rng.normal(0.0, 0.006)), 1e-8)
                iv = implied_volatility(noisy, spot, strike, rate, maturity, option_type="call")
                options.append(
                    MarketOption(
                        strike=float(strike),
                        maturity=float(maturity),
                        market_price=float(noisy),
                        market_iv=float(iv),
                        option_type="call",
                    )
                )

        quote_date = (d0 + timedelta(days=i * step_days)).isoformat()
        panels.append(DatedOptionPanel(quote_date=quote_date, options=options, true_params=true))

    return panels


def _split_by_date(
    panels: Sequence[DatedOptionPanel],
    *,
    train_fraction: float,
    validate_fraction: float,
) -> tuple[List[DatedOptionPanel], List[DatedOptionPanel], List[DatedOptionPanel]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0.0 < validate_fraction < 1.0:
        raise ValueError("validate_fraction must be in (0, 1)")
    if train_fraction + validate_fraction >= 1.0:
        raise ValueError("train_fraction + validate_fraction must be < 1")

    ordered = sorted(panels, key=lambda p: p.quote_date)
    n = len(ordered)
    train_end = max(1, int(np.floor(train_fraction * n)))
    val_end = max(train_end + 1, int(np.floor((train_fraction + validate_fraction) * n)))
    val_end = min(val_end, n - 1)

    train = ordered[:train_end]
    validate = ordered[train_end:val_end]
    test = ordered[val_end:]
    if len(validate) == 0 or len(test) == 0:
        raise ValueError("Need non-empty validate and test slices; increase num_dates")
    return train, validate, test


def _flatten_options(panels: Iterable[DatedOptionPanel]) -> List[MarketOption]:
    out: List[MarketOption] = []
    for panel in panels:
        out.extend(panel.options)
    return out


def _param_vector(params: Dict[str, float]) -> np.ndarray:
    return np.asarray(
        [
            float(params["v0"]),
            float(params["kappa"]),
            float(params["theta"]),
            float(params["xi"]),
            float(params["rho"]),
            float(params["hurst"]),
        ],
        dtype=float,
    )


def run_historical_rough_heston_backtest(
    panels: Sequence[DatedOptionPanel],
    *,
    spot: float,
    rate: float,
    use_iv: bool = False,
    train_fraction: float = 0.6,
    validate_fraction: float = 0.2,
    max_iter: int = 120,
    num_paths: int = 500,
    num_steps: int = 28,
    seed: int = 42,
) -> HistoricalRoughBacktestResult:
    """Run date-sliced rough-Heston train/validate/test evaluation and drift diagnostics."""

    if len(panels) < 5:
        raise ValueError("Need at least 5 dated panels for historical backtest")

    train_panels, validate_panels, test_panels = _split_by_date(
        panels,
        train_fraction=train_fraction,
        validate_fraction=validate_fraction,
    )

    train_options = _flatten_options(train_panels)
    train_fit = calibrate_rough_heston_hook(
        train_options,
        spot=spot,
        rate=rate,
        use_iv=use_iv,
        max_iter=max_iter,
        num_paths=num_paths,
        num_steps=num_steps,
        seed=seed,
    )
    ref_params = rough_params_from_dict(train_fit.best_params)

    validate_eval: List[HistoricalPanelEvaluation] = []
    for i, panel in enumerate(validate_panels):
        rmse = rough_heston_panel_rmse(
            panel.options,
            spot=spot,
            rate=rate,
            params=ref_params,
            use_iv=use_iv,
            num_paths=num_paths,
            num_steps=num_steps,
            seed=seed + 50 + i,
        )
        validate_eval.append(
            HistoricalPanelEvaluation(quote_date=panel.quote_date, rmse=rmse, num_options=len(panel.options))
        )

    test_eval: List[HistoricalPanelEvaluation] = []
    for i, panel in enumerate(test_panels):
        rmse = rough_heston_panel_rmse(
            panel.options,
            spot=spot,
            rate=rate,
            params=ref_params,
            use_iv=use_iv,
            num_paths=num_paths,
            num_steps=num_steps,
            seed=seed + 150 + i,
        )
        test_eval.append(HistoricalPanelEvaluation(quote_date=panel.quote_date, rmse=rmse, num_options=len(panel.options)))

    # Per-date calibrations for parameter drift diagnostics.
    date_calibrations: List[Dict[str, float]] = []
    for i, panel in enumerate(sorted(panels, key=lambda x: x.quote_date)):
        fit = calibrate_rough_heston_hook(
            panel.options,
            spot=spot,
            rate=rate,
            use_iv=use_iv,
            max_iter=max(50, max_iter // 2),
            num_paths=max(300, num_paths // 2),
            num_steps=max(18, num_steps // 2),
            seed=seed + 300 + i,
        )
        row = {"quote_date": panel.quote_date, "rmse": fit.rough_rmse, **fit.best_params}
        date_calibrations.append(row)

    drifts: List[float] = []
    for prev, nxt in zip(date_calibrations[:-1], date_calibrations[1:]):
        p = _param_vector(prev)
        q = _param_vector(nxt)
        drifts.append(float(np.linalg.norm(q - p, ord=2)))
    drift_arr = np.asarray(drifts, dtype=float) if drifts else np.asarray([0.0], dtype=float)
    drift_summary = ParameterDriftSummary(
        mean_l2_drift=float(np.mean(drift_arr)),
        std_l2_drift=float(np.std(drift_arr, ddof=1)) if drift_arr.size > 1 else 0.0,
        max_l2_drift=float(np.max(drift_arr)),
        num_transitions=max(len(date_calibrations) - 1, 0),
    )

    return HistoricalRoughBacktestResult(
        train_dates=[p.quote_date for p in train_panels],
        validate_dates=[p.quote_date for p in validate_panels],
        test_dates=[p.quote_date for p in test_panels],
        train_num_options=len(train_options),
        reference_rough_params=dict(train_fit.best_params),
        train_rmse=float(train_fit.rough_rmse),
        validate_mean_rmse=float(np.mean([r.rmse for r in validate_eval])),
        test_mean_rmse=float(np.mean([r.rmse for r in test_eval])),
        validate_evaluations=validate_eval,
        test_evaluations=test_eval,
        parameter_drift=drift_summary,
        date_calibrations=date_calibrations,
    )


def historical_backtest_to_dict(result: HistoricalRoughBacktestResult) -> Dict[str, object]:
    """Serialize historical backtest result."""

    return {
        "train_dates": list(result.train_dates),
        "validate_dates": list(result.validate_dates),
        "test_dates": list(result.test_dates),
        "train_num_options": result.train_num_options,
        "reference_rough_params": dict(result.reference_rough_params),
        "train_rmse": result.train_rmse,
        "validate_mean_rmse": result.validate_mean_rmse,
        "test_mean_rmse": result.test_mean_rmse,
        "validate_evaluations": [asdict(x) for x in result.validate_evaluations],
        "test_evaluations": [asdict(x) for x in result.test_evaluations],
        "parameter_drift": asdict(result.parameter_drift),
        "date_calibrations": [dict(row) for row in result.date_calibrations],
    }

