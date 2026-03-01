"""Leakage-proof walk-forward protocols for option-surface forecasting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .forecasting import run_rolling_oos_forecast_study
from .historical_backtest import DatedOptionPanel


@dataclass
class WalkForwardWindow:
    """One leakage-proof train/validate/test date window."""

    train_dates: List[str]
    validate_dates: List[str]
    test_dates: List[str]


@dataclass
class WalkForwardWindowResult:
    """Forecast-loss summary for a single walk-forward window."""

    window_index: int
    train_start: str
    train_end: str
    validate_start: str
    validate_end: str
    test_start: str
    test_end: str
    mean_rmse_by_model: Dict[str, float]
    best_model: str
    num_test_transitions: int


@dataclass
class WalkForwardStudyResult:
    """Aggregate walk-forward result across all windows."""

    windows: List[WalkForwardWindowResult]
    aggregate_mean_rmse_by_model: Dict[str, float]
    aggregate_std_rmse_by_model: Dict[str, float]
    best_model_by_aggregate_rmse: str
    num_windows: int


def build_walkforward_windows(
    quote_dates: Sequence[str],
    *,
    train_size: int = 8,
    validate_size: int = 2,
    test_size: int = 2,
    step_size: int = 1,
) -> List[WalkForwardWindow]:
    """Create leakage-proof rolling windows over ordered quote dates."""

    dates = sorted(list(quote_dates))
    n = len(dates)
    if n < train_size + validate_size + test_size:
        raise ValueError("Insufficient dates for requested walk-forward window sizes")
    step = max(1, int(step_size))
    windows: List[WalkForwardWindow] = []
    start = 0
    while True:
        i0 = start
        i1 = i0 + train_size
        i2 = i1 + validate_size
        i3 = i2 + test_size
        if i3 > n:
            break
        windows.append(
            WalkForwardWindow(
                train_dates=dates[i0:i1],
                validate_dates=dates[i1:i2],
                test_dates=dates[i2:i3],
            )
        )
        start += step
    return windows


def run_leakage_free_walkforward(
    panels: Sequence[DatedOptionPanel],
    *,
    spot: float,
    rate: float,
    use_iv: bool = False,
    train_size: int = 8,
    validate_size: int = 2,
    test_size: int = 2,
    step_size: int = 1,
    max_windows: Optional[int] = None,
    heston_max_iter: int = 60,
    rough_max_iter: int = 45,
    rough_num_paths: int = 180,
    rough_num_steps: int = 12,
    seed: int = 42,
) -> WalkForwardStudyResult:
    """Run leakage-proof walk-forward forecasting summaries over dated panels."""

    ordered = sorted(list(panels), key=lambda p: p.quote_date)
    by_date = {p.quote_date: p for p in ordered}
    windows = build_walkforward_windows(
        [p.quote_date for p in ordered],
        train_size=train_size,
        validate_size=validate_size,
        test_size=test_size,
        step_size=step_size,
    )
    if max_windows is not None:
        windows = windows[: int(max_windows)]
    if not windows:
        raise ValueError("No walk-forward windows generated")

    rows: List[WalkForwardWindowResult] = []
    aggregate: Dict[str, List[float]] = {}

    for w_idx, win in enumerate(windows):
        block_dates = win.train_dates + win.validate_dates + win.test_dates
        block_panels = [by_date[d] for d in block_dates]
        oos = run_rolling_oos_forecast_study(
            block_panels,
            spot=spot,
            rate=rate,
            use_iv=use_iv,
            heston_max_iter=heston_max_iter,
            rough_max_iter=rough_max_iter,
            rough_num_paths=rough_num_paths,
            rough_num_steps=rough_num_steps,
            seed=seed + 1000 * w_idx,
        )

        test_set = set(win.test_dates)
        test_obs = [x for x in oos.observations if x.target_date in test_set]
        if not test_obs:
            raise RuntimeError("No test observations in walk-forward window")

        by_model: Dict[str, List[float]] = {}
        for obs in test_obs:
            by_model.setdefault(obs.model, []).append(float(obs.rmse))
        mean_by_model = {k: float(np.mean(v)) for k, v in by_model.items()}
        best_model = min(mean_by_model.keys(), key=lambda k: mean_by_model[k])

        for model, vals in by_model.items():
            aggregate.setdefault(model, []).extend(vals)

        rows.append(
            WalkForwardWindowResult(
                window_index=w_idx,
                train_start=win.train_dates[0],
                train_end=win.train_dates[-1],
                validate_start=win.validate_dates[0],
                validate_end=win.validate_dates[-1],
                test_start=win.test_dates[0],
                test_end=win.test_dates[-1],
                mean_rmse_by_model=mean_by_model,
                best_model=best_model,
                num_test_transitions=len(test_obs),
            )
        )

    agg_mean = {k: float(np.mean(v)) for k, v in aggregate.items()}
    agg_std = {k: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for k, v in aggregate.items()}
    best = min(agg_mean.keys(), key=lambda k: agg_mean[k])

    return WalkForwardStudyResult(
        windows=rows,
        aggregate_mean_rmse_by_model=agg_mean,
        aggregate_std_rmse_by_model=agg_std,
        best_model_by_aggregate_rmse=best,
        num_windows=len(rows),
    )


def walkforward_to_dict(result: WalkForwardStudyResult) -> Dict[str, object]:
    """Serialize walk-forward study result."""

    return {
        "windows": [asdict(x) for x in result.windows],
        "aggregate_mean_rmse_by_model": dict(result.aggregate_mean_rmse_by_model),
        "aggregate_std_rmse_by_model": dict(result.aggregate_std_rmse_by_model),
        "best_model_by_aggregate_rmse": result.best_model_by_aggregate_rmse,
        "num_windows": result.num_windows,
    }
