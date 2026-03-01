"""Cross-sectional multi-asset rough-Heston research diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .historical_backtest import generate_synthetic_market_option_timeseries, run_historical_rough_heston_backtest


@dataclass
class CrossSectionalAssetResult:
    """Per-asset historical rough-Heston diagnostics."""

    symbol: str
    spot: float
    validate_mean_rmse: float
    test_mean_rmse: float
    train_rmse: float
    drift_mean_l2: float
    drift_max_l2: float
    train_dates: int
    validate_dates: int
    test_dates: int
    reference_rough_params: Dict[str, float]


@dataclass
class CrossSectionalStudyResult:
    """Cross-sectional summary for a symbol universe."""

    symbols: List[str]
    asset_results: List[CrossSectionalAssetResult]
    mean_test_rmse: float
    std_test_rmse: float
    mean_validate_rmse: float
    best_symbol_by_test_rmse: str
    worst_symbol_by_test_rmse: str
    ranking_by_test_rmse: List[str]


def _default_symbols() -> List[str]:
    return ["AAPL", "MSFT", "SPY", "QQQ", "TSLA"]


def run_cross_sectional_rough_heston_study(
    *,
    symbols: Optional[Sequence[str]] = None,
    start_date: str = "2024-01-05",
    num_dates: int = 10,
    step_days: int = 7,
    spot: float = 100.0,
    rate: float = 0.03,
    use_iv: bool = False,
    train_fraction: float = 0.6,
    validate_fraction: float = 0.2,
    max_iter: int = 120,
    num_paths: int = 450,
    num_steps: int = 24,
    seed: int = 42,
) -> CrossSectionalStudyResult:
    """Run historical rough-Heston evaluation over multiple synthetic symbols."""

    universe = [s.upper() for s in (symbols if symbols is not None else _default_symbols())]
    if len(universe) < 2:
        raise ValueError("Need at least two symbols for cross-sectional diagnostics")

    asset_rows: List[CrossSectionalAssetResult] = []
    for i, symbol in enumerate(universe):
        # Asset-specific spot level and seed create distinct but comparable panels.
        spot_i = float(max(30.0, spot * (1.0 + 0.04 * (i - (len(universe) - 1) / 2.0))))
        panels = generate_synthetic_market_option_timeseries(
            start_date=start_date,
            num_dates=num_dates,
            step_days=step_days,
            spot=spot_i,
            rate=rate,
            seed=seed + 31 * i,
        )
        hist = run_historical_rough_heston_backtest(
            panels,
            spot=spot_i,
            rate=rate,
            use_iv=use_iv,
            train_fraction=train_fraction,
            validate_fraction=validate_fraction,
            max_iter=max_iter,
            num_paths=num_paths,
            num_steps=num_steps,
            seed=seed + 131 * i,
        )
        asset_rows.append(
            CrossSectionalAssetResult(
                symbol=symbol,
                spot=spot_i,
                validate_mean_rmse=float(hist.validate_mean_rmse),
                test_mean_rmse=float(hist.test_mean_rmse),
                train_rmse=float(hist.train_rmse),
                drift_mean_l2=float(hist.parameter_drift.mean_l2_drift),
                drift_max_l2=float(hist.parameter_drift.max_l2_drift),
                train_dates=len(hist.train_dates),
                validate_dates=len(hist.validate_dates),
                test_dates=len(hist.test_dates),
                reference_rough_params=dict(hist.reference_rough_params),
            )
        )

    test_rmse = np.asarray([x.test_mean_rmse for x in asset_rows], dtype=float)
    validate_rmse = np.asarray([x.validate_mean_rmse for x in asset_rows], dtype=float)
    ranking = sorted(asset_rows, key=lambda x: x.test_mean_rmse)

    return CrossSectionalStudyResult(
        symbols=universe,
        asset_results=asset_rows,
        mean_test_rmse=float(np.mean(test_rmse)),
        std_test_rmse=float(np.std(test_rmse, ddof=1)) if test_rmse.size > 1 else 0.0,
        mean_validate_rmse=float(np.mean(validate_rmse)),
        best_symbol_by_test_rmse=ranking[0].symbol,
        worst_symbol_by_test_rmse=ranking[-1].symbol,
        ranking_by_test_rmse=[x.symbol for x in ranking],
    )


def cross_sectional_study_to_dict(result: CrossSectionalStudyResult) -> Dict[str, object]:
    """Serialize cross-sectional result payload."""

    return {
        "symbols": list(result.symbols),
        "asset_results": [asdict(x) for x in result.asset_results],
        "mean_test_rmse": result.mean_test_rmse,
        "std_test_rmse": result.std_test_rmse,
        "mean_validate_rmse": result.mean_validate_rmse,
        "best_symbol_by_test_rmse": result.best_symbol_by_test_rmse,
        "worst_symbol_by_test_rmse": result.worst_symbol_by_test_rmse,
        "ranking_by_test_rmse": list(result.ranking_by_test_rmse),
    }
