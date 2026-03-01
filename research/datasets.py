"""Dataset loaders/builders for multi-year option panel studies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from calibration import MarketOption

from .historical_backtest import DatedOptionPanel, generate_synthetic_market_option_timeseries
from .market_data import load_snapshot_series
from .real_data_calibration import OptionChainQuote, quotes_to_market_options


@dataclass
class PanelDataset:
    """Container for dated option panels and metadata."""

    source: str
    symbol: str
    panels: List[DatedOptionPanel]
    start_date: str
    end_date: str
    num_panels: int


def _quotes_to_dated_panel(
    quotes: Sequence[OptionChainQuote],
    *,
    quote_date: str,
    true_params: Optional[Dict[str, float]] = None,
) -> DatedOptionPanel:
    options: List[MarketOption] = quotes_to_market_options(quotes)
    return DatedOptionPanel(
        quote_date=quote_date,
        options=options,
        true_params=true_params or {},
    )


def load_snapshot_panels(snapshot_paths: Sequence[str | Path], *, symbol: str) -> PanelDataset:
    """Build dated panel dataset from persisted snapshot JSON files."""

    snaps = load_snapshot_series(snapshot_paths)
    panels = [_quotes_to_dated_panel(s.quotes, quote_date=s.quote_date) for s in snaps]
    if not panels:
        raise ValueError("No snapshot panels found")
    ordered = sorted(panels, key=lambda p: p.quote_date)
    return PanelDataset(
        source="snapshot_series",
        symbol=symbol.upper(),
        panels=ordered,
        start_date=ordered[0].quote_date,
        end_date=ordered[-1].quote_date,
        num_panels=len(ordered),
    )


def build_multi_year_synthetic_dataset(
    *,
    symbol: str = "SYNTH",
    start_date: str = "2020-01-03",
    years: int = 4,
    step_days: int = 7,
    spot: float = 100.0,
    rate: float = 0.03,
    seed: int = 42,
) -> PanelDataset:
    """Generate a multi-year synthetic dataset for doctoral walk-forward studies."""

    if years < 1:
        raise ValueError("years must be >= 1")
    num_dates = max(8, int((365 * years) / max(step_days, 1)))
    panels = generate_synthetic_market_option_timeseries(
        start_date=start_date,
        num_dates=num_dates,
        step_days=step_days,
        spot=spot,
        rate=rate,
        seed=seed,
    )
    ordered = sorted(panels, key=lambda p: p.quote_date)
    return PanelDataset(
        source="synthetic_multi_year",
        symbol=symbol.upper(),
        panels=ordered,
        start_date=ordered[0].quote_date,
        end_date=ordered[-1].quote_date,
        num_panels=len(ordered),
    )


def resample_dataset(
    dataset: PanelDataset,
    *,
    every_n: int = 1,
    max_panels: Optional[int] = None,
) -> PanelDataset:
    """Downsample dataset panels for quick experimentation."""

    n = int(max(1, every_n))
    panels = dataset.panels[::n]
    if max_panels is not None:
        panels = panels[: int(max_panels)]
    if not panels:
        raise ValueError("Resampling produced zero panels")
    return PanelDataset(
        source=dataset.source,
        symbol=dataset.symbol,
        panels=panels,
        start_date=panels[0].quote_date,
        end_date=panels[-1].quote_date,
        num_panels=len(panels),
    )


def panel_dataset_to_dict(dataset: PanelDataset) -> Dict[str, object]:
    """Serialize dataset metadata and lightweight panel headers."""

    return {
        "source": dataset.source,
        "symbol": dataset.symbol,
        "start_date": dataset.start_date,
        "end_date": dataset.end_date,
        "num_panels": dataset.num_panels,
        "panels": [
            {
                "quote_date": p.quote_date,
                "num_options": len(p.options),
            }
            for p in dataset.panels
        ],
    }
