"""Real option-chain ingestion, no-arbitrage checks, and train/test calibration evaluation."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from calibration import (
    CalibrationResult,
    MarketOption,
    calibrate_heston,
    heston_call_price,
    implied_volatility,
)


@dataclass
class OptionChainQuote:
    """One option quote observation from chain data."""

    strike: float
    maturity: float
    option_type: str
    bid: float
    ask: float
    iv: Optional[float] = None
    quote_date: Optional[str] = None
    expiry: Optional[str] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    last_trade_timestamp: Optional[int] = None

    @property
    def mid_price(self) -> float:
        return 0.5 * (self.bid + self.ask)


@dataclass
class NoArbitrageReport:
    """Summary of no-arbitrage quality checks."""

    total_quotes: int
    valid_quotes: int
    removed_invalid_quotes: int
    removed_bound_violations: int
    removed_monotonic_violations: int


@dataclass
class MarketQualityReport:
    """Summary of market microstructure quality filtering."""

    total_quotes: int
    valid_quotes: int
    removed_wide_spread: int
    removed_low_liquidity: int
    removed_stale_quotes: int


@dataclass
class CalibrationOutOfSampleResult:
    """Train/test calibration diagnostics for quant research reporting."""

    calibration: CalibrationResult
    train_rmse: float
    test_rmse: float
    train_size: int
    test_size: int
    no_arb_report: NoArbitrageReport


def _parse_float(row: Dict[str, str], key: str, default: Optional[float] = None) -> Optional[float]:
    val = row.get(key)
    if val is None or str(val).strip() == "":
        return default
    return float(val)


def _infer_maturity(row: Dict[str, str]) -> float:
    for key in ("maturity", "time_to_maturity", "ttm"):
        if row.get(key):
            return float(row[key])

    quote_date = row.get("quote_date") or row.get("date")
    expiry = row.get("expiry") or row.get("expiration")
    if quote_date and expiry:
        d0 = datetime.fromisoformat(quote_date).date()
        d1 = datetime.fromisoformat(expiry).date()
        return max((d1 - d0).days / 365.0, 1e-6)

    raise ValueError("Missing maturity information: expected maturity/ttm or quote_date+expiry")


def load_option_chain_csv(path: str | Path) -> List[OptionChainQuote]:
    """Load option chain CSV into normalized quote objects."""

    quotes: List[OptionChainQuote] = []
    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            option_type = (row.get("option_type") or row.get("type") or "call").strip().lower()
            option_type = "call" if option_type.startswith("c") else "put"

            quotes.append(
                OptionChainQuote(
                    strike=float(row["strike"]),
                    maturity=_infer_maturity(row),
                    option_type=option_type,
                    bid=float(row["bid"]),
                    ask=float(row["ask"]),
                    iv=_parse_float(row, "iv", default=_parse_float(row, "implied_volatility", default=None)),
                    quote_date=row.get("quote_date") or row.get("date"),
                    expiry=row.get("expiry") or row.get("expiration"),
                    volume=int(row["volume"]) if row.get("volume") not in (None, "") else None,
                    open_interest=int(row["open_interest"]) if row.get("open_interest") not in (None, "") else None,
                    bid_size=int(row["bid_size"]) if row.get("bid_size") not in (None, "") else None,
                    ask_size=int(row["ask_size"]) if row.get("ask_size") not in (None, "") else None,
                    last_trade_timestamp=int(row["last_trade_timestamp"])
                    if row.get("last_trade_timestamp") not in (None, "")
                    else None,
                )
            )

    return quotes


def write_option_chain_csv(path: str | Path, quotes: Sequence[OptionChainQuote]) -> None:
    """Write option quotes to CSV for reproducible experiments."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "strike",
                "maturity",
                "option_type",
                "bid",
                "ask",
                "iv",
                "quote_date",
                "expiry",
                "volume",
                "open_interest",
                "bid_size",
                "ask_size",
                "last_trade_timestamp",
            ]
        )
        for q in quotes:
            writer.writerow(
                [
                    q.strike,
                    q.maturity,
                    q.option_type,
                    q.bid,
                    q.ask,
                    "" if q.iv is None else q.iv,
                    "" if q.quote_date is None else q.quote_date,
                    "" if q.expiry is None else q.expiry,
                    "" if q.volume is None else q.volume,
                    "" if q.open_interest is None else q.open_interest,
                    "" if q.bid_size is None else q.bid_size,
                    "" if q.ask_size is None else q.ask_size,
                    "" if q.last_trade_timestamp is None else q.last_trade_timestamp,
                ]
            )


def generate_synthetic_option_chain(
    *,
    spot: float = 100.0,
    rate: float = 0.03,
    true_params: Optional[Dict[str, float]] = None,
    seed: int = 42,
) -> List[OptionChainQuote]:
    """Generate synthetic option chain resembling real calibration data."""

    if true_params is None:
        true_params = {"v0": 0.05, "kappa": 2.5, "theta": 0.04, "xi": 0.45, "rho": -0.65}

    rng = np.random.default_rng(seed)
    strikes = [80, 90, 100, 110, 120]
    maturities = [0.25, 0.5, 1.0]

    quotes: List[OptionChainQuote] = []
    for t in maturities:
        for k in strikes:
            price = heston_call_price(spot, k, rate, t, **true_params)
            noisy = max(price * (1.0 + rng.normal(0.0, 0.01)), 1e-8)
            spread = max(0.01, 0.02 * noisy)
            bid = max(noisy - 0.5 * spread, 1e-8)
            ask = noisy + 0.5 * spread
            iv = implied_volatility(noisy, spot, k, rate, t, option_type="call")
            quotes.append(
                OptionChainQuote(
                    strike=float(k),
                    maturity=float(t),
                    option_type="call",
                    bid=float(bid),
                    ask=float(ask),
                    iv=float(iv),
                )
            )

    return quotes


def _no_arb_price_bounds(
    q: OptionChainQuote,
    spot: float,
    rate: float,
) -> Tuple[float, float]:
    disc_k = q.strike * np.exp(-rate * q.maturity)
    if q.option_type == "call":
        lower = max(0.0, spot - disc_k)
        upper = spot
    else:
        lower = max(0.0, disc_k - spot)
        upper = disc_k
    return lower, upper


def filter_no_arbitrage_quotes(
    quotes: Iterable[OptionChainQuote],
    *,
    spot: float,
    rate: float,
    tol: float = 1e-8,
) -> Tuple[List[OptionChainQuote], NoArbitrageReport]:
    """Filter obvious no-arbitrage violations for robust calibration."""

    quotes = list(quotes)
    valid: List[OptionChainQuote] = []
    removed_invalid = 0
    removed_bounds = 0

    for q in quotes:
        if q.bid < 0 or q.ask <= 0 or q.ask < q.bid:
            removed_invalid += 1
            continue
        lower, upper = _no_arb_price_bounds(q, spot, rate)
        mid = q.mid_price
        if mid < lower - tol or mid > upper + tol:
            removed_bounds += 1
            continue
        valid.append(q)

    # Monotonicity filter by maturity and type.
    by_bucket: Dict[Tuple[float, str], List[OptionChainQuote]] = {}
    for q in valid:
        key = (q.maturity, q.option_type)
        by_bucket.setdefault(key, []).append(q)

    filtered: List[OptionChainQuote] = []
    removed_monotonic = 0
    for (maturity, option_type), bucket in by_bucket.items():
        del maturity  # only for grouping readability
        ordered = sorted(bucket, key=lambda x: x.strike)
        if option_type == "call":
            prev = float("inf")
            for q in ordered:
                mid = q.mid_price
                if mid <= prev + tol:
                    filtered.append(q)
                    prev = min(prev, mid)
                else:
                    removed_monotonic += 1
        else:
            prev = -float("inf")
            for q in ordered:
                mid = q.mid_price
                if mid >= prev - tol:
                    filtered.append(q)
                    prev = max(prev, mid)
                else:
                    removed_monotonic += 1

    report = NoArbitrageReport(
        total_quotes=len(quotes),
        valid_quotes=len(filtered),
        removed_invalid_quotes=removed_invalid,
        removed_bound_violations=removed_bounds,
        removed_monotonic_violations=removed_monotonic,
    )
    return filtered, report


def filter_market_quality_quotes(
    quotes: Iterable[OptionChainQuote],
    *,
    as_of_date: Optional[str] = None,
    max_relative_spread: float = 0.40,
    min_open_interest: int = 1,
    min_volume: int = 0,
    max_staleness_days: int = 7,
) -> Tuple[List[OptionChainQuote], MarketQualityReport]:
    """Filter quotes by spread/liquidity/staleness constraints."""

    rows = list(quotes)
    out: List[OptionChainQuote] = []
    removed_spread = 0
    removed_liquidity = 0
    removed_stale = 0

    ref_date = datetime.fromisoformat(as_of_date).date() if as_of_date else None

    for q in rows:
        mid = max(q.mid_price, 1e-12)
        rel_spread = (q.ask - q.bid) / mid if mid > 0 else float("inf")
        if rel_spread > max_relative_spread:
            removed_spread += 1
            continue

        oi = q.open_interest if q.open_interest is not None else min_open_interest
        vol = q.volume if q.volume is not None else min_volume
        if oi < min_open_interest or vol < min_volume:
            removed_liquidity += 1
            continue

        if ref_date is not None and q.quote_date is not None:
            try:
                qd = datetime.fromisoformat(q.quote_date).date()
            except Exception:
                qd = None
            if qd is not None and (ref_date - qd).days > max_staleness_days:
                removed_stale += 1
                continue

        out.append(q)

    return out, MarketQualityReport(
        total_quotes=len(rows),
        valid_quotes=len(out),
        removed_wide_spread=removed_spread,
        removed_low_liquidity=removed_liquidity,
        removed_stale_quotes=removed_stale,
    )


def quotes_to_market_options(quotes: Iterable[OptionChainQuote]) -> List[MarketOption]:
    """Convert quotes to calibration MarketOption objects."""

    return [
        MarketOption(
            strike=q.strike,
            maturity=q.maturity,
            market_price=q.mid_price,
            market_iv=q.iv,
            option_type=q.option_type,
            bid=q.bid,
            ask=q.ask,
        )
        for q in quotes
    ]


def split_train_test_options(
    options: Sequence[MarketOption],
    *,
    train_fraction: float = 0.7,
    seed: int = 42,
) -> Tuple[List[MarketOption], List[MarketOption]]:
    """Stratified split by maturity for out-of-sample evaluation."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")

    by_maturity: Dict[float, List[MarketOption]] = {}
    for opt in options:
        by_maturity.setdefault(opt.maturity, []).append(opt)

    rng = np.random.default_rng(seed)
    train: List[MarketOption] = []
    test: List[MarketOption] = []

    for maturity, group in by_maturity.items():
        del maturity
        idx = np.arange(len(group))
        rng.shuffle(idx)
        cut = max(1, min(len(group) - 1, int(np.floor(train_fraction * len(group)))))
        train_idx = set(idx[:cut])
        for i, opt in enumerate(group):
            if i in train_idx:
                train.append(opt)
            else:
                test.append(opt)

    return train, test


def _heston_rmse(
    options: Sequence[MarketOption],
    *,
    spot: float,
    rate: float,
    params: Dict[str, float],
    use_iv: bool,
) -> float:
    errs = []
    for opt in options:
        call_price = heston_call_price(
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
            model_price = call_price - spot + opt.strike * np.exp(-rate * opt.maturity)
        else:
            model_price = call_price

        if use_iv and opt.market_iv is not None:
            model_value = implied_volatility(model_price, spot, opt.strike, rate, opt.maturity, opt.option_type)
            target = opt.market_iv
        else:
            model_value = model_price
            target = opt.market_price if opt.market_price is not None else opt.mid_price

        errs.append((model_value - target) ** 2)

    return float(np.sqrt(np.mean(errs))) if errs else float("nan")


def calibrate_heston_train_test_from_quotes(
    quotes: Sequence[OptionChainQuote],
    *,
    spot: float,
    rate: float,
    train_fraction: float = 0.7,
    use_iv: bool = True,
    max_iter: int = 300,
    seed: int = 42,
) -> CalibrationOutOfSampleResult:
    """Run no-arbitrage filtering, train/test split, and calibration diagnostics."""

    filtered, report = filter_no_arbitrage_quotes(quotes, spot=spot, rate=rate)
    options = quotes_to_market_options(filtered)
    train, test = split_train_test_options(options, train_fraction=train_fraction, seed=seed)

    calibration = calibrate_heston(train, spot=spot, rate=rate, use_iv=use_iv, max_iter=max_iter)
    train_rmse = _heston_rmse(train, spot=spot, rate=rate, params=calibration.parameters, use_iv=use_iv)
    test_rmse = _heston_rmse(test, spot=spot, rate=rate, params=calibration.parameters, use_iv=use_iv)

    return CalibrationOutOfSampleResult(
        calibration=calibration,
        train_rmse=train_rmse,
        test_rmse=test_rmse,
        train_size=len(train),
        test_size=len(test),
        no_arb_report=report,
    )


def train_test_result_to_dict(result: CalibrationOutOfSampleResult) -> Dict[str, object]:
    """Serialize train/test calibration result."""

    return {
        "calibration": {
            "success": result.calibration.success,
            "parameters": result.calibration.parameters,
            "objective_value": result.calibration.objective_value,
            "rmse": result.calibration.rmse,
            "num_iterations": result.calibration.num_iterations,
            "calibration_time": result.calibration.calibration_time,
            "message": result.calibration.message,
        },
        "train_rmse": result.train_rmse,
        "test_rmse": result.test_rmse,
        "train_size": result.train_size,
        "test_size": result.test_size,
        "no_arbitrage_report": asdict(result.no_arb_report),
    }
