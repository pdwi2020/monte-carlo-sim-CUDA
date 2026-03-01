"""Free-tier market data ingestion for option-chain research workflows."""

from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .real_data_calibration import OptionChainQuote


@dataclass
class MarketDataSnapshot:
    """One fetched option-chain snapshot."""

    symbol: str
    source: str
    quote_date: str
    expiration: str
    underlying_price: Optional[float]
    quotes: List[OptionChainQuote]


def _to_iso_date_from_unix(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).date().isoformat()


def _fetch_json(url: str, timeout_seconds: int = 12) -> Dict[str, object]:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_call_price(s: float, k: float, r: float, sigma: float, t: float) -> float:
    if t <= 0:
        return max(s - k, 0.0)
    sigma = max(sigma, 1e-8)
    st = sigma * math.sqrt(t)
    d1 = (math.log(max(s, 1e-12) / max(k, 1e-12)) + (r + 0.5 * sigma * sigma) * t) / st
    d2 = d1 - st
    return s * _normal_cdf(d1) - k * math.exp(-r * t) * _normal_cdf(d2)


def _polygon_prev_close(symbol: str, api_key: str, timeout_seconds: int) -> Optional[float]:
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{urllib.parse.quote(symbol)}/prev"
        f"?adjusted=true&apiKey={urllib.parse.quote(api_key)}"
    )
    try:
        data = _fetch_json(url, timeout_seconds=timeout_seconds)
        results = data.get("results") or []
        if results:
            return float(results[0].get("c"))
    except Exception:
        return None
    return None


def _polygon_realized_vol(symbol: str, api_key: str, timeout_seconds: int) -> float:
    """Estimate annualized vol from recent daily closes; fallback to 25%."""

    try:
        now = datetime.now(tz=timezone.utc).date()
        start = now.replace(year=max(now.year - 1, 1971))
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{urllib.parse.quote(symbol)}/range/1/day/"
            f"{start.isoformat()}/{now.isoformat()}?adjusted=true&sort=desc&limit=90&apiKey={urllib.parse.quote(api_key)}"
        )
        data = _fetch_json(url, timeout_seconds=timeout_seconds)
        rows = data.get("results") or []
        closes = [float(r["c"]) for r in rows if r.get("c") is not None]
        if len(closes) < 10:
            return 0.25
        rets = []
        for a, b in zip(closes[:-1], closes[1:]):
            if a > 0 and b > 0:
                rets.append(math.log(a / b))
        if len(rets) < 5:
            return 0.25
        mean_r = sum(rets) / len(rets)
        var = sum((x - mean_r) ** 2 for x in rets) / max(len(rets) - 1, 1)
        return float(max(0.10, min(math.sqrt(var * 252.0), 1.2)))
    except Exception:
        return 0.25


def _fmp_quote_price(symbol: str, api_key: str, timeout_seconds: int) -> Optional[float]:
    url = "https://financialmodelingprep.com/stable/quote?" + urllib.parse.urlencode(
        {"symbol": symbol.upper(), "apikey": api_key}
    )
    try:
        data = _fetch_json(url, timeout_seconds=timeout_seconds)
        rows = _extract_rows(data)
        if rows:
            return _coerce_float(rows[0].get("price") or rows[0].get("close"))
    except Exception:
        return None
    return None


def _fmp_realized_vol(symbol: str, api_key: str, timeout_seconds: int) -> float:
    """Estimate annualized vol from FMP daily prices; fallback to 25%."""

    url = "https://financialmodelingprep.com/stable/historical-price-eod/light?" + urllib.parse.urlencode(
        {"symbol": symbol.upper(), "apikey": api_key}
    )
    try:
        data = _fetch_json(url, timeout_seconds=timeout_seconds)
        rows = _extract_rows(data)
        closes: List[float] = []
        for row in rows[:120]:
            px = _coerce_float(row.get("price") or row.get("close"))
            if px is not None and px > 0:
                closes.append(float(px))
        if len(closes) < 10:
            return 0.25
        rets = []
        for a, b in zip(closes[:-1], closes[1:]):
            if a > 0 and b > 0:
                rets.append(math.log(a / b))
        if len(rets) < 5:
            return 0.25
        mean_r = sum(rets) / len(rets)
        var = sum((x - mean_r) ** 2 for x in rets) / max(len(rets) - 1, 1)
        return float(max(0.10, min(math.sqrt(var * 252.0), 1.2)))
    except Exception:
        return 0.25


def _build_proxy_quotes(
    *,
    quote_date: str,
    expiration: Optional[str],
    spot: float,
    sigma: float,
    rate: float = 0.03,
) -> List[OptionChainQuote]:
    """Construct a synthetic but internally consistent option chain."""

    qd = date.fromisoformat(quote_date)
    if expiration is not None:
        expiries = [expiration]
    else:
        # Explicit integer-day expiries avoid month-end rollover edge cases.
        expiries = [date.fromordinal(qd.toordinal() + d).isoformat() for d in (30, 60, 90)]

    strike_multipliers = [0.75, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.25]
    quotes: List[OptionChainQuote] = []

    for expiry in expiries:
        ed = date.fromisoformat(expiry)
        ttm = max((ed - qd).days / 365.0, 1e-6)
        for mny in strike_multipliers:
            strike = float(max(1e-6, mny * spot))
            call = _bs_call_price(spot, strike, rate, sigma, ttm)
            put = call - spot + strike * math.exp(-rate * ttm)
            for option_type, mid in (("call", call), ("put", put)):
                mid = max(float(mid), 1e-8)
                rel_spread = 0.04 + 0.06 * abs(mny - 1.0)
                spread = max(0.01, rel_spread * mid)
                bid = max(mid - 0.5 * spread, 1e-8)
                ask = mid + 0.5 * spread
                quotes.append(
                    OptionChainQuote(
                        strike=strike,
                        maturity=float(ttm),
                        option_type=option_type,
                        bid=float(bid),
                        ask=float(ask),
                        iv=float(sigma),
                        quote_date=quote_date,
                        expiry=expiry,
                    )
                )
    return quotes


def _coerce_float(value: object) -> Optional[float]:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _coerce_int(value: object) -> Optional[int]:
    if value in (None, "", "None"):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _extract_rows(payload: object) -> List[Dict[str, object]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("options", "results", "data", "optionChain", "chain"):
        obj = payload.get(key)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            nested = obj.get("result")
            if isinstance(nested, list):
                return [x for x in nested if isinstance(x, dict)]
    return []


def fetch_fmp_option_chain(
    symbol: str,
    *,
    expiration: Optional[str] = None,
    api_key: str,
    timeout_seconds: int = 12,
) -> MarketDataSnapshot:
    """Fetch option-chain snapshot from FMP free-tier style endpoints.

    FMP endpoints and payload wrappers vary across API versions/plans. This
    adapter tries multiple documented URL shapes and normalizes field names.
    """

    sym = symbol.upper()
    qdate = datetime.now(tz=timezone.utc).date().isoformat()
    rows: List[Dict[str, object]] = []

    endpoint_urls = [
        "https://financialmodelingprep.com/stable/options-chain?"
        + urllib.parse.urlencode(
            {
                "symbol": sym,
                "apikey": api_key,
                **({"expiration": expiration} if expiration is not None else {}),
            }
        ),
        "https://financialmodelingprep.com/api/v3/options-chain/"
        + urllib.parse.quote(sym)
        + "?"
        + urllib.parse.urlencode(
            {
                "apikey": api_key,
                **({"expiration": expiration} if expiration is not None else {}),
            }
        ),
        "https://financialmodelingprep.com/api/v4/options-chain?"
        + urllib.parse.urlencode(
            {
                "symbol": sym,
                "apikey": api_key,
                **({"expiration": expiration} if expiration is not None else {}),
            }
        ),
    ]

    last_error = None
    for url in endpoint_urls:
        try:
            payload = _fetch_json(url, timeout_seconds=timeout_seconds)
            rows = _extract_rows(payload)
            if rows:
                break
        except Exception as exc:  # pragma: no cover - network error branch
            last_error = exc
            continue

    if not rows:
        # Many free plans do not include direct option-chain entitlement.
        # Fallback to quote + historical-vol proxy chain so the research pipeline
        # can still run end-to-end on free-tier credentials.
        spot = _fmp_quote_price(sym, api_key, timeout_seconds) or 100.0
        sigma = _fmp_realized_vol(sym, api_key, timeout_seconds)
        proxy_quotes = _build_proxy_quotes(
            quote_date=qdate,
            expiration=expiration,
            spot=float(spot),
            sigma=float(sigma),
            rate=0.03,
        )
        return MarketDataSnapshot(
            symbol=sym,
            source="fmp_free_quote_proxy",
            quote_date=qdate,
            expiration=expiration or proxy_quotes[0].expiry or qdate,
            underlying_price=float(spot),
            quotes=proxy_quotes,
        )

    quotes: List[OptionChainQuote] = []
    expiries: List[str] = []
    for row in rows:
        strike = _coerce_float(
            row.get("strike")
            or row.get("strikePrice")
            or row.get("strike_price")
            or row.get("exercisePrice")
        )
        expiry = row.get("expiration") or row.get("expirationDate") or row.get("expiry") or row.get("expiration_date")
        expiry = str(expiry) if expiry not in (None, "") else None
        if strike is None or expiry is None:
            continue
        if expiration is not None and expiry != expiration:
            continue

        quote_date = (
            row.get("quoteDate")
            or row.get("quotedate")
            or row.get("date")
            or row.get("tradeDate")
            or row.get("lastTradeDate")
            or qdate
        )
        quote_date = str(quote_date)[:10]
        try:
            qd = date.fromisoformat(quote_date)
        except Exception:
            qd = date.fromisoformat(qdate)
            quote_date = qdate
        try:
            ed = date.fromisoformat(expiry)
        except Exception:
            continue
        maturity = max((ed - qd).days / 365.0, 1e-6)

        raw_type = str(row.get("optionType") or row.get("type") or row.get("contractType") or "call").lower()
        option_type = "call" if raw_type.startswith("c") else "put"

        bid = _coerce_float(row.get("bid"))
        ask = _coerce_float(row.get("ask"))
        if bid is None or ask is None or ask < bid:
            mid = _coerce_float(row.get("lastPrice") or row.get("mark") or row.get("price"))
            if mid is None:
                continue
            spread = max(0.01, 0.05 * float(mid))
            bid = max(float(mid) - 0.5 * spread, 1e-8)
            ask = float(mid) + 0.5 * spread

        quotes.append(
            OptionChainQuote(
                strike=float(strike),
                maturity=float(maturity),
                option_type=option_type,
                bid=float(bid),
                ask=float(ask),
                iv=_coerce_float(row.get("impliedVolatility") or row.get("implied_volatility") or row.get("iv")),
                quote_date=quote_date,
                expiry=expiry,
                volume=_coerce_int(row.get("volume")),
                open_interest=_coerce_int(row.get("openInterest") or row.get("open_interest")),
                bid_size=_coerce_int(row.get("bidSize") or row.get("bid_size")),
                ask_size=_coerce_int(row.get("askSize") or row.get("ask_size")),
                last_trade_timestamp=_coerce_int(row.get("lastTradeDate") or row.get("last_trade_timestamp")),
            )
        )
        expiries.append(expiry)

    if not quotes:
        spot = _fmp_quote_price(sym, api_key, timeout_seconds) or 100.0
        sigma = _fmp_realized_vol(sym, api_key, timeout_seconds)
        proxy_quotes = _build_proxy_quotes(
            quote_date=qdate,
            expiration=expiration,
            spot=float(spot),
            sigma=float(sigma),
            rate=0.03,
        )
        return MarketDataSnapshot(
            symbol=sym,
            source="fmp_free_quote_proxy",
            quote_date=qdate,
            expiration=expiration or proxy_quotes[0].expiry or qdate,
            underlying_price=float(spot),
            quotes=proxy_quotes,
        )

    spot = None
    for row in rows:
        spot = _coerce_float(row.get("underlyingPrice") or row.get("underlying_price") or row.get("spotPrice"))
        if spot is not None:
            break

    return MarketDataSnapshot(
        symbol=sym,
        source="fmp_free",
        quote_date=qdate,
        expiration=expiration or min(expiries),
        underlying_price=spot,
        quotes=quotes,
    )


def list_yahoo_expirations(symbol: str, timeout_seconds: int = 12) -> List[str]:
    """List available expiration dates from Yahoo free endpoint."""

    url = f"https://query2.finance.yahoo.com/v7/finance/options/{urllib.parse.quote(symbol)}"
    data = _fetch_json(url, timeout_seconds=timeout_seconds)
    result = ((data.get("optionChain") or {}).get("result") or [{}])[0]
    expirations = result.get("expirationDates") or []
    return [_to_iso_date_from_unix(int(x)) for x in expirations]


def fetch_yahoo_option_chain(
    symbol: str,
    *,
    expiration: Optional[str] = None,
    timeout_seconds: int = 12,
) -> MarketDataSnapshot:
    """Fetch option-chain snapshot from Yahoo Finance (free/no-key)."""

    base = f"https://query2.finance.yahoo.com/v7/finance/options/{urllib.parse.quote(symbol)}"
    if expiration is not None:
        exp_date = date.fromisoformat(expiration)
        exp_ts = int(datetime(exp_date.year, exp_date.month, exp_date.day, tzinfo=timezone.utc).timestamp())
        url = f"{base}?date={exp_ts}"
    else:
        url = base

    data = _fetch_json(url, timeout_seconds=timeout_seconds)
    result = ((data.get("optionChain") or {}).get("result") or [{}])[0]
    quotes_block = (result.get("options") or [{}])[0]

    underlying = result.get("quote") or {}
    underlying_price = underlying.get("regularMarketPrice")
    quote_date = _to_iso_date_from_unix(int(underlying.get("regularMarketTime", int(datetime.now(tz=timezone.utc).timestamp()))))
    expiration_ts = quotes_block.get("expirationDate")
    expiration_iso = _to_iso_date_from_unix(int(expiration_ts)) if expiration_ts else (expiration or quote_date)

    out: List[OptionChainQuote] = []
    for field, option_type in (("calls", "call"), ("puts", "put")):
        for row in quotes_block.get(field, []) or []:
            bid = float(row.get("bid", 0.0) or 0.0)
            ask = float(row.get("ask", 0.0) or 0.0)
            strike = float(row["strike"])
            maturity = max((date.fromisoformat(expiration_iso) - date.fromisoformat(quote_date)).days / 365.0, 1e-6)
            out.append(
                OptionChainQuote(
                    strike=strike,
                    maturity=float(maturity),
                    option_type=option_type,
                    bid=bid,
                    ask=ask,
                    iv=float(row["impliedVolatility"]) if row.get("impliedVolatility") is not None else None,
                    quote_date=quote_date,
                    expiry=expiration_iso,
                    volume=int(row["volume"]) if row.get("volume") is not None else None,
                    open_interest=int(row["openInterest"]) if row.get("openInterest") is not None else None,
                    last_trade_timestamp=int(row["lastTradeDate"]) if row.get("lastTradeDate") is not None else None,
                )
            )

    return MarketDataSnapshot(
        symbol=symbol.upper(),
        source="yahoo_free",
        quote_date=quote_date,
        expiration=expiration_iso,
        underlying_price=float(underlying_price) if underlying_price is not None else None,
        quotes=out,
    )


def fetch_option_chain_free(
    symbol: str,
    *,
    provider: str = "yahoo_free",
    expiration: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout_seconds: int = 12,
) -> MarketDataSnapshot:
    """Unified free-tier fetch entrypoint.

    `provider='yahoo_free'` requires no API key and is the default.
    For key-based providers, pass `api_key` once you share it.
    """

    if provider == "yahoo_free":
        return fetch_yahoo_option_chain(symbol, expiration=expiration, timeout_seconds=timeout_seconds)

    if provider == "polygon_free":
        if not api_key:
            raise ValueError("polygon_free provider requires api_key")
        qdate = datetime.now(tz=timezone.utc).date().isoformat()

        # First try true option snapshots (may be plan-restricted).
        try:
            base = f"https://api.polygon.io/v3/snapshot/options/{urllib.parse.quote(symbol)}"
            params = {"limit": "250", "apiKey": api_key}
            if expiration is not None:
                params["expiration_date"] = expiration
            url = f"{base}?{urllib.parse.urlencode(params)}"
            data = _fetch_json(url, timeout_seconds=timeout_seconds)
            results = data.get("results") or []
            quotes: List[OptionChainQuote] = []
            for row in results:
                details = row.get("details") or {}
                quote = row.get("last_quote") or {}
                strike = details.get("strike_price")
                expiry = details.get("expiration_date")
                ctype = str(details.get("contract_type", "call")).lower()
                option_type = "call" if ctype.startswith("c") else "put"
                if strike is None or expiry is None:
                    continue
                maturity = max((date.fromisoformat(expiry) - date.fromisoformat(qdate)).days / 365.0, 1e-6)
                bid = float(quote.get("bid", 0.0) or 0.0)
                ask = float(quote.get("ask", 0.0) or 0.0)
                quotes.append(
                    OptionChainQuote(
                        strike=float(strike),
                        maturity=float(maturity),
                        option_type=option_type,
                        bid=bid,
                        ask=ask,
                        iv=float(row["implied_volatility"]) if row.get("implied_volatility") is not None else None,
                        quote_date=qdate,
                        expiry=expiry,
                        volume=int((row.get("day") or {}).get("volume"))
                        if (row.get("day") or {}).get("volume") is not None
                        else None,
                        open_interest=int(row["open_interest"]) if row.get("open_interest") is not None else None,
                        bid_size=int(quote["bid_size"]) if quote.get("bid_size") is not None else None,
                        ask_size=int(quote["ask_size"]) if quote.get("ask_size") is not None else None,
                    )
                )
            return MarketDataSnapshot(
                symbol=symbol.upper(),
                source="polygon_free",
                quote_date=qdate,
                expiration=expiration or (quotes[0].expiry if quotes else qdate),
                underlying_price=None,
                quotes=quotes,
            )
        except Exception:
            # Fallback for free-tier plans without options snapshot permission:
            # use contracts reference + stock-derived proxy prices.
            pass

        contracts_url = "https://api.polygon.io/v3/reference/options/contracts?" + urllib.parse.urlencode(
            {
                "underlying_ticker": symbol.upper(),
                "limit": 300,
                "as_of": qdate,
                "apiKey": api_key,
            }
        )
        data = _fetch_json(contracts_url, timeout_seconds=timeout_seconds)
        contracts = data.get("results") or []

        spot = _polygon_prev_close(symbol.upper(), api_key, timeout_seconds) or 100.0
        sigma = _polygon_realized_vol(symbol.upper(), api_key, timeout_seconds)
        rate = 0.03

        quotes: List[OptionChainQuote] = []
        for row in contracts:
            strike = row.get("strike_price")
            expiry = row.get("expiration_date")
            ctype = str(row.get("contract_type", "call")).lower()
            option_type = "call" if ctype.startswith("c") else "put"
            if strike is None or expiry is None:
                continue

            ttm = max((date.fromisoformat(expiry) - date.fromisoformat(qdate)).days / 365.0, 1e-6)
            if ttm > 2.0:
                continue
            mny = float(strike) / max(spot, 1e-12)
            if mny < 0.7 or mny > 1.3:
                continue

            call = _bs_call_price(spot, float(strike), rate, sigma, ttm)
            if option_type == "put":
                mid = call - spot + float(strike) * math.exp(-rate * ttm)
            else:
                mid = call
            mid = max(mid, 1e-8)
            rel_spread = 0.04 + 0.06 * abs(mny - 1.0)
            spread = max(0.01, rel_spread * mid)
            bid = max(mid - 0.5 * spread, 1e-8)
            ask = mid + 0.5 * spread

            quotes.append(
                OptionChainQuote(
                    strike=float(strike),
                    maturity=float(ttm),
                    option_type=option_type,
                    bid=float(bid),
                    ask=float(ask),
                    iv=float(sigma),
                    quote_date=qdate,
                    expiry=str(expiry),
                    volume=None,
                    open_interest=None,
                    bid_size=None,
                    ask_size=None,
                )
            )

        if not quotes:
            raise RuntimeError("polygon_free returned no usable option contracts for proxy-chain construction")

        exp = expiration or min((q.expiry for q in quotes if q.expiry is not None), default=qdate)
        return MarketDataSnapshot(
            symbol=symbol.upper(),
            source="polygon_free_reference_proxy",
            quote_date=qdate,
            expiration=exp,
            underlying_price=float(spot),
            quotes=quotes,
        )

    if provider == "fmp_free":
        if not api_key:
            raise ValueError("fmp_free provider requires api_key")
        return fetch_fmp_option_chain(
            symbol,
            expiration=expiration,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )

    raise ValueError(f"Unknown provider '{provider}'")


def write_snapshot_json(snapshot: MarketDataSnapshot, path: str | Path) -> str:
    """Persist market-data snapshot to JSON."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "symbol": snapshot.symbol,
        "source": snapshot.source,
        "quote_date": snapshot.quote_date,
        "expiration": snapshot.expiration,
        "underlying_price": snapshot.underlying_price,
        "quotes": [q.__dict__ for q in snapshot.quotes],
    }
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(p)


def load_snapshot_json(path: str | Path) -> MarketDataSnapshot:
    """Load market-data snapshot from JSON."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    quotes = [OptionChainQuote(**row) for row in payload.get("quotes", [])]
    return MarketDataSnapshot(
        symbol=str(payload["symbol"]),
        source=str(payload["source"]),
        quote_date=str(payload["quote_date"]),
        expiration=str(payload["expiration"]),
        underlying_price=payload.get("underlying_price"),
        quotes=quotes,
    )


def load_snapshot_series(paths: Sequence[str | Path]) -> List[MarketDataSnapshot]:
    """Load and sort snapshots by quote date."""

    snapshots = [load_snapshot_json(p) for p in paths]
    return sorted(snapshots, key=lambda x: x.quote_date)
