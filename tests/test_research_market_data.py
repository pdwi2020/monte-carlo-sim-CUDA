"""Tests for free-tier market data adapters."""

from pathlib import Path

from research.market_data import fetch_option_chain_free, fetch_yahoo_option_chain, load_snapshot_json, write_snapshot_json


def test_fetch_yahoo_option_chain_parsing_with_mock(monkeypatch):
    payload = {
        "optionChain": {
            "result": [
                {
                    "quote": {"regularMarketPrice": 100.0, "regularMarketTime": 1704067200},
                    "options": [
                        {
                            "expirationDate": 1706659200,
                            "calls": [
                                {
                                    "strike": 100.0,
                                    "bid": 2.0,
                                    "ask": 2.2,
                                    "impliedVolatility": 0.25,
                                    "volume": 10,
                                    "openInterest": 20,
                                    "lastTradeDate": 1704067200,
                                }
                            ],
                            "puts": [],
                        }
                    ],
                }
            ]
        }
    }

    monkeypatch.setattr("research.market_data._fetch_json", lambda url, timeout_seconds=12: payload)
    snap = fetch_yahoo_option_chain("AAPL", expiration="2024-01-31")
    assert snap.symbol == "AAPL"
    assert len(snap.quotes) == 1
    assert snap.quotes[0].option_type == "call"


def test_snapshot_roundtrip(tmp_path: Path, monkeypatch):
    payload = {
        "optionChain": {
            "result": [
                {
                    "quote": {"regularMarketPrice": 100.0, "regularMarketTime": 1704067200},
                    "options": [{"expirationDate": 1706659200, "calls": [], "puts": []}],
                }
            ]
        }
    }
    monkeypatch.setattr("research.market_data._fetch_json", lambda url, timeout_seconds=12: payload)
    snap = fetch_yahoo_option_chain("MSFT")
    path = tmp_path / "snap.json"
    write_snapshot_json(snap, path)
    loaded = load_snapshot_json(path)
    assert loaded.symbol == snap.symbol
    assert loaded.quote_date == snap.quote_date


def test_polygon_parser_with_mock(monkeypatch):
    payload = {
        "results": [
            {
                "details": {"strike_price": 100.0, "expiration_date": "2024-03-15", "contract_type": "call"},
                "last_quote": {"bid": 1.0, "ask": 1.2, "bid_size": 10, "ask_size": 12},
                "implied_volatility": 0.22,
                "open_interest": 50,
                "day": {"volume": 20},
            }
        ]
    }
    monkeypatch.setattr("research.market_data._fetch_json", lambda url, timeout_seconds=12: payload)
    snap = fetch_option_chain_free("SPY", provider="polygon_free", api_key="demo")
    assert snap.source == "polygon_free"
    assert len(snap.quotes) == 1
    assert snap.quotes[0].open_interest == 50


def test_polygon_reference_proxy_fallback(monkeypatch):
    calls = {"n": 0}

    def fake_fetch(url, timeout_seconds=12):
        calls["n"] += 1
        if "snapshot/options" in url:
            raise RuntimeError("forbidden")
        if "/prev" in url:
            return {"results": [{"c": 100.0}]}
        if "/range/1/day/" in url:
            return {"results": [{"c": 100.0}, {"c": 101.0}, {"c": 99.0}, {"c": 102.0}, {"c": 100.0}, {"c": 103.0}]}
        return {
            "results": [
                {"strike_price": 100.0, "expiration_date": "2024-03-15", "contract_type": "call"},
                {"strike_price": 100.0, "expiration_date": "2024-03-15", "contract_type": "put"},
            ]
        }

    monkeypatch.setattr("research.market_data._fetch_json", fake_fetch)
    snap = fetch_option_chain_free("SPY", provider="polygon_free", api_key="demo")
    assert snap.source == "polygon_free_reference_proxy"
    assert len(snap.quotes) >= 1


def test_fmp_parser_with_mock(monkeypatch):
    payload = {
        "options": [
            {
                "strikePrice": 100.0,
                "expirationDate": "2026-06-19",
                "optionType": "call",
                "bid": 1.1,
                "ask": 1.3,
                "impliedVolatility": 0.24,
                "openInterest": 200,
                "volume": 50,
                "quoteDate": "2026-03-01",
                "underlyingPrice": 102.5,
            }
        ]
    }
    monkeypatch.setattr("research.market_data._fetch_json", lambda url, timeout_seconds=12: payload)
    snap = fetch_option_chain_free("AAPL", provider="fmp_free", api_key="demo")
    assert snap.source == "fmp_free"
    assert len(snap.quotes) == 1
    assert snap.quotes[0].option_type == "call"
    assert snap.quotes[0].open_interest == 200


def test_fmp_endpoint_fallback(monkeypatch):
    calls = {"n": 0}

    def fake_fetch(url, timeout_seconds=12):
        calls["n"] += 1
        if "stable/options-chain" in url:
            raise RuntimeError("blocked endpoint")
        return {
            "data": [
                {
                    "strike": 95.0,
                    "expiration": "2026-06-19",
                    "type": "put",
                    "lastPrice": 2.4,
                }
            ]
        }

    monkeypatch.setattr("research.market_data._fetch_json", fake_fetch)
    snap = fetch_option_chain_free("MSFT", provider="fmp_free", api_key="demo")
    assert len(snap.quotes) == 1
    assert snap.quotes[0].option_type == "put"
    assert calls["n"] >= 2


def test_fmp_quote_proxy_fallback_when_chain_unavailable(monkeypatch):
    def fake_fetch(url, timeout_seconds=12):
        if "stable/options-chain" in url:
            return []
        if "api/v3/options-chain" in url or "api/v4/options-chain" in url:
            raise RuntimeError("blocked")
        if "stable/quote" in url:
            return [{"symbol": "AAPL", "price": 200.0}]
        if "stable/historical-price-eod/light" in url:
            return [{"symbol": "AAPL", "date": f"2026-01-{d:02d}", "price": 200.0 + (d % 5)} for d in range(1, 80)]
        return {}

    monkeypatch.setattr("research.market_data._fetch_json", fake_fetch)
    snap = fetch_option_chain_free("AAPL", provider="fmp_free", api_key="demo")
    assert snap.source == "fmp_free_quote_proxy"
    assert snap.underlying_price == 200.0
    assert len(snap.quotes) >= 18
