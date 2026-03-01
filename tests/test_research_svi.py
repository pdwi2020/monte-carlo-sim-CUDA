"""Tests for SVI-based surface cleaning."""

from research.real_data_calibration import generate_synthetic_option_chain, quotes_to_market_options
from research.svi import clean_surface_with_svi, svi_cleaning_to_dict


def test_svi_cleaning_runs():
    quotes = generate_synthetic_option_chain(seed=64)
    opts = quotes_to_market_options(quotes)
    out = clean_surface_with_svi(opts, spot=100.0)
    assert len(out.cleaned_options) == len(opts)
    payload = svi_cleaning_to_dict(out)
    assert "arbitrage_report" in payload
    assert payload["num_cleaned_options"] == len(opts)
