"""Tests for real-data calibration pipeline."""

import pytest

from calibration import SCIPY_AVAILABLE
from research.real_data_calibration import (
    calibrate_heston_train_test_from_quotes,
    filter_market_quality_quotes,
    generate_synthetic_option_chain,
)


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy required for calibration tests")
def test_real_data_train_test_calibration_runs():
    quotes = generate_synthetic_option_chain(seed=11)

    # Inject one obviously invalid quote to verify filtering.
    quotes.append(
        type(quotes[0])(
            strike=100.0,
            maturity=0.5,
            option_type="call",
            bid=10.0,
            ask=9.0,
            iv=0.2,
        )
    )

    out = calibrate_heston_train_test_from_quotes(
        quotes,
        spot=100.0,
        rate=0.03,
        train_fraction=0.7,
        use_iv=False,
        max_iter=100,
        seed=12,
    )

    assert out.train_size > 0
    assert out.test_size > 0
    assert out.no_arb_report.removed_invalid_quotes >= 1
    assert out.train_rmse >= 0.0
    assert out.test_rmse >= 0.0


def test_market_quality_filter_flags_wide_spread_and_stale():
    quotes = generate_synthetic_option_chain(seed=13)
    q0 = quotes[0]
    q0.ask = q0.bid * 4.0
    q0.quote_date = "2024-01-01"
    q1 = quotes[1]
    q1.quote_date = "2024-01-01"

    out, report = filter_market_quality_quotes(
        quotes,
        as_of_date="2024-01-20",
        max_relative_spread=0.5,
        min_open_interest=0,
        min_volume=0,
        max_staleness_days=7,
    )
    assert len(out) < len(quotes)
    assert report.removed_wide_spread >= 1
    assert report.removed_stale_quotes >= 1
