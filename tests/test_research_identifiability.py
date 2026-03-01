"""Tests for identifiability diagnostics."""

import pytest

from calibration import SCIPY_AVAILABLE
from research.identifiability import analyze_heston_identifiability, identifiability_to_dict
from research.real_data_calibration import generate_synthetic_option_chain, quotes_to_market_options


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy required for calibration tests")
def test_identifiability_diagnostics_run():
    quotes = generate_synthetic_option_chain(seed=31)
    options = quotes_to_market_options(quotes)

    out = analyze_heston_identifiability(
        options,
        spot=100.0,
        rate=0.03,
        use_iv=True,
        max_iter=80,
        profile_points=7,
        bayesian_samples=40,
        bayesian_burn_in=25,
        seed=32,
    )

    assert len(out.profile_slices) == 5
    assert out.posterior_sample_size == 40
    assert out.posterior_geometry.condition_number >= 1.0
    assert 0.0 <= out.posterior_acceptance_rate <= 1.0

    payload = identifiability_to_dict(out)
    assert "posterior_geometry" in payload
    assert len(payload["posterior_geometry"]["corr_matrix"]) == 5
