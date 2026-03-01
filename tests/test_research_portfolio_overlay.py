"""Tests for portfolio-level hedging risk overlays."""

from research.portfolio_overlay import portfolio_overlay_to_dict, run_portfolio_hedging_overlay


def test_portfolio_overlay_runs():
    out = run_portfolio_hedging_overlay(
        symbols=["AAPL", "MSFT", "SPY"],
        weights=[0.5, 0.3, 0.2],
        num_paths=500,
        num_steps=20,
        seed=19,
    )
    assert len(out.asset_risks) == 3
    assert out.portfolio_var95_loss >= 0.0
    assert out.portfolio_cvar95_loss >= out.portfolio_var95_loss
    assert out.diversification_ratio > 0.0
    assert out.tail_sample_size > 0

    payload = portfolio_overlay_to_dict(out)
    assert len(payload["correlation_matrix"]) == 3
    assert "expected_shortfall_contributions" in payload
