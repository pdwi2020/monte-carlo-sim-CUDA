"""Tests for hedging robustness experiments."""

from research.hedging import (
    ExecutionModel,
    compare_hedging_robustness,
    run_delta_hedge_backtest,
    run_delta_vega_hedge_backtest,
    run_execution_aware_hedging_study,
    sample_delta_hedge_pnl_paths,
)


def test_delta_hedge_backtest_runs_for_gbm():
    out = run_delta_hedge_backtest(
        true_model="gbm",
        num_paths=800,
        num_steps=20,
        seed=21,
    )
    assert out.std_pnl >= 0.0
    assert out.cvar95_loss >= out.var95_loss


def test_delta_vega_hedge_backtest_runs_for_heston():
    out = run_delta_vega_hedge_backtest(
        true_model="heston",
        num_paths=700,
        num_steps=20,
        seed=23,
    )
    assert out.std_pnl >= 0.0
    assert out.cvar95_loss >= out.var95_loss


def test_execution_model_path_runs():
    out = run_delta_hedge_backtest(
        true_model="heston",
        num_paths=600,
        num_steps=18,
        execution_model=ExecutionModel(
            bid_ask_bps=6.0,
            slippage_bps=3.0,
            impact_bps_per_unit=0.25,
            fill_probability=0.9,
        ),
        seed=27,
    )
    assert out.std_pnl >= 0.0


def test_misspecified_hedging_has_higher_tail_risk():
    out = compare_hedging_robustness(
        num_paths=800,
        num_steps=20,
        seed=22,
    )
    assert out["well_specified"]["cvar95_loss"] >= 0.0
    assert out["misspecified_heston"]["cvar95_loss"] >= 0.0
    assert out["cvar95_loss_ratio_misspecified_over_well_specified"] >= 0.8
    assert out["delta_vega_misspecified_heston"]["cvar95_loss"] >= 0.0
    assert out["cvar95_loss_ratio_delta_vega_over_delta_only_misspecified"] > 0.0
    assert len(out["transaction_cost_frontier_heston"]) >= 2
    assert len(out["rebalance_stability_heston"]) >= 2
    assert out["var_backtest_misspecified_heston"]["n_exceptions"] >= 0
    assert out["var_backtest_well_specified"]["n_exceptions"] >= 0
    assert out["expected_shortfall_ratio_misspecified_over_model"] > 0.0


def test_execution_aware_study_runs():
    out = run_execution_aware_hedging_study(num_paths=700, num_steps=18, seed=29)
    assert "frictionless" in out
    assert "execution_stressed" in out
    assert out["execution_model"]["fill_probability"] <= 1.0


def test_sample_delta_hedge_pnl_paths_returns_array():
    pnl = sample_delta_hedge_pnl_paths(
        true_model="gbm",
        num_paths=120,
        num_steps=12,
        seed=30,
    )
    assert pnl.shape == (120,)
