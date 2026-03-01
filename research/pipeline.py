"""End-to-end doctoral-grade research pipeline orchestration."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from calibration import MarketOption, heston_call_price, implied_volatility
from mc_pricer import Backend, DiscretizationScheme, PayoffType, SimulationConfig, VarianceReduction

from .benchmark import (
    benchmark_to_dict,
    default_benchmark_cases,
    default_benchmark_configs,
    run_benchmark,
    summarize_benchmark,
)
from .ablation import ablation_to_dict, run_ablation_study_from_results
from .challengers import challenger_to_dict, run_challenger_baseline_study
from .crisis import crisis_to_dict, run_crisis_subperiod_study
from .calibration_uq import (
    bayesian_heston_calibration,
    bayesian_result_to_dict,
    bootstrap_heston_calibration,
    bootstrap_result_to_dict,
)
from .cuda_tuning import cuda_tuning_to_dict, run_cuda_autotune
from .claims import default_research_claims, evaluate_claim_bundle
from .cross_sectional import cross_sectional_study_to_dict, run_cross_sectional_rough_heston_study
from .datasets import build_multi_year_synthetic_dataset, panel_dataset_to_dict, resample_dataset
from .multiple_testing import multiple_testing_to_dict, run_global_multiple_testing
from .econometrics import (
    diebold_mariano_test,
    holm_bonferroni_correction,
    model_confidence_set,
    moving_block_bootstrap_ci,
    superior_predictive_ability_test,
    white_reality_check,
)
from .error_decomposition import decomposition_to_dict, estimate_error_decomposition
from .execution_microstructure import run_microstructure_hedging_study
from .experiment_registry import write_experiment_registry
from .forecasting import rolling_forecast_to_dict, run_rolling_oos_forecast_study
from .hedging import compare_hedging_robustness, run_execution_aware_hedging_study
from .hpc_scaling import hpc_scaling_to_dict, run_hpc_scaling_study
from .historical_backtest import (
    generate_synthetic_market_option_timeseries,
    historical_backtest_to_dict,
    run_historical_rough_heston_backtest,
)
from .identifiability import analyze_heston_identifiability, identifiability_to_dict
from .mlmc import compare_mlmc_heston_vs_mc, compare_mlmc_vs_mc
from .model_risk import price_model_ensemble, stress_model_ensemble, summarize_model_risk
from .performance import benchmark_backend_scaling, performance_to_dict, summarize_backend_speedup
from .portfolio_overlay import portfolio_overlay_to_dict, run_portfolio_hedging_overlay
from .paper_pack import generate_paper_package
from .rough_bergomi import calibrate_rough_bergomi_hook, rough_bergomi_to_dict
from .curves import build_flat_curve, forward_price
from .market_data import fetch_option_chain_free
from .real_data_calibration import (
    calibrate_heston_train_test_from_quotes,
    filter_market_quality_quotes,
    generate_synthetic_option_chain,
    train_test_result_to_dict,
)
from .results_chapter import generate_results_chapter
from .regime import regime_diagnostics_to_dict, run_regime_diagnostics
from .reporting import generate_publication_assets
from .repro import (
    collect_manifest,
    verify_reproducibility_hash_bundle,
    write_reproducibility_hash_bundle,
    write_research_artifacts,
)
from .structural_breaks import run_structural_break_diagnostics, structural_breaks_to_dict
from .rough_full_calibration import calibrate_rough_heston_full, rough_full_calibration_to_dict
from .rough_heston import calibrate_rough_heston_hook, rough_heston_calibration_to_dict
from .state_space import run_heston_state_space_filter, state_space_to_dict
from .traceability import generate_traceability_package
from .svi import clean_surface_with_svi, svi_cleaning_to_dict
from .walkforward import run_leakage_free_walkforward, walkforward_to_dict


def _synthetic_market_options(seed: int = 42) -> List[MarketOption]:
    """Generate synthetic market option panel from a known Heston surface."""

    rng = np.random.default_rng(seed)
    spot = 100.0
    rate = 0.03
    true = {"v0": 0.05, "kappa": 2.5, "theta": 0.04, "xi": 0.45, "rho": -0.65}

    strikes = [80, 90, 100, 110, 120]
    maturities = [0.25, 0.5, 1.0]
    options: List[MarketOption] = []

    for t in maturities:
        for k in strikes:
            price = heston_call_price(spot, k, rate, t, **true)
            noisy_price = max(price * (1.0 + rng.normal(0.0, 0.005)), 1e-8)
            iv = implied_volatility(noisy_price, spot, k, rate, t, option_type="call")
            options.append(
                MarketOption(
                    strike=float(k),
                    maturity=float(t),
                    market_price=float(noisy_price),
                    market_iv=float(iv),
                    option_type="call",
                )
            )

    return options


def _extract_efficiency_metric_samples(observations) -> Dict[str, np.ndarray]:
    """Pair baseline/VR samples for claim C1 from repeated benchmark runs."""

    by_key = {}
    for obs in observations:
        metric = obs.std_error * obs.runtime_seconds
        by_key[(obs.case_id, obs.config_label, obs.repeat)] = metric

    baseline = []
    candidate = []
    pair_keys = sorted({(case_id, repeat) for (case_id, _, repeat) in by_key.keys()})
    for case_id, r in pair_keys:
        b = by_key.get((case_id, "baseline_mc", r))
        c = by_key.get((case_id, "vr_mc", r))
        if b is not None and c is not None:
            baseline.append(b)
            candidate.append(c)

    if len(baseline) < 2:
        raise RuntimeError("Insufficient paired benchmark samples for claim C1 efficiency metric")

    return {
        "baseline": np.asarray(baseline, dtype=float),
        "candidate": np.asarray(candidate, dtype=float),
    }


def run_research_pipeline(
    output_dir: str = "artifacts/research",
    *,
    seed: int = 42,
    quick: bool = True,
    benchmark_repeats: Optional[int] = None,
    mlmc_num_runs: Optional[int] = None,
    bootstrap_runs: Optional[int] = None,
    diagnostic_num_paths: Optional[int] = None,
    market_symbol: Optional[str] = None,
    market_data_provider: str = "yahoo_free",
    market_data_api_key: Optional[str] = None,
    market_expiration: Optional[str] = None,
    benchmark_cases=None,
    benchmark_configs=None,
) -> Dict[str, object]:
    """Run complete research pipeline and persist artifacts."""

    repeats = benchmark_repeats if benchmark_repeats is not None else (5 if quick else 8)
    mlmc_repeats = mlmc_num_runs if mlmc_num_runs is not None else (6 if quick else 12)
    n_bootstrap = bootstrap_runs if bootstrap_runs is not None else (8 if quick else 30)
    cases = list(benchmark_cases) if benchmark_cases is not None else default_benchmark_cases()
    configs = list(benchmark_configs) if benchmark_configs is not None else default_benchmark_configs()
    diag_paths = diagnostic_num_paths if diagnostic_num_paths is not None else (6000 if quick else 20000)
    diag_config = SimulationConfig(
        num_paths=diag_paths,
        num_steps=100,
        backend=Backend.NUMPY,
        scheme=DiscretizationScheme.QE,
        variance_reduction=VarianceReduction.ANTITHETIC,
        seed=seed + 400,
    )

    # 1) Benchmark harness
    observations = run_benchmark(
        cases=cases,
        configs=configs,
        repeats=repeats,
        seed_start=seed,
    )
    benchmark_summary = summarize_benchmark(observations)

    # 2) MLMC novelty experiments
    mlmc_runs = []
    for i in range(mlmc_repeats):
        mlmc_runs.append(
            compare_mlmc_vs_mc(
                payoff_style="european",
                seed=seed + 100 + i,
            )
        )
    heston_mlmc_runs = []
    for i in range(max(2, mlmc_repeats // 2)):
        heston_mlmc_runs.append(compare_mlmc_heston_vs_mc(seed=seed + 200 + i))

    # 3) Calibration + UQ (baseline and stronger optimizer budget)
    market_options = _synthetic_market_options(seed=seed)
    bootstrap_baseline = bootstrap_heston_calibration(
        market_options,
        spot=100.0,
        rate=0.03,
        use_iv=False,
        n_bootstrap=n_bootstrap,
        max_iter=80,
        seed=seed,
    )
    bootstrap_candidate = bootstrap_heston_calibration(
        market_options,
        spot=100.0,
        rate=0.03,
        use_iv=False,
        n_bootstrap=n_bootstrap,
        max_iter=220,
        seed=seed + 1,
    )
    bayesian_uq = bayesian_heston_calibration(
        market_options,
        spot=100.0,
        rate=0.03,
        use_iv=False,
        max_iter=180 if quick else 260,
        n_samples=80 if quick else 280,
        burn_in=50 if quick else 180,
        proposal_scale=0.12,
        seed=seed + 2,
    )
    rough_heston_hook = calibrate_rough_heston_hook(
        market_options,
        spot=100.0,
        rate=0.03,
        use_iv=False,
        max_iter=140 if quick else 220,
        hurst_grid=[0.12, 0.2, 0.3, 0.5] if quick else [0.08, 0.12, 0.18, 0.25, 0.35, 0.5],
        xi_scales=[0.85, 1.0, 1.15] if quick else [0.75, 0.9, 1.0, 1.1, 1.25],
        num_paths=350 if quick else 1200,
        num_steps=24 if quick else 64,
        seed=seed + 3,
    )
    rough_heston_full = calibrate_rough_heston_full(
        market_options,
        spot=100.0,
        rate=0.03,
        use_iv=True,
        max_iter=100 if quick else 160,
        num_paths=420 if quick else 1400,
        num_steps=22 if quick else 56,
        n_global_samples=10 if quick else 30,
        local_iterations=6 if quick else 20,
        seed=seed + 4,
    )
    identifiability = analyze_heston_identifiability(
        market_options,
        spot=100.0,
        rate=0.03,
        use_iv=True,
        max_iter=70 if quick else 220,
        profile_points=7 if quick else 13,
        bayesian_samples=50 if quick else 240,
        bayesian_burn_in=30 if quick else 150,
        seed=seed + 5,
    )
    svi_cleaning = clean_surface_with_svi(market_options, spot=100.0)
    rough_bergomi = calibrate_rough_bergomi_hook(
        svi_cleaning.cleaned_options,
        spot=100.0,
        rate=0.03,
        use_iv=False,
        max_iter=80 if quick else 180,
        hurst_grid=[0.12, 0.2, 0.35, 0.5] if quick else [0.08, 0.12, 0.18, 0.25, 0.35, 0.5],
        eta_grid=[0.2, 0.35, 0.5] if quick else [0.15, 0.25, 0.35, 0.5, 0.7],
    )

    market_data = {"source": "synthetic", "status": "fallback", "symbol": market_symbol}
    if market_symbol:
        try:
            snapshot = fetch_option_chain_free(
                market_symbol,
                provider=market_data_provider,
                expiration=market_expiration,
                api_key=market_data_api_key,
            )
            quality_quotes, quality_report = filter_market_quality_quotes(
                snapshot.quotes,
                as_of_date=snapshot.quote_date,
                max_relative_spread=0.45,
                min_open_interest=0 if quick else 5,
                min_volume=0 if quick else 2,
                max_staleness_days=7,
            )
            if len(quality_quotes) >= 8:
                real_chain = quality_quotes
                market_data = {
                    "source": snapshot.source,
                    "status": "ok",
                    "symbol": snapshot.symbol,
                    "quote_date": snapshot.quote_date,
                    "expiration": snapshot.expiration,
                    "underlying_price": snapshot.underlying_price,
                    "raw_quotes": len(snapshot.quotes),
                    "quality_filtered_quotes": len(quality_quotes),
                    "quality_report": asdict(quality_report),
                    "forward_1y": forward_price(
                        snapshot.underlying_price if snapshot.underlying_price is not None else 100.0,
                        1.0,
                        rate_curve=build_flat_curve(0.03),
                        dividend_curve=build_flat_curve(0.0),
                    ),
                }
            else:
                real_chain = generate_synthetic_option_chain(seed=seed + 25)
                market_data = {
                    "source": snapshot.source,
                    "status": "fallback_insufficient_quality_quotes",
                    "symbol": snapshot.symbol,
                    "quote_date": snapshot.quote_date,
                    "expiration": snapshot.expiration,
                    "raw_quotes": len(snapshot.quotes),
                    "quality_filtered_quotes": len(quality_quotes),
                    "quality_report": asdict(quality_report),
                }
        except Exception as exc:
            real_chain = generate_synthetic_option_chain(seed=seed + 25)
            market_data = {
                "source": market_data_provider,
                "status": "fallback_fetch_error",
                "symbol": market_symbol,
                "error": str(exc),
            }
    else:
        real_chain = generate_synthetic_option_chain(seed=seed + 25)

    real_data_calibration = calibrate_heston_train_test_from_quotes(
        real_chain,
        spot=100.0,
        rate=0.03,
        train_fraction=0.7,
        use_iv=False,
        max_iter=220 if not quick else 120,
        seed=seed + 26,
    )
    historical_panels = generate_synthetic_market_option_timeseries(
        start_date="2024-01-05",
        num_dates=8 if quick else 16,
        step_days=7,
        spot=100.0,
        rate=0.03,
        seed=seed + 27,
    )
    historical_backtest = run_historical_rough_heston_backtest(
        historical_panels,
        spot=100.0,
        rate=0.03,
        use_iv=False,
        train_fraction=0.6,
        validate_fraction=0.2,
        max_iter=80 if quick else 180,
        num_paths=320 if quick else 1000,
        num_steps=22 if quick else 48,
        seed=seed + 28,
    )
    state_space_filter = run_heston_state_space_filter(
        historical_panels,
        spot=100.0,
        rate=0.03,
        use_iv=False,
        max_iter=40 if quick else 110,
        process_noise=0.03,
        measurement_noise=0.08,
    )
    multi_year_dataset = resample_dataset(
        build_multi_year_synthetic_dataset(
            symbol="SYNTH",
            start_date="2020-01-03",
            years=1 if quick else 3,
            step_days=14 if quick else 7,
            spot=100.0,
            rate=0.03,
            seed=seed + 31,
        ),
        every_n=2 if quick else 1,
        max_panels=16 if quick else None,
    )
    walkforward = run_leakage_free_walkforward(
        multi_year_dataset.panels,
        spot=100.0,
        rate=0.03,
        use_iv=False,
        train_size=6 if quick else 12,
        validate_size=2 if quick else 4,
        test_size=2 if quick else 4,
        step_size=2 if quick else 2,
        max_windows=2 if quick else 6,
        heston_max_iter=35 if quick else 90,
        rough_max_iter=30 if quick else 70,
        rough_num_paths=90 if quick else 250,
        rough_num_steps=8 if quick else 16,
        seed=seed + 32,
    )
    cross_sectional_study = run_cross_sectional_rough_heston_study(
        symbols=["AAPL", "MSFT", "SPY"] if quick else None,
        num_dates=5 if quick else 12,
        step_days=7,
        spot=100.0,
        rate=0.03,
        use_iv=False,
        train_fraction=0.6,
        validate_fraction=0.2,
        max_iter=35 if quick else 140,
        num_paths=120 if quick else 900,
        num_steps=10 if quick else 36,
        seed=seed + 29,
    )
    forecasting_oos = run_rolling_oos_forecast_study(
        historical_panels,
        spot=100.0,
        rate=0.03,
        use_iv=False,
        heston_max_iter=55 if quick else 160,
        rough_max_iter=45 if quick else 140,
        rough_num_paths=140 if quick else 700,
        rough_num_steps=10 if quick else 32,
        max_transitions=4 if quick else None,
        seed=seed + 30,
    )
    challenger_baselines = run_challenger_baseline_study(
        multi_year_dataset.panels,
        spot=100.0,
        rate=0.03,
        use_iv=True,
        max_transitions=6 if quick else 18,
    )
    forecasting_oos_multi_year = run_rolling_oos_forecast_study(
        multi_year_dataset.panels,
        spot=100.0,
        rate=0.03,
        use_iv=False,
        heston_max_iter=40 if quick else 120,
        rough_max_iter=30 if quick else 90,
        rough_num_paths=80 if quick else 300,
        rough_num_steps=8 if quick else 20,
        max_transitions=6 if quick else 18,
        seed=seed + 33,
    )
    crisis_subperiod = run_crisis_subperiod_study(
        multi_year_dataset.panels,
        forecast_observations=forecasting_oos_multi_year.observations,
        spot=100.0,
    )
    structural_breaks = run_structural_break_diagnostics(
        dates_by_series={
            **{
                f"forecast::{model}": list(forecasting_oos_multi_year.target_dates)
                for model in forecasting_oos_multi_year.losses_by_model.keys()
            },
            "state_space::innovation_l2": [row.quote_date for row in state_space_filter.estimates],
            "state_space::filtered_panel_rmse": [row.quote_date for row in state_space_filter.estimates],
            "state_space::raw_panel_rmse": [row.quote_date for row in state_space_filter.estimates],
        },
        values_by_series={
            **{
                f"forecast::{model}": list(vals)
                for model, vals in forecasting_oos_multi_year.losses_by_model.items()
            },
            "state_space::innovation_l2": [row.innovation_l2 for row in state_space_filter.estimates],
            "state_space::filtered_panel_rmse": [row.panel_rmse for row in state_space_filter.estimates],
            "state_space::raw_panel_rmse": [row.raw_panel_rmse for row in state_space_filter.estimates],
        },
        min_segment_size=2 if quick else 4,
        n_bootstrap=120 if quick else 600,
        seed=seed + 36,
    )
    regime_diagnostics = run_regime_diagnostics(
        historical_panels,
        forecast_observations=forecasting_oos.observations,
        spot=100.0,
    )

    # 4) Model risk and stress
    ensemble = price_model_ensemble(
        spot=100.0,
        strike=100.0,
        rate=0.03,
        maturity=1.0,
        payoff_type=PayoffType.EUROPEAN_CALL,
        sigma=0.2,
        config=diag_config,
    )

    # 4b) Hedging robustness under model misspecification
    hedging_robustness = compare_hedging_robustness(
        s0=100.0,
        strike=100.0,
        r=0.03,
        maturity=1.0,
        sigma=0.2,
        num_paths=1200 if quick else 4000,
        num_steps=26 if quick else 52,
        transaction_cost=0.0005,
        seed=seed + 300,
    )
    execution_aware_hedging = run_execution_aware_hedging_study(
        s0=100.0,
        strike=100.0,
        r=0.03,
        maturity=1.0,
        sigma=0.2,
        num_paths=900 if quick else 3000,
        num_steps=26 if quick else 52,
        seed=seed + 301,
    )
    microstructure_hedging = run_microstructure_hedging_study(
        true_model="heston",
        s0=100.0,
        strike=100.0,
        r=0.03,
        maturity=1.0,
        sigma_hedger=0.2,
        sigma_true=0.2,
        num_paths=600 if quick else 2200,
        num_steps=20 if quick else 52,
        rebalance_every=1,
        transaction_cost=0.0005,
        seed=seed + 304,
    )
    portfolio_overlay = run_portfolio_hedging_overlay(
        symbols=["AAPL", "MSFT", "SPY"] if quick else None,
        s0=100.0,
        strike=100.0,
        r=0.03,
        maturity=1.0,
        base_sigma=0.2,
        num_paths=600 if quick else 2800,
        num_steps=26 if quick else 52,
        transaction_cost=0.0005,
        seed=seed + 302,
    )
    model_risk_summary = summarize_model_risk(ensemble)
    stress_results = stress_model_ensemble(
        spot=100.0,
        strike=100.0,
        rate=0.03,
        maturity=1.0,
        payoff_type=PayoffType.EUROPEAN_CALL,
        sigma=0.2,
        config=diag_config,
    )

    # 5) Performance scaling diagnostics
    perf_results = benchmark_backend_scaling(
        num_paths_grid=[4000, 8000] if quick else [4000, 8000, 16000, 32000],
        num_steps=100,
        seed=seed + 500,
    )
    perf_summary = summarize_backend_speedup(perf_results)
    hpc_scaling = run_hpc_scaling_study(
        problem_sizes=(120_000, 220_000) if quick else (200_000, 500_000, 1_000_000),
        repeats=2 if quick else 4,
        seed=seed + 505,
    )
    cuda_tuning = run_cuda_autotune(
        num_paths=60_000 if quick else 200_000,
        num_steps=84 if quick else 252,
        repeats=1 if quick else 2,
        threads_grid=(64, 128, 256, 512) if quick else (64, 128, 256, 512),
        streams_grid=(1, 2, 4) if quick else (1, 2, 4, 8),
    )

    # 6) Error decomposition on canonical test case
    decomposition = estimate_error_decomposition(
        cases[0],
        seed=seed + 777,
        reference_num_paths=diag_paths,
        bias_num_paths=max(diag_paths // 2, 1000),
    )

    # 7) Claim evaluation
    claims = default_research_claims()
    c1_samples = _extract_efficiency_metric_samples(observations)
    c2_baseline = np.asarray([run["mc"]["runtime_seconds"] for run in heston_mlmc_runs], dtype=float)
    c2_candidate = np.asarray([run["mlmc"]["runtime_seconds"] for run in heston_mlmc_runs], dtype=float)

    c3_candidate = np.asarray(bootstrap_candidate.rmse_samples, dtype=float)
    c3_candidate = np.abs(c3_candidate - np.mean(c3_candidate))
    c3_baseline = np.full_like(c3_candidate, 0.02, dtype=float)

    claim_inputs = {
        "std_error_times_runtime": c1_samples,
        "runtime_seconds": {"baseline": c2_baseline, "candidate": c2_candidate},
        "heston_stability_error": {"baseline": c3_baseline, "candidate": c3_candidate},
    }
    claim_evaluations = evaluate_claim_bundle(claims, claim_inputs, seed=seed)
    if c1_samples["baseline"].size >= 5:
        dm_efficiency = diebold_mariano_test(
            c1_samples["baseline"],
            c1_samples["candidate"],
            alternative="greater",
            lag=1,
        )
    else:
        dm_efficiency = diebold_mariano_test(
            np.pad(c1_samples["baseline"], (0, 5 - c1_samples["baseline"].size), mode="edge"),
            np.pad(c1_samples["candidate"], (0, 5 - c1_samples["candidate"].size), mode="edge"),
            alternative="greater",
            lag=1,
        )
    hist_rmse_series = [x.rmse for x in historical_backtest.test_evaluations]
    if len(hist_rmse_series) < 2:
        hist_rmse_series = [x.rmse for x in historical_backtest.validate_evaluations] + hist_rmse_series
    hist_block_ci = moving_block_bootstrap_ci(
        hist_rmse_series,
        block_size=min(2, len(hist_rmse_series)),
        n_bootstrap=300 if quick else 1200,
        seed=seed + 990,
    )
    holm = holm_bonferroni_correction([ev.p_value for ev in claim_evaluations], alpha=0.05)
    forecast_losses_for_tests: Dict[str, List[float]] = {}
    for model, series in forecasting_oos.losses_by_model.items():
        arr = np.asarray(series, dtype=float)
        if arr.size == 0:
            continue
        if arr.size < 5:
            arr = np.pad(arr, (0, 5 - arr.size), mode="edge")
        forecast_losses_for_tests[model] = [float(x) for x in arr.tolist()]

    spa = superior_predictive_ability_test(
        forecast_losses_for_tests,
        benchmark_model="naive_last_surface",
        n_bootstrap=250 if quick else 1200,
        block_size=2,
        seed=seed + 991,
    )
    white_rc = white_reality_check(
        forecast_losses_for_tests,
        benchmark_model="naive_last_surface",
        n_bootstrap=250 if quick else 1200,
        block_size=2,
        seed=seed + 992,
    )
    mcs = model_confidence_set(
        forecast_losses_for_tests,
        alpha=0.10,
        n_bootstrap=200 if quick else 1000,
        block_size=2,
        seed=seed + 993,
    )

    results = {
        "benchmark": {
            "observations": benchmark_to_dict(observations),
            "summary": benchmark_summary,
        },
        "mlmc": {
            "runs": mlmc_runs,
            "avg_speedup": float(np.mean([r["runtime_speedup_mc_over_mlmc"] for r in mlmc_runs])),
        },
        "heston_mlmc": {
            "runs": heston_mlmc_runs,
            "avg_speedup": float(np.mean([r["runtime_speedup_mc_over_mlmc"] for r in heston_mlmc_runs])),
        },
        "calibration_uq": {
            "baseline": bootstrap_result_to_dict(bootstrap_baseline),
            "candidate": bootstrap_result_to_dict(bootstrap_candidate),
            "bayesian": bayesian_result_to_dict(bayesian_uq),
        },
        "market_data": market_data,
        "rough_heston": rough_heston_calibration_to_dict(rough_heston_hook),
        "rough_heston_full": rough_full_calibration_to_dict(rough_heston_full),
        "rough_bergomi": rough_bergomi_to_dict(rough_bergomi),
        "identifiability": identifiability_to_dict(identifiability),
        "svi_cleaning": svi_cleaning_to_dict(svi_cleaning),
        "real_data_calibration": train_test_result_to_dict(real_data_calibration),
        "historical_backtest": historical_backtest_to_dict(historical_backtest),
        "state_space_filter": state_space_to_dict(state_space_filter),
        "multi_year_dataset": panel_dataset_to_dict(multi_year_dataset),
        "walkforward_leakage_free": walkforward_to_dict(walkforward),
        "cross_sectional_study": cross_sectional_study_to_dict(cross_sectional_study),
        "forecasting_oos": rolling_forecast_to_dict(forecasting_oos),
        "challenger_baselines": challenger_to_dict(challenger_baselines),
        "forecasting_oos_multi_year": rolling_forecast_to_dict(forecasting_oos_multi_year),
        "crisis_subperiod_study": crisis_to_dict(crisis_subperiod),
        "structural_break_diagnostics": structural_breaks_to_dict(structural_breaks),
        "regime_diagnostics": regime_diagnostics_to_dict(regime_diagnostics),
        "model_risk": {
            "ensemble": [asdict(s) for s in ensemble],
            "summary": asdict(model_risk_summary),
            "stress": stress_results,
        },
        "hedging_robustness": hedging_robustness,
        "execution_aware_hedging": execution_aware_hedging,
        "microstructure_hedging": microstructure_hedging,
        "portfolio_overlay": portfolio_overlay_to_dict(portfolio_overlay),
        "statistical_validation": {
            "diebold_mariano_efficiency": asdict(dm_efficiency),
            "historical_rmse_block_bootstrap_ci": asdict(hist_block_ci),
            "holm_bonferroni_claims": holm,
            "spa_vs_naive_forecast": asdict(spa),
            "white_reality_check_vs_naive": asdict(white_rc),
            "model_confidence_set": asdict(mcs),
        },
        "performance": {
            "scaling": performance_to_dict(perf_results),
            "summary": perf_summary,
        },
        "hpc_scaling": hpc_scaling_to_dict(hpc_scaling),
        "cuda_tuning": cuda_tuning_to_dict(cuda_tuning),
        "error_decomposition": decomposition_to_dict(decomposition),
    }
    results["ablation_study"] = ablation_to_dict(
        run_ablation_study_from_results(results, n_bootstrap=500 if quick else 2000, seed=seed + 1200)
    )

    claim_payload = [
        {
            "claim": asdict(claim),
            "evaluation": {
                "claim_id": ev.claim_id,
                "passed": ev.passed,
                "observed_effect": ev.observed_effect,
                "confidence_interval": asdict(ev.confidence_interval),
                "p_value": ev.p_value,
                "details": ev.details,
            },
        }
        for claim, ev in zip(claims, claim_evaluations)
    ]
    results["global_multiple_testing"] = multiple_testing_to_dict(
        run_global_multiple_testing(results=results, claims=claim_payload, alpha=0.05)
    )

    manifest = collect_manifest(seed=seed, cwd=Path.cwd())
    artifact_paths = write_research_artifacts(
        output_dir=Path(output_dir),
        manifest=manifest,
        results=results,
        claims=claim_payload,
    )
    publication_assets = generate_publication_assets(
        output_dir=Path(output_dir) / "publication_assets",
        results=results,
        claims=claim_payload,
    )
    artifact_paths["publication_assets"] = str(Path(output_dir) / "publication_assets")
    artifact_paths["publication_index"] = str(Path(output_dir) / "publication_assets" / "tables")
    artifact_paths["publication_files"] = publication_assets
    experiment_registry = write_experiment_registry(
        output_dir=Path(output_dir) / "registry",
        manifest=manifest,
        results=results,
        tags={
            "mode": "quick" if quick else "full",
            "market_provider": market_data_provider,
            "market_symbol": str(market_symbol),
        },
    )
    artifact_paths["experiment_registry"] = experiment_registry
    paper_pack = generate_paper_package(
        output_dir=Path(output_dir) / "paper_package",
        results=results,
        claims=claim_payload,
        artifact_paths=artifact_paths,
    )
    artifact_paths["paper_package"] = paper_pack
    repro_bundle = write_reproducibility_hash_bundle(
        output_dir=Path(output_dir),
        manifest=manifest,
        seed=seed,
    )
    artifact_paths["reproducibility_hash_bundle"] = repro_bundle
    artifact_paths["reproducibility_verification"] = verify_reproducibility_hash_bundle(repro_bundle)
    results_chapter = generate_results_chapter(
        output_dir=Path(output_dir) / "results_chapter",
        results=results,
        claims=claim_payload,
        artifact_paths=artifact_paths,
    )
    artifact_paths["results_chapter"] = results_chapter
    traceability = generate_traceability_package(
        output_dir=Path(output_dir) / "traceability",
        claims=claim_payload,
        results=results,
        artifact_paths=artifact_paths,
    )
    artifact_paths["traceability_package"] = traceability

    return {
        "artifact_paths": artifact_paths,
        "claims": claim_payload,
        "results": results,
    }


def main() -> None:
    """CLI entrypoint for research pipeline."""

    parser = argparse.ArgumentParser(description="Run doctoral-grade research pipeline")
    parser.add_argument("--output-dir", default="artifacts/research", help="Directory for artifacts")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full (slower) experiment set instead of quick mode",
    )
    parser.add_argument("--market-symbol", default=None, help="Optional symbol for free option-chain fetch (e.g. AAPL)")
    parser.add_argument("--market-provider", default="yahoo_free", help="Market data provider (default: yahoo_free)")
    parser.add_argument("--market-api-key", default=None, help="Optional API key for key-based free providers")
    parser.add_argument("--market-expiration", default=None, help="Optional expiration date YYYY-MM-DD for market fetch")
    args = parser.parse_args()

    out = run_research_pipeline(
        output_dir=args.output_dir,
        seed=args.seed,
        quick=not args.full,
        market_symbol=args.market_symbol,
        market_data_provider=args.market_provider,
        market_data_api_key=args.market_api_key,
        market_expiration=args.market_expiration,
    )

    print("Research pipeline complete.")
    for key, value in out["artifact_paths"].items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
