"""Benchmark harness for pricing models, convergence, and cost/error analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np

from mc_pricer import (
    Backend,
    BarrierParams,
    DiscretizationScheme,
    HestonParams,
    JumpParams,
    MarketData,
    PayoffType,
    RoughHestonParams,
    SABRParams,
    SCIPY_AVAILABLE,
    SimulationConfig,
    VarianceReduction,
    black_scholes_call,
    black_scholes_put,
    price_option,
    price_rough_heston_option,
    price_sabr_option,
)


@dataclass
class BenchmarkCase:
    """Single benchmark scenario."""

    case_id: str
    name: str
    model: str  # gbm | heston | bates | sabr | rough_heston
    payoff_type: PayoffType
    market: MarketData
    strike: float
    maturity: float
    sigma: Optional[float] = None
    heston: Optional[HestonParams] = None
    jump: Optional[JumpParams] = None
    barrier: Optional[BarrierParams] = None
    sabr: Optional[SABRParams] = None
    rough: Optional[RoughHestonParams] = None


@dataclass
class BenchmarkConfig:
    """Simulation configuration label for benchmark sweeps."""

    label: str
    num_paths: int
    num_steps: int
    variance_reduction: VarianceReduction = VarianceReduction.ANTITHETIC
    scheme: DiscretizationScheme = DiscretizationScheme.QE
    backend: Backend = Backend.NUMPY
    use_sobol: bool = False


@dataclass
class BenchmarkObservation:
    """One benchmark run result."""

    case_id: str
    case_name: str
    config_label: str
    repeat: int
    price: float
    std_error: float
    runtime_seconds: float
    reference_price: Optional[float]
    abs_error: Optional[float]


def default_benchmark_cases() -> List[BenchmarkCase]:
    """Curated baseline scenarios spanning supported model classes."""

    market = MarketData(S0=100.0, r=0.03, q=0.0)
    heston = HestonParams(v0=0.04, kappa=2.2, theta=0.04, xi=0.5, rho=-0.6)
    jump = JumpParams(lambda_j=0.35, mu_j=-0.08, sigma_j=0.15)

    return [
        BenchmarkCase(
            case_id="gbm_eur_call_atm",
            name="GBM European ATM Call",
            model="gbm",
            payoff_type=PayoffType.EUROPEAN_CALL,
            market=market,
            strike=100.0,
            maturity=1.0,
            sigma=0.2,
        ),
        BenchmarkCase(
            case_id="gbm_asian_call",
            name="GBM Asian Arithmetic Call",
            model="gbm",
            payoff_type=PayoffType.ASIAN_CALL,
            market=market,
            strike=100.0,
            maturity=1.0,
            sigma=0.2,
        ),
        BenchmarkCase(
            case_id="heston_eur_put",
            name="Heston European Put",
            model="heston",
            payoff_type=PayoffType.EUROPEAN_PUT,
            market=market,
            strike=95.0,
            maturity=0.75,
            heston=heston,
        ),
        BenchmarkCase(
            case_id="bates_otm_put",
            name="Bates OTM Put",
            model="bates",
            payoff_type=PayoffType.EUROPEAN_PUT,
            market=market,
            strike=85.0,
            maturity=1.0,
            heston=heston,
            jump=jump,
        ),
        BenchmarkCase(
            case_id="sabr_eur_call",
            name="SABR European Call",
            model="sabr",
            payoff_type=PayoffType.EUROPEAN_CALL,
            market=market,
            strike=100.0,
            maturity=1.0,
            sabr=SABRParams(alpha=0.3, beta=0.6, rho=-0.3, nu=0.2),
        ),
        BenchmarkCase(
            case_id="rough_heston_call",
            name="Rough Heston European Call",
            model="rough_heston",
            payoff_type=PayoffType.EUROPEAN_CALL,
            market=market,
            strike=100.0,
            maturity=1.0,
            rough=RoughHestonParams(v0=0.04, theta=0.04, lambda_=2.0, nu=0.3, rho=-0.6, H=0.15),
        ),
    ]


def default_benchmark_configs() -> List[BenchmarkConfig]:
    """Reference simulation configurations for convergence/cost comparisons."""

    return [
        BenchmarkConfig(
            label="baseline_mc",
            num_paths=8192,
            num_steps=80,
            variance_reduction=VarianceReduction.NONE,
            scheme=DiscretizationScheme.EULER,
        ),
        BenchmarkConfig(
            label="vr_mc",
            num_paths=2048,
            num_steps=80,
            variance_reduction=VarianceReduction.ANTITHETIC_CV,
            scheme=DiscretizationScheme.QE,
            use_sobol=SCIPY_AVAILABLE,
        ),
    ]


def _analytic_reference(case: BenchmarkCase) -> Optional[float]:
    if case.model != "gbm" or not SCIPY_AVAILABLE:
        return None
    if case.payoff_type == PayoffType.EUROPEAN_CALL and case.sigma is not None:
        return black_scholes_call(case.market.S0, case.strike, case.market.r, case.sigma, case.maturity)
    if case.payoff_type == PayoffType.EUROPEAN_PUT and case.sigma is not None:
        return black_scholes_put(case.market.S0, case.strike, case.market.r, case.sigma, case.maturity)
    return None


def _run_single(case: BenchmarkCase, sim_cfg: SimulationConfig):
    if case.model == "sabr":
        if case.sabr is None:
            raise ValueError("SABR case requires sabr parameters")
        return price_sabr_option(
            F0=case.market.S0,
            K=case.strike,
            T=case.maturity,
            r=case.market.r,
            sabr=case.sabr,
            is_call=case.payoff_type == PayoffType.EUROPEAN_CALL,
            config=sim_cfg,
        )

    if case.model == "rough_heston":
        if case.rough is None:
            raise ValueError("rough_heston case requires rough parameters")
        return price_rough_heston_option(
            market=case.market,
            K=case.strike,
            T=case.maturity,
            params=case.rough,
            payoff_type=case.payoff_type,
            config=sim_cfg,
        )

    return price_option(
        market=case.market,
        K=case.strike,
        T=case.maturity,
        payoff_type=case.payoff_type,
        heston=case.heston,
        jump=case.jump,
        barrier=case.barrier,
        sigma=case.sigma,
        config=sim_cfg,
    )


def run_benchmark(
    cases: Optional[Iterable[BenchmarkCase]] = None,
    configs: Optional[Iterable[BenchmarkConfig]] = None,
    repeats: int = 5,
    seed_start: int = 1234,
) -> List[BenchmarkObservation]:
    """Execute benchmark sweep and return granular observations."""

    case_list = list(cases) if cases is not None else default_benchmark_cases()
    config_list = list(configs) if configs is not None else default_benchmark_configs()

    observations: List[BenchmarkObservation] = []

    for case in case_list:
        ref = _analytic_reference(case)
        for cfg in config_list:
            for repeat in range(repeats):
                sim_cfg = SimulationConfig(
                    num_paths=cfg.num_paths,
                    num_steps=cfg.num_steps,
                    backend=cfg.backend,
                    scheme=cfg.scheme,
                    variance_reduction=cfg.variance_reduction,
                    seed=seed_start + repeat,
                    use_sobol=cfg.use_sobol,
                )
                result = _run_single(case, sim_cfg)
                abs_error = None if ref is None else abs(result.price - ref)
                observations.append(
                    BenchmarkObservation(
                        case_id=case.case_id,
                        case_name=case.name,
                        config_label=cfg.label,
                        repeat=repeat,
                        price=result.price,
                        std_error=result.std_error,
                        runtime_seconds=float(result.elapsed_time or 0.0),
                        reference_price=ref,
                        abs_error=abs_error,
                    )
                )

    return observations


def summarize_benchmark(observations: List[BenchmarkObservation]) -> List[Dict[str, float]]:
    """Aggregate benchmark observations by case/config into a summary table."""

    if not observations:
        return []

    rows: List[Dict[str, float]] = []
    grouped: Dict[tuple, List[BenchmarkObservation]] = {}
    for obs in observations:
        key = (obs.case_id, obs.case_name, obs.config_label)
        grouped.setdefault(key, []).append(obs)

    for (case_id, case_name, config_label), group in grouped.items():
        prices = np.array([g.price for g in group], dtype=float)
        std_errs = np.array([g.std_error for g in group], dtype=float)
        runtimes = np.array([g.runtime_seconds for g in group], dtype=float)
        abs_errors = np.array([g.abs_error for g in group if g.abs_error is not None], dtype=float)

        row = {
            "case_id": case_id,
            "case_name": case_name,
            "config_label": config_label,
            "mean_price": float(np.mean(prices)),
            "price_std": float(np.std(prices, ddof=1)) if prices.size > 1 else 0.0,
            "mean_std_error": float(np.mean(std_errs)),
            "mean_runtime_seconds": float(np.mean(runtimes)),
            "rmse": float(np.sqrt(np.mean(abs_errors ** 2))) if abs_errors.size > 0 else np.nan,
            "rmse_times_runtime": float(np.sqrt(np.mean(abs_errors ** 2)) * np.mean(runtimes))
            if abs_errors.size > 0
            else np.nan,
        }
        rows.append(row)

    rows.sort(key=lambda r: (r["case_id"], r["config_label"]))
    return rows


def benchmark_to_dict(observations: List[BenchmarkObservation]) -> List[Dict[str, object]]:
    """Serialize observations to dictionaries."""

    return [asdict(o) for o in observations]
