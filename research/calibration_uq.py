"""Calibration uncertainty quantification utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from calibration import CalibrationResult, MarketOption, calibrate_heston
from .statistics import ConfidenceInterval, bootstrap_ci


@dataclass
class ParameterUQ:
    """Uncertainty summary for one calibrated parameter."""

    name: str
    mean: float
    std: float
    ci_low: float
    ci_high: float


@dataclass
class BootstrapCalibrationResult:
    """Bootstrap calibration summary."""

    base: CalibrationResult
    successful_runs: int
    total_runs: int
    parameter_uq: Dict[str, ParameterUQ]
    rmse_ci: ConfidenceInterval
    rmse_samples: List[float]
    parameter_samples: Dict[str, List[float]]


@dataclass
class BayesianCalibrationResult:
    """Posterior diagnostics from Bayesian Heston calibration."""

    base: CalibrationResult
    acceptance_rate: float
    n_samples: int
    burn_in: int
    parameter_uq: Dict[str, ParameterUQ]
    posterior_samples: Dict[str, List[float]]
    log_posterior_ci: ConfidenceInterval


_PARAM_ORDER = ("v0", "kappa", "theta", "xi", "rho")
_PARAM_BOUNDS = np.asarray(
    [
        (0.001, 1.0),
        (0.01, 10.0),
        (0.001, 1.0),
        (0.01, 2.0),
        (-0.99, 0.99),
    ],
    dtype=float,
)


def _extract_targets(options: Sequence[MarketOption], use_iv: bool) -> Tuple[List[MarketOption], np.ndarray]:
    filtered: List[MarketOption] = []
    targets: List[float] = []
    for opt in options:
        if use_iv and opt.market_iv is not None:
            filtered.append(opt)
            targets.append(float(opt.market_iv))
        elif opt.market_price is not None:
            filtered.append(opt)
            targets.append(float(opt.market_price))
        elif opt.mid_price is not None:
            filtered.append(opt)
            targets.append(float(opt.mid_price))
    return filtered, np.asarray(targets, dtype=float)


def _model_values_for_params(
    params: np.ndarray,
    options: Sequence[MarketOption],
    *,
    spot: float,
    rate: float,
    use_iv: bool,
) -> np.ndarray:
    from calibration import heston_call_price, implied_volatility

    v0, kappa, theta, xi, rho = [float(x) for x in params]
    vals: List[float] = []
    for opt in options:
        try:
            call_price = heston_call_price(spot, opt.strike, rate, opt.maturity, v0, kappa, theta, xi, rho)
            if opt.option_type == "put":
                model_price = call_price - spot + opt.strike * np.exp(-rate * opt.maturity)
            else:
                model_price = call_price

            if use_iv and opt.market_iv is not None:
                model_val = implied_volatility(model_price, spot, opt.strike, rate, opt.maturity, opt.option_type)
            else:
                model_val = model_price
        except Exception:
            model_val = np.nan
        vals.append(float(model_val))
    return np.asarray(vals, dtype=float)


def _log_posterior(
    params: np.ndarray,
    *,
    options: Sequence[MarketOption],
    targets: np.ndarray,
    spot: float,
    rate: float,
    use_iv: bool,
    prior_mean: np.ndarray,
    prior_std: np.ndarray,
    noise_scale: float,
) -> float:
    low = _PARAM_BOUNDS[:, 0]
    high = _PARAM_BOUNDS[:, 1]
    if np.any(params < low) or np.any(params > high):
        return -np.inf

    v0, kappa, theta, xi, _rho = [float(x) for x in params]
    feller_gap = xi ** 2 - 2.0 * kappa * theta
    feller_penalty = 0.0 if feller_gap <= 0 else 50.0 * (feller_gap ** 2)

    model = _model_values_for_params(
        params,
        options,
        spot=spot,
        rate=rate,
        use_iv=use_iv,
    )
    if np.any(~np.isfinite(model)):
        return -np.inf

    residual = model - targets
    log_like = -0.5 * float(np.sum((residual / noise_scale) ** 2))
    z = (params - prior_mean) / prior_std
    log_prior = -0.5 * float(np.sum(z ** 2))
    return log_like + log_prior - feller_penalty


def _resample_options(options: List[MarketOption], rng: np.random.Generator) -> List[MarketOption]:
    idx = rng.integers(0, len(options), size=len(options))
    return [options[i] for i in idx]


def bootstrap_heston_calibration(
    market_options: List[MarketOption],
    spot: float,
    rate: float,
    *,
    use_iv: bool = True,
    n_bootstrap: int = 30,
    max_iter: int = 200,
    seed: Optional[int] = 42,
) -> BootstrapCalibrationResult:
    """Run bootstrap UQ over Heston calibration parameters."""

    if len(market_options) < 5:
        raise ValueError("At least 5 market options are required for calibration bootstrap")

    base = calibrate_heston(
        market_options,
        spot=spot,
        rate=rate,
        use_iv=use_iv,
        max_iter=max_iter,
    )

    rng = np.random.default_rng(seed)
    params = {"v0": [], "kappa": [], "theta": [], "xi": [], "rho": []}
    rmse_values: List[float] = []
    successes = 0

    for _ in range(n_bootstrap):
        sample = _resample_options(market_options, rng)
        try:
            result = calibrate_heston(
                sample,
                spot=spot,
                rate=rate,
                use_iv=use_iv,
                max_iter=max_iter,
            )
        except Exception:
            continue

        if not result.success:
            continue

        successes += 1
        rmse_values.append(float(result.rmse))
        for name in params:
            params[name].append(float(result.parameters[name]))

    if successes == 0:
        raise RuntimeError("No successful bootstrap calibrations")

    uq: Dict[str, ParameterUQ] = {}
    for name, values in params.items():
        arr = np.asarray(values, dtype=float)
        ci = bootstrap_ci(arr, statistic=np.mean, n_bootstrap=1000, seed=seed)
        uq[name] = ParameterUQ(
            name=name,
            mean=float(np.mean(arr)),
            std=float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            ci_low=ci.low,
            ci_high=ci.high,
        )

    rmse_ci = bootstrap_ci(np.asarray(rmse_values, dtype=float), statistic=np.mean, n_bootstrap=1000, seed=seed)

    return BootstrapCalibrationResult(
        base=base,
        successful_runs=successes,
        total_runs=n_bootstrap,
        parameter_uq=uq,
        rmse_ci=rmse_ci,
        rmse_samples=rmse_values,
        parameter_samples={k: list(v) for k, v in params.items()},
    )


def bootstrap_result_to_dict(result: BootstrapCalibrationResult) -> Dict[str, object]:
    """Serialize bootstrap result to plain dict."""

    return {
        "base": {
            "success": result.base.success,
            "parameters": result.base.parameters,
            "objective_value": result.base.objective_value,
            "rmse": result.base.rmse,
            "num_iterations": result.base.num_iterations,
            "calibration_time": result.base.calibration_time,
            "message": result.base.message,
        },
        "successful_runs": result.successful_runs,
        "total_runs": result.total_runs,
        "parameter_uq": {k: asdict(v) for k, v in result.parameter_uq.items()},
        "rmse_ci": asdict(result.rmse_ci),
        "rmse_samples": list(result.rmse_samples),
        "parameter_samples": {k: list(v) for k, v in result.parameter_samples.items()},
    }


def bayesian_heston_calibration(
    market_options: List[MarketOption],
    *,
    spot: float,
    rate: float,
    use_iv: bool = True,
    max_iter: int = 200,
    n_samples: int = 250,
    burn_in: int = 150,
    proposal_scale: float = 0.15,
    seed: Optional[int] = 42,
) -> BayesianCalibrationResult:
    """Run random-walk Metropolis-Hastings around calibrated Heston parameters."""

    if len(market_options) < 5:
        raise ValueError("At least 5 market options are required for Bayesian calibration")

    options, targets = _extract_targets(market_options, use_iv=use_iv)
    if len(options) < 5:
        raise ValueError("At least 5 options with usable market targets are required")

    base = calibrate_heston(
        options,
        spot=spot,
        rate=rate,
        use_iv=use_iv,
        max_iter=max_iter,
    )

    current = np.asarray([base.parameters[name] for name in _PARAM_ORDER], dtype=float)
    prior_mean = current.copy()
    prior_std = np.asarray([0.05, 1.2, 0.05, 0.25, 0.25], dtype=float)
    proposal_std = proposal_scale * prior_std

    noise_scale = max(float(base.rmse), 1e-3)
    rng = np.random.default_rng(seed)
    total_steps = burn_in + n_samples

    def _run_chain(step_scale: float) -> Tuple[int, List[np.ndarray], List[float]]:
        chain_current = current.copy()
        chain_logp = _log_posterior(
            chain_current,
            options=options,
            targets=targets,
            spot=spot,
            rate=rate,
            use_iv=use_iv,
            prior_mean=prior_mean,
            prior_std=prior_std,
            noise_scale=noise_scale,
        )
        chain_accepted = 0
        chain_samples: List[np.ndarray] = []
        chain_trace: List[float] = []

        for i in range(total_steps):
            proposal = chain_current + step_scale * proposal_std * rng.standard_normal(chain_current.shape[0])
            proposal = np.clip(proposal, _PARAM_BOUNDS[:, 0], _PARAM_BOUNDS[:, 1])
            logp_prop = _log_posterior(
                proposal,
                options=options,
                targets=targets,
                spot=spot,
                rate=rate,
                use_iv=use_iv,
                prior_mean=prior_mean,
                prior_std=prior_std,
                noise_scale=noise_scale,
            )
            if np.isfinite(logp_prop):
                accept = np.log(rng.random()) < (logp_prop - chain_logp)
            else:
                accept = False

            if accept:
                chain_current = proposal
                chain_logp = logp_prop
                chain_accepted += 1

            if i >= burn_in:
                chain_samples.append(chain_current.copy())
                chain_trace.append(float(chain_logp))
        return chain_accepted, chain_samples, chain_trace

    accepted, kept_samples, trace = _run_chain(step_scale=1.0)
    if accepted == 0:
        for scale in (0.25, 0.08, 0.02):
            accepted, kept_samples, trace = _run_chain(step_scale=scale)
            if accepted > 0:
                break

    if not kept_samples:
        raise RuntimeError("No posterior samples were retained")

    arr = np.asarray(kept_samples, dtype=float)
    posterior_uq: Dict[str, ParameterUQ] = {}
    posterior_samples: Dict[str, List[float]] = {}
    for j, name in enumerate(_PARAM_ORDER):
        s = arr[:, j]
        ci_low, ci_high = np.quantile(s, [0.025, 0.975])
        posterior_uq[name] = ParameterUQ(
            name=name,
            mean=float(np.mean(s)),
            std=float(np.std(s, ddof=1)) if s.size > 1 else 0.0,
            ci_low=float(ci_low),
            ci_high=float(ci_high),
        )
        posterior_samples[name] = [float(v) for v in s.tolist()]

    trace_arr = np.asarray(trace, dtype=float)
    lp_low, lp_high = np.quantile(trace_arr, [0.025, 0.975])
    log_posterior_ci = ConfidenceInterval(
        low=float(lp_low),
        high=float(lp_high),
        mean=float(np.mean(trace_arr)),
        std=float(np.std(trace_arr, ddof=1)) if trace_arr.size > 1 else 0.0,
        n=int(trace_arr.size),
    )

    return BayesianCalibrationResult(
        base=base,
        acceptance_rate=float(accepted / max(total_steps, 1)),
        n_samples=n_samples,
        burn_in=burn_in,
        parameter_uq=posterior_uq,
        posterior_samples=posterior_samples,
        log_posterior_ci=log_posterior_ci,
    )


def bayesian_result_to_dict(result: BayesianCalibrationResult) -> Dict[str, object]:
    """Serialize Bayesian calibration output."""

    return {
        "base": {
            "success": result.base.success,
            "parameters": result.base.parameters,
            "objective_value": result.base.objective_value,
            "rmse": result.base.rmse,
            "num_iterations": result.base.num_iterations,
            "calibration_time": result.base.calibration_time,
            "message": result.base.message,
        },
        "acceptance_rate": result.acceptance_rate,
        "n_samples": result.n_samples,
        "burn_in": result.burn_in,
        "parameter_uq": {k: asdict(v) for k, v in result.parameter_uq.items()},
        "posterior_samples": {k: list(v) for k, v in result.posterior_samples.items()},
        "log_posterior_ci": asdict(result.log_posterior_ci),
    }
