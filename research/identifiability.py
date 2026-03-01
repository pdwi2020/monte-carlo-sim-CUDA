"""Parameter identifiability and posterior-geometry diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from calibration import MarketOption, calibrate_heston, heston_call_price, implied_volatility
from .calibration_uq import bayesian_heston_calibration

_PARAM_ORDER = ["v0", "kappa", "theta", "xi", "rho"]
_PARAM_BOUNDS = {
    "v0": (0.001, 1.0),
    "kappa": (0.01, 10.0),
    "theta": (0.001, 1.0),
    "xi": (0.01, 2.0),
    "rho": (-0.99, 0.99),
}


@dataclass
class ProfileSlice:
    """One-dimensional profile-likelihood style slice."""

    parameter: str
    grid: List[float]
    rmse: List[float]
    argmin_value: float
    min_rmse: float


@dataclass
class PosteriorGeometry:
    """Posterior geometry diagnostics from MCMC samples."""

    parameter_order: List[str]
    corr_matrix: List[List[float]]
    covariance_matrix: List[List[float]]
    condition_number: float
    ridge_score: float


@dataclass
class IdentifiabilityResult:
    """Combined identifiability output payload."""

    base_parameters: Dict[str, float]
    base_rmse: float
    profile_slices: List[ProfileSlice]
    posterior_geometry: PosteriorGeometry
    posterior_acceptance_rate: float
    posterior_sample_size: int


def _prepare_targets(options: Sequence[MarketOption], *, use_iv: bool) -> Tuple[List[MarketOption], np.ndarray]:
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
    if len(filtered) < 5:
        raise ValueError("Need at least 5 observations for identifiability analysis")
    return filtered, np.asarray(targets, dtype=float)


def _model_values(
    options: Sequence[MarketOption],
    *,
    spot: float,
    rate: float,
    params: Dict[str, float],
    use_iv: bool,
) -> np.ndarray:
    out = []
    for opt in options:
        call = heston_call_price(
            spot,
            opt.strike,
            rate,
            opt.maturity,
            params["v0"],
            params["kappa"],
            params["theta"],
            params["xi"],
            params["rho"],
        )
        if opt.option_type == "put":
            price = call - spot + opt.strike * np.exp(-rate * opt.maturity)
        else:
            price = call

        if use_iv and opt.market_iv is not None:
            try:
                val = implied_volatility(price, spot, opt.strike, rate, opt.maturity, opt.option_type)
            except Exception:
                val = np.sqrt(max(params["v0"], 1e-8))
        else:
            val = price
        out.append(float(val))
    return np.asarray(out, dtype=float)


def _rmse(
    options: Sequence[MarketOption],
    targets: np.ndarray,
    *,
    spot: float,
    rate: float,
    params: Dict[str, float],
    use_iv: bool,
) -> float:
    vals = _model_values(options, spot=spot, rate=rate, params=params, use_iv=use_iv)
    return float(np.sqrt(np.mean((vals - targets) ** 2)))


def analyze_heston_identifiability(
    market_options: Sequence[MarketOption],
    *,
    spot: float,
    rate: float,
    use_iv: bool = True,
    max_iter: int = 220,
    profile_points: int = 11,
    bayesian_samples: int = 140,
    bayesian_burn_in: int = 90,
    seed: int = 42,
) -> IdentifiabilityResult:
    """Run profile slices + posterior geometry diagnostics."""

    options, targets = _prepare_targets(list(market_options), use_iv=use_iv)
    base = calibrate_heston(options, spot=spot, rate=rate, use_iv=use_iv, max_iter=max_iter)
    base_params = {k: float(v) for k, v in base.parameters.items()}
    base_rmse = _rmse(options, targets, spot=spot, rate=rate, params=base_params, use_iv=use_iv)

    slices: List[ProfileSlice] = []
    for name in _PARAM_ORDER:
        lo, hi = _PARAM_BOUNDS[name]
        center = base_params[name]
        width = 0.35 * max(abs(center), 1e-3)
        grid = np.linspace(max(lo, center - width), min(hi, center + width), profile_points)
        rmses = []
        for x in grid:
            p = dict(base_params)
            p[name] = float(x)
            rmses.append(_rmse(options, targets, spot=spot, rate=rate, params=p, use_iv=use_iv))
        j = int(np.argmin(rmses))
        slices.append(
            ProfileSlice(
                parameter=name,
                grid=[float(v) for v in grid.tolist()],
                rmse=[float(v) for v in rmses],
                argmin_value=float(grid[j]),
                min_rmse=float(rmses[j]),
            )
        )

    bayes = bayesian_heston_calibration(
        options,
        spot=spot,
        rate=rate,
        use_iv=use_iv,
        max_iter=max(80, max_iter // 2),
        n_samples=bayesian_samples,
        burn_in=bayesian_burn_in,
        proposal_scale=0.12,
        seed=seed + 1,
    )
    sample_mat = np.column_stack([np.asarray(bayes.posterior_samples[k], dtype=float) for k in _PARAM_ORDER])
    cov = np.cov(sample_mat, rowvar=False)
    corr = np.corrcoef(sample_mat, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    min_e = float(np.clip(np.min(eigvals), 1e-12, None))
    max_e = float(np.max(eigvals))
    cond = float(max_e / min_e)
    ridge = float(max_e / max(np.sum(eigvals), 1e-12))

    geom = PosteriorGeometry(
        parameter_order=list(_PARAM_ORDER),
        corr_matrix=[[float(x) for x in row] for row in corr.tolist()],
        covariance_matrix=[[float(x) for x in row] for row in cov.tolist()],
        condition_number=cond,
        ridge_score=ridge,
    )

    return IdentifiabilityResult(
        base_parameters=base_params,
        base_rmse=base_rmse,
        profile_slices=slices,
        posterior_geometry=geom,
        posterior_acceptance_rate=float(bayes.acceptance_rate),
        posterior_sample_size=int(bayesian_samples),
    )


def identifiability_to_dict(result: IdentifiabilityResult) -> Dict[str, object]:
    """Serialize identifiability diagnostics."""

    return {
        "base_parameters": dict(result.base_parameters),
        "base_rmse": result.base_rmse,
        "profile_slices": [asdict(s) for s in result.profile_slices],
        "posterior_geometry": asdict(result.posterior_geometry),
        "posterior_acceptance_rate": result.posterior_acceptance_rate,
        "posterior_sample_size": result.posterior_sample_size,
    }

