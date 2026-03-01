"""SVI/SSVI-inspired static-arbitrage cleaning for implied-vol surfaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from calibration import MarketOption

try:
    from scipy.optimize import minimize

    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover
    minimize = None
    SCIPY_AVAILABLE = False


@dataclass
class SVIParams:
    """Raw-SVI parameters for one maturity slice."""

    a: float
    b: float
    rho: float
    m: float
    sigma: float


@dataclass
class SVIArbitrageReport:
    """Diagnostics from static-arbitrage checks on fitted SVI slices."""

    num_maturities: int
    min_total_variance: float
    butterfly_violations: int
    calendar_violations: int
    is_static_arbitrage_free: bool


@dataclass
class SVICleaningResult:
    """Surface-cleaning payload based on SVI fitting and arbitrage checks."""

    params_by_maturity: Dict[str, SVIParams]
    arbitrage_report: SVIArbitrageReport
    cleaned_options: List[MarketOption]


def svi_total_variance(k: np.ndarray | float, p: SVIParams) -> np.ndarray:
    """Raw SVI total variance w(k)."""

    k_arr = np.asarray(k, dtype=float)
    inside = (k_arr - p.m) ** 2 + p.sigma**2
    return p.a + p.b * (p.rho * (k_arr - p.m) + np.sqrt(np.maximum(inside, 1e-12)))


def _fit_svi_slice(log_m: np.ndarray, total_var: np.ndarray, *, seed: int = 42) -> SVIParams:
    if log_m.size < 4:
        raise ValueError("Need at least 4 points to fit SVI slice")
    x = np.asarray(log_m, dtype=float)
    y = np.asarray(total_var, dtype=float)

    # Stable default guess.
    y_mean = float(np.mean(y))
    x_med = float(np.median(x))
    p0 = np.asarray([max(1e-6, 0.5 * y_mean), 0.15, 0.0, x_med, 0.2], dtype=float)
    bounds = [
        (1e-8, 5.0),   # a
        (1e-5, 5.0),   # b
        (-0.999, 0.999),  # rho
        (-3.0, 3.0),   # m
        (1e-4, 3.0),   # sigma
    ]

    def obj(theta: np.ndarray) -> float:
        p = SVIParams(*[float(v) for v in theta])
        w = svi_total_variance(x, p)
        # Penalize obviously invalid total variances.
        penalty = 1e3 * float(np.mean(np.maximum(-w, 0.0) ** 2))
        return float(np.mean((w - y) ** 2) + penalty)

    if SCIPY_AVAILABLE:
        out = minimize(
            obj,
            x0=p0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 300},
        )
        theta = out.x if out.success else p0
    else:
        # Lightweight fallback random search around p0.
        rng = np.random.default_rng(seed)
        theta = p0.copy()
        best = obj(theta)
        for _ in range(400):
            cand = theta + np.asarray([0.05, 0.08, 0.12, 0.20, 0.10]) * rng.normal(size=5)
            for i, (lo, hi) in enumerate(bounds):
                cand[i] = float(np.clip(cand[i], lo, hi))
            val = obj(cand)
            if val < best:
                best = val
                theta = cand

    return SVIParams(*[float(v) for v in theta])


def fit_svi_surface(
    options: Sequence[MarketOption],
    *,
    spot: float,
) -> Dict[float, SVIParams]:
    """Fit SVI parameters independently by maturity from option-IV targets."""

    by_mat: Dict[float, List[Tuple[float, float]]] = {}
    for opt in options:
        if opt.market_iv is None:
            continue
        if opt.maturity <= 0:
            continue
        k = float(np.log(max(opt.strike, 1e-12) / max(spot, 1e-12)))
        w = float((opt.market_iv**2) * opt.maturity)
        by_mat.setdefault(float(opt.maturity), []).append((k, w))
    if not by_mat:
        raise ValueError("No options with implied volatility for SVI fit")

    out: Dict[float, SVIParams] = {}
    for maturity, rows in sorted(by_mat.items(), key=lambda kv: kv[0]):
        arr = np.asarray(rows, dtype=float)
        out[maturity] = _fit_svi_slice(arr[:, 0], arr[:, 1], seed=int(1000 * maturity))
    return out


def evaluate_static_arbitrage(
    params_by_maturity: Dict[float, SVIParams],
    *,
    k_grid: Optional[np.ndarray] = None,
    tol: float = 1e-6,
) -> SVIArbitrageReport:
    """Run approximate static arbitrage checks across fitted SVI slices."""

    maturities = sorted(params_by_maturity.keys())
    if not maturities:
        raise ValueError("No maturity slices provided")
    k = np.linspace(-0.6, 0.6, 41) if k_grid is None else np.asarray(k_grid, dtype=float)

    min_w = float("inf")
    butterfly_viol = 0
    prev_w: Optional[np.ndarray] = None
    calendar_viol = 0

    for t in maturities:
        p = params_by_maturity[t]
        w = svi_total_variance(k, p)
        min_w = min(min_w, float(np.min(w)))

        # Approximate butterfly condition via convexity of total variance.
        d2 = np.diff(w, n=2)
        butterfly_viol += int(np.sum(d2 < -tol))

        if prev_w is not None:
            calendar_viol += int(np.sum(w < prev_w - tol))
        prev_w = w

    return SVIArbitrageReport(
        num_maturities=len(maturities),
        min_total_variance=float(min_w),
        butterfly_violations=int(butterfly_viol),
        calendar_violations=int(calendar_viol),
        is_static_arbitrage_free=(min_w >= -tol and butterfly_viol == 0 and calendar_viol == 0),
    )


def clean_surface_with_svi(
    options: Sequence[MarketOption],
    *,
    spot: float,
) -> SVICleaningResult:
    """Fit SVI and replace option market IV with arbitrage-cleaned values."""

    params = fit_svi_surface(options, spot=spot)
    report = evaluate_static_arbitrage(params)

    cleaned: List[MarketOption] = []
    for opt in options:
        row = MarketOption(
            strike=opt.strike,
            maturity=opt.maturity,
            market_price=opt.market_price,
            market_iv=opt.market_iv,
            option_type=opt.option_type,
            bid=opt.bid,
            ask=opt.ask,
        )
        p = params.get(float(opt.maturity))
        if p is not None and opt.maturity > 0:
            k = float(np.log(max(opt.strike, 1e-12) / max(spot, 1e-12)))
            w = float(max(svi_total_variance(k, p), 1e-10))
            row.market_iv = float(np.sqrt(w / opt.maturity))
        cleaned.append(row)

    return SVICleaningResult(
        params_by_maturity={str(k): v for k, v in params.items()},
        arbitrage_report=report,
        cleaned_options=cleaned,
    )


def svi_cleaning_to_dict(result: SVICleaningResult) -> Dict[str, object]:
    """Serialize SVI cleaning output."""

    return {
        "params_by_maturity": {k: asdict(v) for k, v in result.params_by_maturity.items()},
        "arbitrage_report": asdict(result.arbitrage_report),
        "num_cleaned_options": len(result.cleaned_options),
    }
