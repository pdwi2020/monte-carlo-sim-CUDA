"""Multilevel Monte Carlo prototype for GBM options."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np

from mc_pricer import HestonParams


@dataclass
class MLMCLevelStats:
    """Level statistics for MLMC estimator diagnostics."""

    level: int
    n_samples: int
    mean: float
    variance: float
    cost_per_sample: int


@dataclass
class MLMCResult:
    """MLMC price estimate and diagnostics."""

    price: float
    std_error: float
    runtime_seconds: float
    total_cost_units: int
    levels: List[MLMCLevelStats]


@dataclass
class MCResult:
    """Single-level Monte Carlo estimate."""

    price: float
    std_error: float
    runtime_seconds: float
    num_paths: int
    num_steps: int


def _payoff_from_terminal(terminal: np.ndarray, strike: float, is_call: bool) -> np.ndarray:
    if is_call:
        return np.maximum(terminal - strike, 0.0)
    return np.maximum(strike - terminal, 0.0)


def _simulate_gbm_from_increments(
    s0: float,
    r: float,
    sigma: float,
    dt: float,
    increments: np.ndarray,
) -> np.ndarray:
    """Simulate GBM path from Brownian increments."""

    log_s = np.log(s0) + np.cumsum((r - 0.5 * sigma ** 2) * dt + sigma * increments, axis=1)
    out = np.empty((increments.shape[0], increments.shape[1] + 1), dtype=float)
    out[:, 0] = s0
    out[:, 1:] = np.exp(log_s)
    return out


def _coupled_level_differences(
    *,
    n_samples: int,
    level: int,
    base_steps: int,
    s0: float,
    strike: float,
    r: float,
    sigma: float,
    maturity: float,
    is_call: bool,
    payoff_style: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Coupled payoff differences Y_l = P_l - P_{l-1}."""

    fine_steps = base_steps * (2 ** level)
    dt_f = maturity / fine_steps
    increments_f = np.sqrt(dt_f) * rng.standard_normal((n_samples, fine_steps))

    fine_paths = _simulate_gbm_from_increments(s0, r, sigma, dt_f, increments_f)

    if payoff_style == "european":
        payoff_f = _payoff_from_terminal(fine_paths[:, -1], strike, is_call)
    elif payoff_style == "asian":
        payoff_f = _payoff_from_terminal(np.mean(fine_paths[:, 1:], axis=1), strike, is_call)
    else:
        raise ValueError("payoff_style must be 'european' or 'asian'")

    if level == 0:
        return np.exp(-r * maturity) * payoff_f

    coarse_steps = fine_steps // 2
    dt_c = maturity / coarse_steps
    increments_c = increments_f.reshape(n_samples, coarse_steps, 2).sum(axis=2)
    coarse_paths = _simulate_gbm_from_increments(s0, r, sigma, dt_c, increments_c)

    if payoff_style == "european":
        payoff_c = _payoff_from_terminal(coarse_paths[:, -1], strike, is_call)
    else:
        payoff_c = _payoff_from_terminal(np.mean(coarse_paths[:, 1:], axis=1), strike, is_call)

    discount = np.exp(-r * maturity)
    return discount * (payoff_f - payoff_c)


def mlmc_price_gbm_option(
    *,
    s0: float,
    strike: float,
    r: float,
    sigma: float,
    maturity: float,
    is_call: bool = True,
    payoff_style: str = "european",
    max_level: int = 5,
    base_steps: int = 4,
    min_level_samples: int = 2000,
    seed: Optional[int] = 42,
) -> MLMCResult:
    """Price GBM option using a practical MLMC estimator."""

    import time

    start = time.time()
    rng = np.random.default_rng(seed)

    level_stats: List[MLMCLevelStats] = []
    level_means = []
    level_vars = []
    total_cost = 0

    for level in range(max_level + 1):
        n_l = max(min_level_samples // (2 ** level), 64)
        diffs = _coupled_level_differences(
            n_samples=n_l,
            level=level,
            base_steps=base_steps,
            s0=s0,
            strike=strike,
            r=r,
            sigma=sigma,
            maturity=maturity,
            is_call=is_call,
            payoff_style=payoff_style,
            rng=rng,
        )

        mean_l = float(np.mean(diffs))
        var_l = float(np.var(diffs, ddof=1)) if diffs.size > 1 else 0.0
        cost_per_sample = base_steps * (2 ** level)
        total_cost += n_l * cost_per_sample

        level_stats.append(
            MLMCLevelStats(
                level=level,
                n_samples=n_l,
                mean=mean_l,
                variance=var_l,
                cost_per_sample=cost_per_sample,
            )
        )
        level_means.append(mean_l)
        level_vars.append(var_l / n_l)

    price = float(np.sum(level_means))
    std_error = float(np.sqrt(np.sum(level_vars)))

    return MLMCResult(
        price=price,
        std_error=std_error,
        runtime_seconds=float(time.time() - start),
        total_cost_units=int(total_cost),
        levels=level_stats,
    )


def mc_price_gbm_option(
    *,
    s0: float,
    strike: float,
    r: float,
    sigma: float,
    maturity: float,
    is_call: bool = True,
    payoff_style: str = "european",
    num_steps: int = 128,
    target_std_error: Optional[float] = None,
    max_paths: int = 300000,
    seed: Optional[int] = 42,
) -> MCResult:
    """Standard MC baseline with optional dynamic path count for target standard error."""

    import time

    start = time.time()
    rng = np.random.default_rng(seed)
    dt = maturity / num_steps

    num_paths = 20000
    if target_std_error is not None:
        pilot_paths = 10000
        inc = np.sqrt(dt) * rng.standard_normal((pilot_paths, num_steps))
        paths = _simulate_gbm_from_increments(s0, r, sigma, dt, inc)
        if payoff_style == "european":
            pay = _payoff_from_terminal(paths[:, -1], strike, is_call)
        else:
            pay = _payoff_from_terminal(np.mean(paths[:, 1:], axis=1), strike, is_call)
        pay = np.exp(-r * maturity) * pay
        var = float(np.var(pay, ddof=1))
        num_paths = int(np.ceil(var / max(target_std_error, 1e-8) ** 2))
        num_paths = int(np.clip(num_paths, 5000, max_paths))

    inc = np.sqrt(dt) * rng.standard_normal((num_paths, num_steps))
    paths = _simulate_gbm_from_increments(s0, r, sigma, dt, inc)

    if payoff_style == "european":
        payoffs = _payoff_from_terminal(paths[:, -1], strike, is_call)
    elif payoff_style == "asian":
        payoffs = _payoff_from_terminal(np.mean(paths[:, 1:], axis=1), strike, is_call)
    else:
        raise ValueError("payoff_style must be 'european' or 'asian'")

    discounted = np.exp(-r * maturity) * payoffs
    return MCResult(
        price=float(np.mean(discounted)),
        std_error=float(np.std(discounted, ddof=1) / np.sqrt(num_paths)),
        runtime_seconds=float(time.time() - start),
        num_paths=num_paths,
        num_steps=num_steps,
    )


def compare_mlmc_vs_mc(
    *,
    s0: float = 100.0,
    strike: float = 100.0,
    r: float = 0.03,
    sigma: float = 0.2,
    maturity: float = 1.0,
    is_call: bool = True,
    payoff_style: str = "european",
    seed: int = 123,
) -> Dict[str, object]:
    """Run MLMC and matched-error MC and return comparison diagnostics."""

    mlmc = mlmc_price_gbm_option(
        s0=s0,
        strike=strike,
        r=r,
        sigma=sigma,
        maturity=maturity,
        is_call=is_call,
        payoff_style=payoff_style,
        seed=seed,
    )

    mc = mc_price_gbm_option(
        s0=s0,
        strike=strike,
        r=r,
        sigma=sigma,
        maturity=maturity,
        is_call=is_call,
        payoff_style=payoff_style,
        num_steps=4 * (2 ** 5),
        target_std_error=max(mlmc.std_error, 1e-4),
        seed=seed + 1,
    )

    speedup = mc.runtime_seconds / max(mlmc.runtime_seconds, 1e-12)

    return {
        "mlmc": {
            "price": mlmc.price,
            "std_error": mlmc.std_error,
            "runtime_seconds": mlmc.runtime_seconds,
            "total_cost_units": mlmc.total_cost_units,
            "levels": [asdict(level) for level in mlmc.levels],
        },
        "mc": asdict(mc),
        "runtime_speedup_mc_over_mlmc": speedup,
        "abs_price_gap": abs(mlmc.price - mc.price),
    }


def _simulate_heston_payoff_from_normals(
    *,
    s0: float,
    strike: float,
    r: float,
    maturity: float,
    heston: HestonParams,
    z1: np.ndarray,
    z2: np.ndarray,
    is_call: bool,
    payoff_style: str,
) -> np.ndarray:
    """Simulate discounted Heston payoffs from coupled normal draws."""

    n_paths, n_steps = z1.shape
    dt = maturity / n_steps
    sqrt_dt = np.sqrt(dt)

    s = np.full(n_paths, s0, dtype=float)
    v = np.full(n_paths, heston.v0, dtype=float)

    if payoff_style == "asian":
        running = np.zeros(n_paths, dtype=float)
    elif payoff_style == "european":
        running = None
    else:
        raise ValueError("payoff_style must be 'european' or 'asian'")

    rho_comp = np.sqrt(max(1.0 - heston.rho ** 2, 0.0))

    for t in range(n_steps):
        z1_t = z1[:, t]
        z2_t = z2[:, t]

        v_pos = np.maximum(v, 0.0)
        dWv = sqrt_dt * z1_t
        dWs = sqrt_dt * (heston.rho * z1_t + rho_comp * z2_t)

        # Full truncation Euler for variance process.
        v = np.maximum(
            v + heston.kappa * (heston.theta - v_pos) * dt + heston.xi * np.sqrt(v_pos) * dWv,
            0.0,
        )
        s = s * np.exp((r - 0.5 * v_pos) * dt + np.sqrt(v_pos) * dWs)

        if running is not None:
            running += s

    if running is None:
        terminal = s
    else:
        terminal = running / n_steps

    payoff = _payoff_from_terminal(terminal, strike, is_call)
    return np.exp(-r * maturity) * payoff


def _coupled_heston_level_differences(
    *,
    n_samples: int,
    level: int,
    base_steps: int,
    s0: float,
    strike: float,
    r: float,
    maturity: float,
    heston: HestonParams,
    is_call: bool,
    payoff_style: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Coupled MLMC level differences for Heston model."""

    fine_steps = base_steps * (2 ** level)
    z1_f = rng.standard_normal((n_samples, fine_steps))
    z2_f = rng.standard_normal((n_samples, fine_steps))

    payoff_f = _simulate_heston_payoff_from_normals(
        s0=s0,
        strike=strike,
        r=r,
        maturity=maturity,
        heston=heston,
        z1=z1_f,
        z2=z2_f,
        is_call=is_call,
        payoff_style=payoff_style,
    )

    if level == 0:
        return payoff_f

    coarse_steps = fine_steps // 2
    z1_c = z1_f.reshape(n_samples, coarse_steps, 2).sum(axis=2) / np.sqrt(2.0)
    z2_c = z2_f.reshape(n_samples, coarse_steps, 2).sum(axis=2) / np.sqrt(2.0)

    payoff_c = _simulate_heston_payoff_from_normals(
        s0=s0,
        strike=strike,
        r=r,
        maturity=maturity,
        heston=heston,
        z1=z1_c,
        z2=z2_c,
        is_call=is_call,
        payoff_style=payoff_style,
    )
    return payoff_f - payoff_c


def mlmc_price_heston_option(
    *,
    s0: float,
    strike: float,
    r: float,
    maturity: float,
    heston: HestonParams,
    is_call: bool = True,
    payoff_style: str = "european",
    max_level: int = 5,
    base_steps: int = 4,
    min_level_samples: int = 2000,
    seed: Optional[int] = 42,
) -> MLMCResult:
    """Price Heston option with a coupled MLMC estimator."""

    import time

    start = time.time()
    rng = np.random.default_rng(seed)

    level_stats: List[MLMCLevelStats] = []
    level_means = []
    level_vars = []
    total_cost = 0

    for level in range(max_level + 1):
        n_l = max(min_level_samples // (2 ** level), 64)
        diffs = _coupled_heston_level_differences(
            n_samples=n_l,
            level=level,
            base_steps=base_steps,
            s0=s0,
            strike=strike,
            r=r,
            maturity=maturity,
            heston=heston,
            is_call=is_call,
            payoff_style=payoff_style,
            rng=rng,
        )

        mean_l = float(np.mean(diffs))
        var_l = float(np.var(diffs, ddof=1)) if diffs.size > 1 else 0.0
        cost_per_sample = base_steps * (2 ** level)
        total_cost += n_l * cost_per_sample

        level_stats.append(
            MLMCLevelStats(
                level=level,
                n_samples=n_l,
                mean=mean_l,
                variance=var_l,
                cost_per_sample=cost_per_sample,
            )
        )
        level_means.append(mean_l)
        level_vars.append(var_l / n_l)

    return MLMCResult(
        price=float(np.sum(level_means)),
        std_error=float(np.sqrt(np.sum(level_vars))),
        runtime_seconds=float(time.time() - start),
        total_cost_units=int(total_cost),
        levels=level_stats,
    )


def mc_price_heston_option(
    *,
    s0: float,
    strike: float,
    r: float,
    maturity: float,
    heston: HestonParams,
    is_call: bool = True,
    payoff_style: str = "european",
    num_steps: int = 128,
    target_std_error: Optional[float] = None,
    max_paths: int = 300000,
    seed: Optional[int] = 42,
) -> MCResult:
    """Single-level MC baseline for Heston options."""

    import time

    start = time.time()
    rng = np.random.default_rng(seed)

    num_paths = 20000
    if target_std_error is not None:
        pilot_paths = 10000
        z1 = rng.standard_normal((pilot_paths, num_steps))
        z2 = rng.standard_normal((pilot_paths, num_steps))
        pay = _simulate_heston_payoff_from_normals(
            s0=s0,
            strike=strike,
            r=r,
            maturity=maturity,
            heston=heston,
            z1=z1,
            z2=z2,
            is_call=is_call,
            payoff_style=payoff_style,
        )
        var = float(np.var(pay, ddof=1))
        num_paths = int(np.ceil(var / max(target_std_error, 1e-8) ** 2))
        num_paths = int(np.clip(num_paths, 5000, max_paths))

    z1 = rng.standard_normal((num_paths, num_steps))
    z2 = rng.standard_normal((num_paths, num_steps))
    discounted = _simulate_heston_payoff_from_normals(
        s0=s0,
        strike=strike,
        r=r,
        maturity=maturity,
        heston=heston,
        z1=z1,
        z2=z2,
        is_call=is_call,
        payoff_style=payoff_style,
    )
    return MCResult(
        price=float(np.mean(discounted)),
        std_error=float(np.std(discounted, ddof=1) / np.sqrt(num_paths)),
        runtime_seconds=float(time.time() - start),
        num_paths=num_paths,
        num_steps=num_steps,
    )


def compare_mlmc_heston_vs_mc(
    *,
    s0: float = 100.0,
    strike: float = 100.0,
    r: float = 0.03,
    maturity: float = 1.0,
    heston: Optional[HestonParams] = None,
    is_call: bool = True,
    payoff_style: str = "european",
    seed: int = 321,
) -> Dict[str, object]:
    """Compare Heston MLMC against matched-error standard MC."""

    heston_params = heston or HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.4, rho=-0.6)
    mlmc = mlmc_price_heston_option(
        s0=s0,
        strike=strike,
        r=r,
        maturity=maturity,
        heston=heston_params,
        is_call=is_call,
        payoff_style=payoff_style,
        seed=seed,
    )
    mc = mc_price_heston_option(
        s0=s0,
        strike=strike,
        r=r,
        maturity=maturity,
        heston=heston_params,
        is_call=is_call,
        payoff_style=payoff_style,
        num_steps=4 * (2 ** 5),
        target_std_error=max(mlmc.std_error, 1e-4),
        seed=seed + 1,
    )

    speedup = mc.runtime_seconds / max(mlmc.runtime_seconds, 1e-12)
    return {
        "mlmc": {
            "price": mlmc.price,
            "std_error": mlmc.std_error,
            "runtime_seconds": mlmc.runtime_seconds,
            "total_cost_units": mlmc.total_cost_units,
            "levels": [asdict(level) for level in mlmc.levels],
        },
        "mc": asdict(mc),
        "runtime_speedup_mc_over_mlmc": speedup,
        "abs_price_gap": abs(mlmc.price - mc.price),
    }
