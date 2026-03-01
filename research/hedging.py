"""Hedging robustness experiments under model misspecification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from mc_pricer import HestonParams

try:
    from scipy.stats import chi2, norm

    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    chi2 = None
    norm = None
    SCIPY_AVAILABLE = False


@dataclass
class HedgingBacktestResult:
    """Distributional summary for hedging PnL."""

    mean_pnl: float
    std_pnl: float
    rmse_pnl: float
    var95_loss: float
    cvar95_loss: float
    num_paths: int
    num_steps: int


@dataclass
class VaRBacktestResult:
    """VaR exception backtesting summary."""

    alpha: float
    n_obs: int
    n_exceptions: int
    exception_rate: float
    lr_uc: float
    p_value_uc: float
    lr_ind: float
    p_value_ind: float
    lr_cc: float
    p_value_cc: float


@dataclass
class ExecutionModel:
    """Execution frictions for realistic hedging fills."""

    bid_ask_bps: float = 2.0
    slippage_bps: float = 1.0
    impact_bps_per_unit: float = 0.10
    fill_probability: float = 1.0


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    if SCIPY_AVAILABLE:
        return norm.cdf(x)

    from math import erf

    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(erf)(x / np.sqrt(2.0)))


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    if SCIPY_AVAILABLE:
        return norm.pdf(x)
    return np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)


def _chi2_survival(x: float, df: int) -> float:
    if x <= 0:
        return 1.0
    if SCIPY_AVAILABLE:
        return float(chi2.sf(x, df))
    return float(np.exp(-0.5 * x))


def _safe_log(p: float) -> float:
    return float(np.log(np.clip(p, 1e-12, 1.0 - 1e-12)))


def _execute_trade(
    desired_trade: np.ndarray,
    mid_price: np.ndarray,
    *,
    execution_model: Optional[ExecutionModel],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert desired trades into executed trades and cashflow."""

    if execution_model is None:
        executed = desired_trade
        cashflow = executed * mid_price
        return executed, cashflow

    fill_mask = rng.random(size=desired_trade.shape[0]) < float(np.clip(execution_model.fill_probability, 0.0, 1.0))
    executed = desired_trade * fill_mask.astype(float)
    abs_trade = np.abs(executed)
    sign = np.sign(executed)

    half_spread = 0.5 * execution_model.bid_ask_bps * 1e-4 * mid_price
    slippage = execution_model.slippage_bps * 1e-4 * mid_price
    impact = execution_model.impact_bps_per_unit * 1e-4 * mid_price * abs_trade
    execution_price = mid_price + sign * (half_spread + slippage + impact)
    cashflow = executed * execution_price
    return executed, cashflow


def _bs_d1(spot: np.ndarray, strike: float, rate: float, sigma: float, tau: float) -> np.ndarray:
    sigma = max(sigma, 1e-8)
    tau = max(tau, 1e-10)
    return (np.log(np.maximum(spot, 1e-12) / strike) + (rate + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))


def _bs_call_price(spot: np.ndarray, strike: float, rate: float, sigma: float, tau: float) -> np.ndarray:
    if tau <= 0:
        return np.maximum(spot - strike, 0.0)

    d1 = _bs_d1(spot, strike, rate, sigma, tau)
    d2 = d1 - sigma * np.sqrt(tau)
    return spot * _norm_cdf(d1) - strike * np.exp(-rate * tau) * _norm_cdf(d2)


def _bs_delta(spot: np.ndarray, strike: float, rate: float, sigma: float, tau: float, is_call: bool) -> np.ndarray:
    if tau <= 0:
        if is_call:
            return (spot > strike).astype(float)
        return -(spot < strike).astype(float)

    d1 = _bs_d1(spot, strike, rate, sigma, tau)
    call_delta = _norm_cdf(d1)
    if is_call:
        return call_delta
    return call_delta - 1.0


def _bs_vega(spot: np.ndarray, strike: float, rate: float, sigma: float, tau: float) -> np.ndarray:
    if tau <= 0:
        return np.zeros_like(spot, dtype=float)
    d1 = _bs_d1(spot, strike, rate, sigma, tau)
    return np.maximum(spot, 1e-12) * np.sqrt(tau) * _norm_pdf(d1)


def _option_price(
    spot: np.ndarray, strike: float, rate: float, sigma: float, tau: float, is_call: bool
) -> np.ndarray:
    call = _bs_call_price(spot, strike, rate, sigma, tau)
    if is_call:
        return call
    return call - spot + strike * np.exp(-rate * max(tau, 0.0))


def _option_payoff(spot: np.ndarray, strike: float, is_call: bool) -> np.ndarray:
    if is_call:
        return np.maximum(spot - strike, 0.0)
    return np.maximum(strike - spot, 0.0)


def _simulate_true_paths(
    *,
    true_model: str,
    s0: float,
    r: float,
    maturity: float,
    num_paths: int,
    num_steps: int,
    sigma_true: float,
    heston_true: Optional[HestonParams],
    seed: int,
) -> np.ndarray:
    """Simulate underlying paths under true market dynamics."""

    rng = np.random.default_rng(seed)
    dt = maturity / num_steps
    sqrt_dt = np.sqrt(dt)

    s = np.empty((num_paths, num_steps + 1), dtype=float)
    s[:, 0] = s0

    if true_model == "gbm":
        z = rng.standard_normal((num_paths, num_steps))
        drift = (r - 0.5 * sigma_true**2) * dt
        for t in range(num_steps):
            s[:, t + 1] = s[:, t] * np.exp(drift + sigma_true * sqrt_dt * z[:, t])
        return s

    if true_model == "heston":
        params = heston_true or HestonParams(v0=sigma_true**2, kappa=2.0, theta=sigma_true**2, xi=0.5, rho=-0.7)
        v = np.full(num_paths, params.v0, dtype=float)
        rho_comp = np.sqrt(max(1.0 - params.rho**2, 0.0))

        z1 = rng.standard_normal((num_paths, num_steps))
        z2 = rng.standard_normal((num_paths, num_steps))

        for t in range(num_steps):
            v_pos = np.maximum(v, 0.0)
            dWv = sqrt_dt * z1[:, t]
            dWs = sqrt_dt * (params.rho * z1[:, t] + rho_comp * z2[:, t])

            s[:, t + 1] = s[:, t] * np.exp((r - 0.5 * v_pos) * dt + np.sqrt(v_pos) * dWs)
            v = np.maximum(v + params.kappa * (params.theta - v_pos) * dt + params.xi * np.sqrt(v_pos) * dWv, 0.0)

        return s

    raise ValueError("true_model must be 'gbm' or 'heston'")


def _delta_hedge_pnl(
    *,
    true_model: str,
    s0: float,
    strike: float,
    r: float,
    maturity: float,
    sigma_hedger: float,
    sigma_true: float,
    heston_true: Optional[HestonParams],
    is_call: bool,
    num_paths: int,
    num_steps: int,
    rebalance_every: int,
    transaction_cost: float,
    execution_model: Optional[ExecutionModel],
    seed: int,
) -> np.ndarray:
    paths = _simulate_true_paths(
        true_model=true_model,
        s0=s0,
        r=r,
        maturity=maturity,
        num_paths=num_paths,
        num_steps=num_steps,
        sigma_true=sigma_true,
        heston_true=heston_true,
        seed=seed,
    )

    dt = maturity / num_steps
    s_t = paths[:, 0]

    option_0 = _option_price(s_t, strike, r, sigma_hedger, maturity, is_call=is_call)
    delta = _bs_delta(s_t, strike, r, sigma_hedger, maturity, is_call=is_call)
    cash = option_0 - delta * s_t
    rng_exec = np.random.default_rng(seed + 9000)

    for t in range(1, num_steps):
        tau = maturity - t * dt
        s_t = paths[:, t]
        cash *= np.exp(r * dt)

        if t % max(rebalance_every, 1) == 0:
            delta_new = _bs_delta(s_t, strike, r, sigma_hedger, tau, is_call=is_call)
            trade = delta_new - delta
            executed_trade, cashflow = _execute_trade(
                trade,
                s_t,
                execution_model=execution_model,
                rng=rng_exec,
            )
            cash -= cashflow
            if transaction_cost > 0:
                cash -= np.abs(executed_trade) * s_t * transaction_cost
            delta = delta + executed_trade

    s_t = paths[:, -1]
    cash *= np.exp(r * dt)
    portfolio_t = cash + delta * s_t
    payoff = _option_payoff(s_t, strike, is_call=is_call)
    return portfolio_t - payoff


def _delta_vega_hedge_pnl(
    *,
    true_model: str,
    s0: float,
    strike: float,
    r: float,
    maturity: float,
    sigma_hedger: float,
    sigma_true: float,
    heston_true: Optional[HestonParams],
    is_call: bool,
    hedge_strike: float,
    hedge_maturity: float,
    hedge_is_call: bool,
    num_paths: int,
    num_steps: int,
    rebalance_every: int,
    transaction_cost: float,
    execution_model: Optional[ExecutionModel],
    seed: int,
) -> np.ndarray:
    paths = _simulate_true_paths(
        true_model=true_model,
        s0=s0,
        r=r,
        maturity=maturity,
        num_paths=num_paths,
        num_steps=num_steps,
        sigma_true=sigma_true,
        heston_true=heston_true,
        seed=seed,
    )

    dt = maturity / num_steps
    s_t = paths[:, 0]

    target_0 = _option_price(s_t, strike, r, sigma_hedger, maturity, is_call=is_call)
    hedge_0 = _option_price(s_t, hedge_strike, r, sigma_hedger, hedge_maturity, is_call=hedge_is_call)

    target_delta = _bs_delta(s_t, strike, r, sigma_hedger, maturity, is_call=is_call)
    target_vega = _bs_vega(s_t, strike, r, sigma_hedger, maturity)
    hedge_delta = _bs_delta(s_t, hedge_strike, r, sigma_hedger, hedge_maturity, is_call=hedge_is_call)
    hedge_vega = _bs_vega(s_t, hedge_strike, r, sigma_hedger, hedge_maturity)

    n_vega = -target_vega / np.maximum(hedge_vega, 1e-8)
    n_stock = target_delta + n_vega * hedge_delta
    cash = target_0 - n_stock * s_t - n_vega * hedge_0
    rng_exec = np.random.default_rng(seed + 9100)

    for t in range(1, num_steps):
        tau_target = maturity - t * dt
        tau_hedge = max(hedge_maturity - t * dt, 0.0)
        s_t = paths[:, t]
        cash *= np.exp(r * dt)

        hedge_price_t = _option_price(s_t, hedge_strike, r, sigma_hedger, tau_hedge, is_call=hedge_is_call)
        if t % max(rebalance_every, 1) == 0:
            tgt_delta_new = _bs_delta(s_t, strike, r, sigma_hedger, tau_target, is_call=is_call)
            tgt_vega_new = _bs_vega(s_t, strike, r, sigma_hedger, tau_target)
            h_delta_new = _bs_delta(s_t, hedge_strike, r, sigma_hedger, tau_hedge, is_call=hedge_is_call)
            h_vega_new = _bs_vega(s_t, hedge_strike, r, sigma_hedger, tau_hedge)

            n_vega_new = -tgt_vega_new / np.maximum(h_vega_new, 1e-8)
            n_stock_new = tgt_delta_new + n_vega_new * h_delta_new

            trade_stock = n_stock_new - n_stock
            trade_hedge = n_vega_new - n_vega
            executed_stock, cash_stock = _execute_trade(
                trade_stock,
                s_t,
                execution_model=execution_model,
                rng=rng_exec,
            )
            executed_hedge, cash_hedge = _execute_trade(
                trade_hedge,
                np.maximum(hedge_price_t, 1e-12),
                execution_model=execution_model,
                rng=rng_exec,
            )
            cash -= cash_stock + cash_hedge
            if transaction_cost > 0:
                cash -= np.abs(executed_stock) * s_t * transaction_cost
                cash -= np.abs(executed_hedge) * np.maximum(hedge_price_t, 1e-12) * transaction_cost
            n_stock = n_stock + executed_stock
            n_vega = n_vega + executed_hedge

    s_t = paths[:, -1]
    cash *= np.exp(r * dt)
    tau_hedge_end = max(hedge_maturity - maturity, 0.0)
    if tau_hedge_end <= 0:
        hedge_value_end = _option_payoff(s_t, hedge_strike, is_call=hedge_is_call)
    else:
        hedge_value_end = _option_price(s_t, hedge_strike, r, sigma_hedger, tau_hedge_end, is_call=hedge_is_call)
    target_payoff = _option_payoff(s_t, strike, is_call=is_call)
    portfolio_t = cash + n_stock * s_t + n_vega * hedge_value_end
    return portfolio_t - target_payoff


def _summarize_pnl(pnl: np.ndarray, *, num_steps: int) -> HedgingBacktestResult:
    losses = -pnl
    var95 = float(np.quantile(losses, 0.95))
    tail = losses[losses >= var95]
    cvar95 = float(np.mean(tail)) if tail.size > 0 else var95
    return HedgingBacktestResult(
        mean_pnl=float(np.mean(pnl)),
        std_pnl=float(np.std(pnl, ddof=1)),
        rmse_pnl=float(np.sqrt(np.mean(pnl**2))),
        var95_loss=var95,
        cvar95_loss=cvar95,
        num_paths=int(pnl.size),
        num_steps=num_steps,
    )


def _var_backtest(losses: np.ndarray, *, var_threshold: float, alpha: float) -> VaRBacktestResult:
    exceptions = (losses > var_threshold).astype(int)
    n_obs = int(exceptions.size)
    n_exc = int(np.sum(exceptions))
    exc_rate = float(n_exc / max(n_obs, 1))

    p = float(alpha)
    pi_hat = float(np.clip(exc_rate, 1e-12, 1.0 - 1e-12))
    log_l0 = (n_obs - n_exc) * _safe_log(1.0 - p) + n_exc * _safe_log(p)
    log_l1 = (n_obs - n_exc) * _safe_log(1.0 - pi_hat) + n_exc * _safe_log(pi_hat)
    lr_uc = float(max(0.0, -2.0 * (log_l0 - log_l1)))
    p_uc = _chi2_survival(lr_uc, 1)

    if n_obs < 2:
        lr_ind = 0.0
        p_ind = 1.0
    else:
        e0 = exceptions[:-1]
        e1 = exceptions[1:]
        n00 = int(np.sum((e0 == 0) & (e1 == 0)))
        n01 = int(np.sum((e0 == 0) & (e1 == 1)))
        n10 = int(np.sum((e0 == 1) & (e1 == 0)))
        n11 = int(np.sum((e0 == 1) & (e1 == 1)))

        total_transitions = n00 + n01 + n10 + n11
        if total_transitions == 0:
            lr_ind = 0.0
            p_ind = 1.0
        else:
            pi01 = n01 / max(n00 + n01, 1)
            pi11 = n11 / max(n10 + n11, 1)
            pi1 = (n01 + n11) / max(total_transitions, 1)

            log_l_ind = (n00 + n10) * _safe_log(1.0 - pi1) + (n01 + n11) * _safe_log(pi1)
            log_l_dep = (
                n00 * _safe_log(1.0 - pi01)
                + n01 * _safe_log(pi01)
                + n10 * _safe_log(1.0 - pi11)
                + n11 * _safe_log(pi11)
            )
            lr_ind = float(max(0.0, -2.0 * (log_l_ind - log_l_dep)))
            p_ind = _chi2_survival(lr_ind, 1)

    lr_cc = float(lr_uc + lr_ind)
    p_cc = _chi2_survival(lr_cc, 2)

    return VaRBacktestResult(
        alpha=alpha,
        n_obs=n_obs,
        n_exceptions=n_exc,
        exception_rate=exc_rate,
        lr_uc=lr_uc,
        p_value_uc=p_uc,
        lr_ind=lr_ind,
        p_value_ind=p_ind,
        lr_cc=lr_cc,
        p_value_cc=p_cc,
    )


def run_delta_hedge_backtest(
    *,
    true_model: str,
    s0: float = 100.0,
    strike: float = 100.0,
    r: float = 0.03,
    maturity: float = 1.0,
    sigma_hedger: float = 0.2,
    sigma_true: float = 0.2,
    heston_true: Optional[HestonParams] = None,
    is_call: bool = True,
    num_paths: int = 3000,
    num_steps: int = 52,
    rebalance_every: int = 1,
    transaction_cost: float = 0.0,
    execution_model: Optional[ExecutionModel] = None,
    seed: int = 42,
) -> HedgingBacktestResult:
    """Run discrete-time delta hedging backtest and return PnL distribution summary."""

    pnl = _delta_hedge_pnl(
        true_model=true_model,
        s0=s0,
        strike=strike,
        r=r,
        maturity=maturity,
        sigma_hedger=sigma_hedger,
        sigma_true=sigma_true,
        heston_true=heston_true,
        is_call=is_call,
        num_paths=num_paths,
        num_steps=num_steps,
        rebalance_every=rebalance_every,
        transaction_cost=transaction_cost,
        execution_model=execution_model,
        seed=seed,
    )
    return _summarize_pnl(pnl, num_steps=num_steps)


def sample_delta_hedge_pnl_paths(
    *,
    true_model: str,
    s0: float = 100.0,
    strike: float = 100.0,
    r: float = 0.03,
    maturity: float = 1.0,
    sigma_hedger: float = 0.2,
    sigma_true: float = 0.2,
    heston_true: Optional[HestonParams] = None,
    is_call: bool = True,
    num_paths: int = 3000,
    num_steps: int = 52,
    rebalance_every: int = 1,
    transaction_cost: float = 0.0,
    execution_model: Optional[ExecutionModel] = None,
    seed: int = 42,
) -> np.ndarray:
    """Return path-level delta-hedging PnL samples for portfolio studies."""

    return _delta_hedge_pnl(
        true_model=true_model,
        s0=s0,
        strike=strike,
        r=r,
        maturity=maturity,
        sigma_hedger=sigma_hedger,
        sigma_true=sigma_true,
        heston_true=heston_true,
        is_call=is_call,
        num_paths=num_paths,
        num_steps=num_steps,
        rebalance_every=rebalance_every,
        transaction_cost=transaction_cost,
        execution_model=execution_model,
        seed=seed,
    )


def run_delta_vega_hedge_backtest(
    *,
    true_model: str,
    s0: float = 100.0,
    strike: float = 100.0,
    r: float = 0.03,
    maturity: float = 1.0,
    sigma_hedger: float = 0.2,
    sigma_true: float = 0.2,
    heston_true: Optional[HestonParams] = None,
    is_call: bool = True,
    hedge_strike: float = 110.0,
    hedge_maturity: Optional[float] = None,
    hedge_is_call: bool = True,
    num_paths: int = 3000,
    num_steps: int = 52,
    rebalance_every: int = 1,
    transaction_cost: float = 0.0,
    execution_model: Optional[ExecutionModel] = None,
    seed: int = 42,
) -> HedgingBacktestResult:
    """Run delta-vega hedging using one liquid secondary option."""

    hm = maturity if hedge_maturity is None else max(hedge_maturity, maturity)
    pnl = _delta_vega_hedge_pnl(
        true_model=true_model,
        s0=s0,
        strike=strike,
        r=r,
        maturity=maturity,
        sigma_hedger=sigma_hedger,
        sigma_true=sigma_true,
        heston_true=heston_true,
        is_call=is_call,
        hedge_strike=hedge_strike,
        hedge_maturity=hm,
        hedge_is_call=hedge_is_call,
        num_paths=num_paths,
        num_steps=num_steps,
        rebalance_every=rebalance_every,
        transaction_cost=transaction_cost,
        execution_model=execution_model,
        seed=seed,
    )
    return _summarize_pnl(pnl, num_steps=num_steps)


def hedging_transaction_cost_frontier(
    *,
    true_model: str,
    s0: float,
    strike: float,
    r: float,
    maturity: float,
    sigma: float,
    sigma_true: float,
    heston_true: Optional[HestonParams],
    costs: Sequence[float],
    num_paths: int,
    num_steps: int,
    rebalance_every: int,
    seed: int,
    execution_model: Optional[ExecutionModel] = None,
) -> List[Dict[str, float]]:
    """Compute delta vs delta-vega risk frontier across transaction costs."""

    rows: List[Dict[str, float]] = []
    for i, cost in enumerate(costs):
        delta = run_delta_hedge_backtest(
            true_model=true_model,
            s0=s0,
            strike=strike,
            r=r,
            maturity=maturity,
            sigma_hedger=sigma,
            sigma_true=sigma_true,
            heston_true=heston_true,
            num_paths=num_paths,
            num_steps=num_steps,
            rebalance_every=rebalance_every,
            transaction_cost=float(cost),
            execution_model=execution_model,
            seed=seed + 100 + i,
        )
        dvega = run_delta_vega_hedge_backtest(
            true_model=true_model,
            s0=s0,
            strike=strike,
            r=r,
            maturity=maturity,
            sigma_hedger=sigma,
            sigma_true=sigma_true,
            heston_true=heston_true,
            num_paths=num_paths,
            num_steps=num_steps,
            rebalance_every=rebalance_every,
            transaction_cost=float(cost),
            execution_model=execution_model,
            seed=seed + 200 + i,
        )
        rows.append(
            {
                "transaction_cost": float(cost),
                "delta_only_cvar95_loss": float(delta.cvar95_loss),
                "delta_vega_cvar95_loss": float(dvega.cvar95_loss),
                "delta_only_rmse_pnl": float(delta.rmse_pnl),
                "delta_vega_rmse_pnl": float(dvega.rmse_pnl),
            }
        )
    return rows


def hedging_rebalance_stability(
    *,
    true_model: str,
    s0: float,
    strike: float,
    r: float,
    maturity: float,
    sigma: float,
    sigma_true: float,
    heston_true: Optional[HestonParams],
    rebalance_grid: Sequence[int],
    transaction_cost: float,
    num_paths: int,
    num_steps: int,
    seed: int,
    execution_model: Optional[ExecutionModel] = None,
) -> List[Dict[str, float]]:
    """Stability diagnostics by rebalance frequency for both strategies."""

    rows: List[Dict[str, float]] = []
    for i, freq in enumerate(rebalance_grid):
        delta = run_delta_hedge_backtest(
            true_model=true_model,
            s0=s0,
            strike=strike,
            r=r,
            maturity=maturity,
            sigma_hedger=sigma,
            sigma_true=sigma_true,
            heston_true=heston_true,
            num_paths=num_paths,
            num_steps=num_steps,
            rebalance_every=int(max(freq, 1)),
            transaction_cost=transaction_cost,
            execution_model=execution_model,
            seed=seed + 300 + i,
        )
        dvega = run_delta_vega_hedge_backtest(
            true_model=true_model,
            s0=s0,
            strike=strike,
            r=r,
            maturity=maturity,
            sigma_hedger=sigma,
            sigma_true=sigma_true,
            heston_true=heston_true,
            num_paths=num_paths,
            num_steps=num_steps,
            rebalance_every=int(max(freq, 1)),
            transaction_cost=transaction_cost,
            execution_model=execution_model,
            seed=seed + 400 + i,
        )
        rows.append(
            {
                "rebalance_every_steps": int(max(freq, 1)),
                "delta_only_cvar95_loss": float(delta.cvar95_loss),
                "delta_vega_cvar95_loss": float(dvega.cvar95_loss),
                "delta_only_var95_loss": float(delta.var95_loss),
                "delta_vega_var95_loss": float(dvega.var95_loss),
            }
        )
    return rows


def compare_hedging_robustness(
    *,
    s0: float = 100.0,
    strike: float = 100.0,
    r: float = 0.03,
    maturity: float = 1.0,
    sigma: float = 0.2,
    num_paths: int = 3000,
    num_steps: int = 52,
    transaction_cost: float = 0.0005,
    alpha_var: float = 0.05,
    execution_model: Optional[ExecutionModel] = None,
    seed: int = 42,
) -> Dict[str, object]:
    """Compare well-specified vs misspecified hedging outcomes."""

    heston_true = HestonParams(v0=sigma**2, kappa=2.5, theta=sigma**2, xi=0.7, rho=-0.75)

    well_pnl = _delta_hedge_pnl(
        true_model="gbm",
        s0=s0,
        strike=strike,
        r=r,
        maturity=maturity,
        sigma_hedger=sigma,
        sigma_true=sigma,
        num_paths=num_paths,
        num_steps=num_steps,
        rebalance_every=1,
        transaction_cost=transaction_cost,
        execution_model=execution_model,
        seed=seed,
        is_call=True,
        heston_true=None,
    )
    miss_pnl = _delta_hedge_pnl(
        true_model="heston",
        s0=s0,
        strike=strike,
        r=r,
        maturity=maturity,
        sigma_hedger=sigma,
        sigma_true=sigma,
        heston_true=heston_true,
        num_paths=num_paths,
        num_steps=num_steps,
        rebalance_every=1,
        transaction_cost=transaction_cost,
        execution_model=execution_model,
        seed=seed + 1,
        is_call=True,
    )

    # New study: delta-vega strategy under the same true dynamics.
    well_dv = run_delta_vega_hedge_backtest(
        true_model="gbm",
        s0=s0,
        strike=strike,
        r=r,
        maturity=maturity,
        sigma_hedger=sigma,
        sigma_true=sigma,
        heston_true=None,
        num_paths=num_paths,
        num_steps=num_steps,
        rebalance_every=1,
        transaction_cost=transaction_cost,
        execution_model=execution_model,
        seed=seed + 2,
    )
    miss_dv = run_delta_vega_hedge_backtest(
        true_model="heston",
        s0=s0,
        strike=strike,
        r=r,
        maturity=maturity,
        sigma_hedger=sigma,
        sigma_true=sigma,
        heston_true=heston_true,
        num_paths=num_paths,
        num_steps=num_steps,
        rebalance_every=1,
        transaction_cost=transaction_cost,
        execution_model=execution_model,
        seed=seed + 3,
    )

    well_specified = _summarize_pnl(well_pnl, num_steps=num_steps)
    misspecified = _summarize_pnl(miss_pnl, num_steps=num_steps)
    robustness_ratio = misspecified.cvar95_loss / max(well_specified.cvar95_loss, 1e-12)
    delta_vega_ratio = miss_dv.cvar95_loss / max(misspecified.cvar95_loss, 1e-12)

    well_losses = -well_pnl
    miss_losses = -miss_pnl
    var_threshold = float(np.quantile(well_losses, 1.0 - alpha_var))
    es_model = float(np.mean(well_losses[well_losses >= var_threshold]))
    es_well = float(np.mean(well_losses[well_losses >= var_threshold]))
    tail_miss = miss_losses[miss_losses >= var_threshold]
    es_miss = float(np.mean(tail_miss)) if tail_miss.size > 0 else var_threshold
    es_ratio = es_miss / max(es_model, 1e-12)

    var_backtest_well = _var_backtest(well_losses, var_threshold=var_threshold, alpha=alpha_var)
    var_backtest_miss = _var_backtest(miss_losses, var_threshold=var_threshold, alpha=alpha_var)

    frontier = hedging_transaction_cost_frontier(
        true_model="heston",
        s0=s0,
        strike=strike,
        r=r,
        maturity=maturity,
        sigma=sigma,
        sigma_true=sigma,
        heston_true=heston_true,
        costs=[0.0, 0.00025, 0.0005, 0.001],
        num_paths=max(700, num_paths // 3),
        num_steps=num_steps,
        rebalance_every=1,
        seed=seed,
        execution_model=execution_model,
    )
    stability = hedging_rebalance_stability(
        true_model="heston",
        s0=s0,
        strike=strike,
        r=r,
        maturity=maturity,
        sigma=sigma,
        sigma_true=sigma,
        heston_true=heston_true,
        rebalance_grid=[1, 2, 4, 8],
        transaction_cost=transaction_cost,
        num_paths=max(700, num_paths // 3),
        num_steps=num_steps,
        seed=seed,
        execution_model=execution_model,
    )

    return {
        "well_specified": asdict(well_specified),
        "misspecified_heston": asdict(misspecified),
        "delta_vega_well_specified": asdict(well_dv),
        "delta_vega_misspecified_heston": asdict(miss_dv),
        "cvar95_loss_ratio_misspecified_over_well_specified": robustness_ratio,
        "cvar95_loss_ratio_delta_vega_over_delta_only_misspecified": delta_vega_ratio,
        "var_backtest_alpha": alpha_var,
        "var_threshold_loss": var_threshold,
        "expected_shortfall_model_loss": es_model,
        "expected_shortfall_realized_well_specified": es_well,
        "expected_shortfall_realized_misspecified": es_miss,
        "expected_shortfall_ratio_misspecified_over_model": es_ratio,
        "var_backtest_well_specified": asdict(var_backtest_well),
        "var_backtest_misspecified_heston": asdict(var_backtest_miss),
        "transaction_cost_frontier_heston": frontier,
        "rebalance_stability_heston": stability,
    }


def run_execution_aware_hedging_study(
    *,
    s0: float = 100.0,
    strike: float = 100.0,
    r: float = 0.03,
    maturity: float = 1.0,
    sigma: float = 0.2,
    num_paths: int = 2200,
    num_steps: int = 52,
    seed: int = 42,
) -> Dict[str, object]:
    """Execution-aware stress study for realistic frictions."""

    base = compare_hedging_robustness(
        s0=s0,
        strike=strike,
        r=r,
        maturity=maturity,
        sigma=sigma,
        num_paths=num_paths,
        num_steps=num_steps,
        transaction_cost=0.0005,
        execution_model=None,
        seed=seed,
    )
    stressed_model = ExecutionModel(
        bid_ask_bps=8.0,
        slippage_bps=5.0,
        impact_bps_per_unit=0.40,
        fill_probability=0.92,
    )
    stressed = compare_hedging_robustness(
        s0=s0,
        strike=strike,
        r=r,
        maturity=maturity,
        sigma=sigma,
        num_paths=num_paths,
        num_steps=num_steps,
        transaction_cost=0.0005,
        execution_model=stressed_model,
        seed=seed + 1,
    )
    return {
        "frictionless": base,
        "execution_stressed": stressed,
        "execution_model": asdict(stressed_model),
    }
