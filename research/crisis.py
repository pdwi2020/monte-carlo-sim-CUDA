"""Crisis/subperiod empirical diagnostics for model robustness."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from calibration import MarketOption
from .econometrics import diebold_mariano_test
from .historical_backtest import DatedOptionPanel


@dataclass
class EpisodeDefinition:
    """One empirical subperiod episode."""

    episode_id: str
    label: str
    start_date: str
    end_date: str
    kind: str


@dataclass
class EpisodeModelPerformance:
    """Forecast performance for one model in one episode."""

    episode_id: str
    model: str
    mean_rmse: float
    std_rmse: float
    count: int


@dataclass
class EpisodeDMTest:
    """Episode-level DM test best model vs naive benchmark."""

    episode_id: str
    best_model: str
    benchmark_model: str
    statistic: float
    p_value: float
    n: int


@dataclass
class CrisisSubperiodStudyResult:
    """Aggregate crisis/subperiod study output."""

    episodes: List[EpisodeDefinition]
    episode_performance: List[EpisodeModelPerformance]
    best_model_by_episode: Dict[str, str]
    dm_tests: List[EpisodeDMTest]
    stress_episode_id: Optional[str]
    stress_best_model: Optional[str]
    num_dates: int
    num_observations: int


_KNOWN_EPISODES = [
    ("covid_crash_2020", "COVID Crash", "2020-02-15", "2020-06-30"),
    ("rates_shock_2022", "Inflation/Rate Shock", "2022-03-01", "2022-12-31"),
    ("bank_stress_2023", "Regional Bank Stress", "2023-03-01", "2023-05-31"),
]


def _safe_iv(opt: MarketOption, *, spot: float) -> Optional[float]:
    if opt.market_iv is not None:
        return float(opt.market_iv)
    px = opt.market_price if opt.market_price is not None else opt.mid_price
    if px is None or opt.maturity <= 0:
        return None
    approx = np.sqrt(2.0 * np.pi / max(float(opt.maturity), 1e-8)) * float(px) / max(float(spot), 1e-12)
    return float(np.clip(approx, 0.01, 3.0))


def _panel_atm_iv(panel: DatedOptionPanel, *, spot: float) -> float:
    ivs: List[Tuple[float, float]] = []
    for opt in panel.options:
        iv = _safe_iv(opt, spot=spot)
        if iv is None:
            continue
        m = abs(float(opt.strike) - float(spot))
        ivs.append((m, float(iv)))
    if not ivs:
        return 0.2
    ivs.sort(key=lambda x: x[0])
    return float(ivs[0][1])


def _extract_obs(obs: object) -> Optional[Tuple[str, str, float]]:
    if isinstance(obs, dict):
        model = obs.get("model")
        target = obs.get("target_date")
        rmse = obs.get("rmse")
    else:
        model = getattr(obs, "model", None)
        target = getattr(obs, "target_date", None)
        rmse = getattr(obs, "rmse", None)
    if model is None or target is None or rmse is None:
        return None
    return str(model), str(target), float(rmse)


def _date_in_range(d: str, start: str, end: str) -> bool:
    dd = date.fromisoformat(d)
    return date.fromisoformat(start) <= dd <= date.fromisoformat(end)


def _build_episodes(dates: Sequence[str]) -> List[EpisodeDefinition]:
    episodes: List[EpisodeDefinition] = []
    if dates:
        episodes.append(
            EpisodeDefinition(
                episode_id="full_sample",
                label="Full Sample",
                start_date=min(dates),
                end_date=max(dates),
                kind="global",
            )
        )
    for eid, label, start, end in _KNOWN_EPISODES:
        if any(_date_in_range(d, start, end) for d in dates):
            episodes.append(
                EpisodeDefinition(
                    episode_id=eid,
                    label=label,
                    start_date=start,
                    end_date=end,
                    kind="named_crisis",
                )
            )
    return episodes


def run_crisis_subperiod_study(
    panels: Sequence[DatedOptionPanel],
    *,
    forecast_observations: Sequence[object],
    spot: float = 100.0,
) -> CrisisSubperiodStudyResult:
    """Evaluate forecasting models by crisis-aware and stress-quantile subperiods."""

    ordered = sorted(list(panels), key=lambda p: p.quote_date)
    if len(ordered) < 5:
        raise ValueError("Need at least 5 panels for crisis/subperiod analysis")

    atm_by_date: Dict[str, float] = {p.quote_date: _panel_atm_iv(p, spot=spot) for p in ordered}
    dates = sorted(atm_by_date.keys())
    atm = np.asarray([atm_by_date[d] for d in dates], dtype=float)
    q_low, q_high = np.quantile(atm, [0.33, 0.67]) if atm.size >= 3 else (float(np.min(atm)), float(np.max(atm)))

    date_to_episode: Dict[str, str] = {}
    for d in dates:
        named = None
        for eid, _, start, end in _KNOWN_EPISODES:
            if _date_in_range(d, start, end):
                named = eid
                break
        if named is not None:
            date_to_episode[d] = named
            continue
        val = atm_by_date[d]
        if val <= q_low:
            date_to_episode[d] = "low_stress_quantile"
        elif val <= q_high:
            date_to_episode[d] = "mid_stress_quantile"
        else:
            date_to_episode[d] = "high_stress_quantile"

    episodes = _build_episodes(dates)
    for ep in ("low_stress_quantile", "mid_stress_quantile", "high_stress_quantile"):
        ep_dates = [d for d in dates if date_to_episode.get(d) == ep]
        if ep_dates:
            episodes.append(
                EpisodeDefinition(
                    episode_id=ep,
                    label=ep.replace("_", " ").title(),
                    start_date=min(ep_dates),
                    end_date=max(ep_dates),
                    kind="quantile_stress",
                )
            )

    by_episode_model: Dict[Tuple[str, str], List[float]] = {}
    by_episode_model_dates: Dict[Tuple[str, str], List[str]] = {}
    num_obs = 0
    for obs in forecast_observations:
        parsed = _extract_obs(obs)
        if parsed is None:
            continue
        model, target_date, rmse = parsed
        num_obs += 1
        primary = date_to_episode.get(target_date)
        active_episodes = ["full_sample"]
        if primary is not None:
            active_episodes.append(primary)
        for ep in active_episodes:
            key = (ep, model)
            by_episode_model.setdefault(key, []).append(float(rmse))
            by_episode_model_dates.setdefault(key, []).append(target_date)

    perf_rows: List[EpisodeModelPerformance] = []
    best_model_by_episode: Dict[str, str] = {}
    by_episode_summary: Dict[str, Dict[str, float]] = {}
    for (ep, model), vals in sorted(by_episode_model.items()):
        arr = np.asarray(vals, dtype=float)
        perf_rows.append(
            EpisodeModelPerformance(
                episode_id=ep,
                model=model,
                mean_rmse=float(np.mean(arr)),
                std_rmse=float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
                count=int(arr.size),
            )
        )
        by_episode_summary.setdefault(ep, {})[model] = float(np.mean(arr))

    for ep, scores in by_episode_summary.items():
        best_model_by_episode[ep] = min(scores.keys(), key=lambda m: scores[m])

    dm_rows: List[EpisodeDMTest] = []
    for ep, scores in by_episode_summary.items():
        best = best_model_by_episode.get(ep)
        if best is None or best == "naive_last_surface":
            continue
        key_best = (ep, best)
        key_naive = (ep, "naive_last_surface")
        if key_naive not in by_episode_model or key_best not in by_episode_model:
            continue
        b = np.asarray(by_episode_model[key_best], dtype=float)
        n = np.asarray(by_episode_model[key_naive], dtype=float)
        k = int(min(b.size, n.size))
        if k < 5:
            continue
        dm = diebold_mariano_test(b[:k], n[:k], alternative="less", lag=1)
        dm_rows.append(
            EpisodeDMTest(
                episode_id=ep,
                best_model=best,
                benchmark_model="naive_last_surface",
                statistic=float(dm.statistic),
                p_value=float(dm.p_value),
                n=int(dm.n),
            )
        )

    stress_candidates = [ep for ep in by_episode_summary.keys() if ep != "full_sample"]
    if stress_candidates:
        stress_episode = max(
            stress_candidates,
            key=lambda ep: float(
                np.mean([atm_by_date[d] for d in dates if date_to_episode.get(d) == ep] or [0.0])
            ),
        )
    else:
        stress_episode = None
    stress_best_model = best_model_by_episode.get(stress_episode) if stress_episode is not None else None

    return CrisisSubperiodStudyResult(
        episodes=episodes,
        episode_performance=perf_rows,
        best_model_by_episode=best_model_by_episode,
        dm_tests=dm_rows,
        stress_episode_id=stress_episode,
        stress_best_model=stress_best_model,
        num_dates=len(dates),
        num_observations=num_obs,
    )


def crisis_to_dict(result: CrisisSubperiodStudyResult) -> Dict[str, object]:
    """Serialize crisis/subperiod study result."""

    return {
        "episodes": [asdict(x) for x in result.episodes],
        "episode_performance": [asdict(x) for x in result.episode_performance],
        "best_model_by_episode": dict(result.best_model_by_episode),
        "dm_tests": [asdict(x) for x in result.dm_tests],
        "stress_episode_id": result.stress_episode_id,
        "stress_best_model": result.stress_best_model,
        "num_dates": result.num_dates,
        "num_observations": result.num_observations,
    }
