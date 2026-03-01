"""Global multiple-testing corrections across all reported hypothesis tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence

import numpy as np


@dataclass
class HypothesisTestRecord:
    """One hypothesis test entry with global adjustments."""

    test_id: str
    family: str
    raw_p_value: float
    holm_adjusted_p_value: float
    holm_reject: bool
    bh_adjusted_p_value: float
    bh_reject: bool


@dataclass
class GlobalMultipleTestingResult:
    """Global correction summary output."""

    alpha: float
    num_tests: int
    num_reject_holm: int
    num_reject_bh: int
    records: List[HypothesisTestRecord]


def _holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return np.asarray([], dtype=float)
    order = np.argsort(p)
    adj_sorted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order, start=1):
        mult = m - rank + 1
        val = min(1.0, float(p[idx] * mult))
        running = max(running, val)
        adj_sorted[rank - 1] = running
    out = np.empty(m, dtype=float)
    out[order] = adj_sorted
    return out


def _bh_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return np.asarray([], dtype=float)
    order = np.argsort(p)
    p_sorted = p[order]
    adj_sorted = np.empty(m, dtype=float)
    running = 1.0
    for i in range(m, 0, -1):
        rank = i
        val = min(1.0, float(p_sorted[i - 1] * m / max(rank, 1)))
        running = min(running, val)
        adj_sorted[i - 1] = running
    out = np.empty(m, dtype=float)
    out[order] = adj_sorted
    return out


def _collect_records(results: Dict[str, Any], claims: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for item in claims or []:
        claim = item.get("claim", {}) if isinstance(item, dict) else {}
        ev = item.get("evaluation", {}) if isinstance(item, dict) else {}
        p = ev.get("p_value")
        cid = claim.get("claim_id") or ev.get("claim_id")
        if p is not None and cid is not None:
            rows.append(
                {
                    "test_id": f"claim::{cid}",
                    "family": "claims",
                    "raw_p_value": float(p),
                }
            )

    sv = results.get("statistical_validation", {}) or {}
    dm = sv.get("diebold_mariano_efficiency") or {}
    if dm.get("p_value") is not None:
        rows.append({"test_id": "econometrics::diebold_mariano_efficiency", "family": "econometrics", "raw_p_value": float(dm["p_value"])})
    spa = sv.get("spa_vs_naive_forecast") or {}
    if spa.get("p_value") is not None:
        rows.append({"test_id": "econometrics::spa_vs_naive_forecast", "family": "econometrics", "raw_p_value": float(spa["p_value"])})
    rc = sv.get("white_reality_check_vs_naive") or {}
    if rc.get("p_value") is not None:
        rows.append({"test_id": "econometrics::white_reality_check_vs_naive", "family": "econometrics", "raw_p_value": float(rc["p_value"])})

    crisis = results.get("crisis_subperiod_study", {}) or {}
    for row in crisis.get("dm_tests") or []:
        p = row.get("p_value") if isinstance(row, dict) else None
        eid = row.get("episode_id") if isinstance(row, dict) else None
        if p is None or eid is None:
            continue
        rows.append(
            {
                "test_id": f"crisis_dm::{eid}",
                "family": "crisis_subperiod",
                "raw_p_value": float(p),
            }
        )

    ab = results.get("ablation_study", {}) or {}
    for row in ab.get("scenarios") or []:
        p = row.get("p_value") if isinstance(row, dict) else None
        sid = row.get("scenario_id") if isinstance(row, dict) else None
        if p is None or sid is None:
            continue
        rows.append(
            {
                "test_id": f"ablation::{sid}",
                "family": "ablation",
                "raw_p_value": float(p),
            }
        )

    return rows


def run_global_multiple_testing(
    *,
    results: Dict[str, Any],
    claims: Sequence[Dict[str, Any]],
    alpha: float = 0.05,
) -> GlobalMultipleTestingResult:
    """Apply global Holm and BH corrections across all pipeline p-values."""

    rows = _collect_records(results, claims)
    if not rows:
        return GlobalMultipleTestingResult(
            alpha=float(alpha),
            num_tests=0,
            num_reject_holm=0,
            num_reject_bh=0,
            records=[],
        )

    p = np.asarray([r["raw_p_value"] for r in rows], dtype=float)
    p = np.clip(p, 0.0, 1.0)
    holm = _holm_adjust(p)
    bh = _bh_adjust(p)

    recs: List[HypothesisTestRecord] = []
    for i, row in enumerate(rows):
        recs.append(
            HypothesisTestRecord(
                test_id=str(row["test_id"]),
                family=str(row["family"]),
                raw_p_value=float(p[i]),
                holm_adjusted_p_value=float(holm[i]),
                holm_reject=bool(holm[i] <= alpha),
                bh_adjusted_p_value=float(bh[i]),
                bh_reject=bool(bh[i] <= alpha),
            )
        )

    return GlobalMultipleTestingResult(
        alpha=float(alpha),
        num_tests=len(recs),
        num_reject_holm=int(sum(r.holm_reject for r in recs)),
        num_reject_bh=int(sum(r.bh_reject for r in recs)),
        records=recs,
    )


def multiple_testing_to_dict(result: GlobalMultipleTestingResult) -> Dict[str, object]:
    """Serialize global multiple-testing output."""

    return {
        "alpha": result.alpha,
        "num_tests": result.num_tests,
        "num_reject_holm": result.num_reject_holm,
        "num_reject_bh": result.num_reject_bh,
        "records": [asdict(r) for r in result.records],
    }
