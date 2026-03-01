"""Lightweight experiment registry (MLflow-style JSON/CSV artifacts)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "__dict__"):
        return _to_jsonable(obj.__dict__)
    return str(obj)


def _extract_key_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out["mlmc_avg_speedup"] = float((results.get("mlmc") or {}).get("avg_speedup", 0.0) or 0.0)
    out["heston_mlmc_avg_speedup"] = float((results.get("heston_mlmc") or {}).get("avg_speedup", 0.0) or 0.0)
    out["historical_test_rmse"] = float((results.get("historical_backtest") or {}).get("test_mean_rmse", 0.0) or 0.0)
    out["forecast_best_rmse"] = float(
        min(((results.get("forecasting_oos") or {}).get("mean_rmse_by_model") or {"_": 0.0}).values())
    )
    out["portfolio_cvar95"] = float((results.get("portfolio_overlay") or {}).get("portfolio_cvar95_loss", 0.0) or 0.0)
    return out


def write_experiment_registry(
    *,
    output_dir: str | Path,
    manifest: Any,
    results: Dict[str, Any],
    tags: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Write registry entries under output directory."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": _to_jsonable(manifest),
        "tags": dict(tags or {}),
        "metrics": _extract_key_metrics(results),
    }
    run_json = out_dir / f"run_{run_id}.json"
    run_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    metrics_csv = out_dir / "metrics_latest.csv"
    with metrics_csv.open("w", encoding="utf-8", newline="") as f:
        f.write("metric,value\n")
        for k, v in payload["metrics"].items():
            f.write(f"{k},{v}\n")

    tags_json = out_dir / "tags_latest.json"
    tags_json.write_text(json.dumps(payload["tags"], indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "run_json": str(run_json),
        "metrics_csv": str(metrics_csv),
        "tags_json": str(tags_json),
    }
