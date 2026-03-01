"""CUDA kernel auto-tuning harness for Bates pipeline configuration."""

from __future__ import annotations

import importlib
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class CudaTuningCandidate:
    """Runtime observation for one CUDA launch configuration."""

    variant: str
    threads_per_block: int
    num_streams: int
    runtime_seconds: float
    price: float


@dataclass
class CudaTuningResult:
    """Auto-tuning payload."""

    status: str
    message: str
    module_available: bool
    baseline_runtime_seconds: Optional[float]
    best_runtime_seconds: Optional[float]
    best_speedup_over_baseline: Optional[float]
    best_config: Optional[Dict[str, object]]
    candidates: List[CudaTuningCandidate]


def _time_call(fn, *args, repeats: int = 1):
    t0 = time.perf_counter()
    out = None
    for _ in range(max(1, repeats)):
        out = fn(*args)
    dt = (time.perf_counter() - t0) / max(1, repeats)
    return float(dt), float(out)


def run_cuda_autotune(
    *,
    num_paths: int = 120_000,
    num_steps: int = 126,
    maturity: float = 1.0,
    strike: float = 100.0,
    s0: float = 100.0,
    r: float = 0.03,
    v0: float = 0.04,
    kappa: float = 2.0,
    theta: float = 0.04,
    xi: float = 0.3,
    rho: float = -0.7,
    lambda_j: float = 0.1,
    mu_j: float = -0.05,
    sigma_j: float = 0.1,
    threads_grid: Iterable[int] = (64, 128, 256, 512),
    streams_grid: Iterable[int] = (1, 2, 4, 8),
    repeats: int = 1,
    module=None,
) -> CudaTuningResult:
    """Auto-tune CUDA launch settings for available Bates extension."""

    mod = module
    if mod is None:
        try:
            mod = importlib.import_module("bates_kernel_cpp")
        except Exception as exc:
            return CudaTuningResult(
                status="not_available",
                message=f"bates_kernel_cpp unavailable: {exc}",
                module_available=False,
                baseline_runtime_seconds=None,
                best_runtime_seconds=None,
                best_speedup_over_baseline=None,
                best_config=None,
                candidates=[],
            )

    args = (
        int(num_paths),
        int(num_steps),
        float(maturity),
        float(strike),
        float(s0),
        float(r),
        float(v0),
        float(kappa),
        float(theta),
        float(xi),
        float(rho),
        float(lambda_j),
        float(mu_j),
        float(sigma_j),
    )

    try:
        baseline_dt, baseline_price = _time_call(mod.price_bates_full_sequence, *args, repeats=repeats)
    except Exception as exc:
        return CudaTuningResult(
            status="error",
            message=f"baseline call failed: {exc}",
            module_available=True,
            baseline_runtime_seconds=None,
            best_runtime_seconds=None,
            best_speedup_over_baseline=None,
            best_config=None,
            candidates=[],
        )

    candidates: List[CudaTuningCandidate] = []
    best_dt = baseline_dt
    best_cfg: Optional[Dict[str, object]] = {
        "variant": "baseline_default",
        "threads_per_block": 256,
        "num_streams": 1,
        "price": baseline_price,
    }

    for threads in threads_grid:
        for streams in streams_grid:
            try:
                dt, price = _time_call(
                    mod.price_bates_full_sequence_with_config,
                    *args,
                    int(threads),
                    int(streams),
                    repeats=repeats,
                )
                row = CudaTuningCandidate(
                    variant="double_precision",
                    threads_per_block=int(threads),
                    num_streams=int(streams),
                    runtime_seconds=float(dt),
                    price=float(price),
                )
                candidates.append(row)
                if dt < best_dt:
                    best_dt = dt
                    best_cfg = {
                        "variant": row.variant,
                        "threads_per_block": row.threads_per_block,
                        "num_streams": row.num_streams,
                        "price": row.price,
                    }
            except Exception:
                continue

            if hasattr(mod, "price_bates_full_sequence_mixed_precision_with_config"):
                try:
                    dt_f, price_f = _time_call(
                        mod.price_bates_full_sequence_mixed_precision_with_config,
                        *args,
                        int(threads),
                        int(streams),
                        repeats=repeats,
                    )
                    row_f = CudaTuningCandidate(
                        variant="mixed_precision",
                        threads_per_block=int(threads),
                        num_streams=int(streams),
                        runtime_seconds=float(dt_f),
                        price=float(price_f),
                    )
                    candidates.append(row_f)
                    if dt_f < best_dt:
                        best_dt = dt_f
                        best_cfg = {
                            "variant": row_f.variant,
                            "threads_per_block": row_f.threads_per_block,
                            "num_streams": row_f.num_streams,
                            "price": row_f.price,
                        }
                except Exception:
                    continue

    speedup = float(baseline_dt / best_dt) if best_dt > 0 else None
    return CudaTuningResult(
        status="ok",
        message="autotune completed",
        module_available=True,
        baseline_runtime_seconds=float(baseline_dt),
        best_runtime_seconds=float(best_dt),
        best_speedup_over_baseline=speedup,
        best_config=best_cfg,
        candidates=candidates,
    )


def cuda_tuning_to_dict(result: CudaTuningResult) -> Dict[str, object]:
    """Serialize CUDA tuning result."""

    return {
        "status": result.status,
        "message": result.message,
        "module_available": result.module_available,
        "baseline_runtime_seconds": result.baseline_runtime_seconds,
        "best_runtime_seconds": result.best_runtime_seconds,
        "best_speedup_over_baseline": result.best_speedup_over_baseline,
        "best_config": result.best_config,
        "num_candidates": len(result.candidates),
        "candidates": [asdict(x) for x in result.candidates],
    }
