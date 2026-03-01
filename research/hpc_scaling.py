"""HPC scaling diagnostics including optional GPU/multi-device estimates."""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence

import numpy as np

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except Exception:  # pragma: no cover
    cp = None
    CUPY_AVAILABLE = False


@dataclass
class ScalingRow:
    """Runtime row for one backend/problem size."""

    backend: str
    problem_size: int
    runtime_seconds: float


@dataclass
class HPCScalingResult:
    """Combined single-device and multi-device scaling summary."""

    rows: List[ScalingRow]
    cpu_backend: str
    gpu_available: bool
    gpu_count: int
    max_single_gpu_speedup: float
    estimated_multi_gpu_speedup_2x: float
    estimated_multi_gpu_speedup_4x: float


def _bench_numpy(n: int, repeats: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    for _ in range(repeats):
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        _ = float(np.mean(np.exp(-0.5 * (x - y) ** 2)))
    return float((time.perf_counter() - t0) / max(repeats, 1))


def _bench_cupy(n: int, repeats: int, seed: int) -> float:
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy unavailable")
    rs = cp.random.RandomState(seed)
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        x = rs.standard_normal(n, dtype=cp.float32)
        y = rs.standard_normal(n, dtype=cp.float32)
        z = cp.mean(cp.exp(-0.5 * (x - y) ** 2))
        _ = float(z.get())
    cp.cuda.Stream.null.synchronize()
    return float((time.perf_counter() - t0) / max(repeats, 1))


def run_hpc_scaling_study(
    *,
    problem_sizes: Sequence[int] = (200_000, 500_000, 1_000_000),
    repeats: int = 3,
    seed: int = 42,
) -> HPCScalingResult:
    """Benchmark CPU/GPU kernels and report multi-GPU scaling estimates."""

    rows: List[ScalingRow] = []
    cpu_times = []
    gpu_times = []

    for i, n in enumerate(problem_sizes):
        t_cpu = _bench_numpy(int(n), repeats=repeats, seed=seed + i)
        rows.append(ScalingRow(backend="numpy_cpu", problem_size=int(n), runtime_seconds=t_cpu))
        cpu_times.append(t_cpu)

        if CUPY_AVAILABLE:
            try:
                t_gpu = _bench_cupy(int(n), repeats=repeats, seed=seed + 500 + i)
                rows.append(ScalingRow(backend="cupy_gpu", problem_size=int(n), runtime_seconds=t_gpu))
                gpu_times.append(t_gpu)
            except Exception:
                pass

    gpu_count = 0
    if CUPY_AVAILABLE:
        try:
            gpu_count = int(cp.cuda.runtime.getDeviceCount())
        except Exception:
            gpu_count = 0

    max_speedup = 1.0
    if gpu_times and cpu_times:
        for c, g in zip(cpu_times[: len(gpu_times)], gpu_times):
            max_speedup = max(max_speedup, float(c / max(g, 1e-12)))

    # Amdahl-style rough projection with overhead penalty.
    parallel_frac = 0.94
    overhead = 0.10
    s2 = 1.0 / ((1.0 - parallel_frac) + parallel_frac / 2.0 + overhead)
    s4 = 1.0 / ((1.0 - parallel_frac) + parallel_frac / 4.0 + overhead)
    return HPCScalingResult(
        rows=rows,
        cpu_backend="numpy_cpu",
        gpu_available=bool(gpu_times),
        gpu_count=max(gpu_count, 0),
        max_single_gpu_speedup=float(max_speedup),
        estimated_multi_gpu_speedup_2x=float(s2),
        estimated_multi_gpu_speedup_4x=float(s4),
    )


def hpc_scaling_to_dict(result: HPCScalingResult) -> Dict[str, object]:
    """Serialize HPC scaling result."""

    return {
        "rows": [asdict(x) for x in result.rows],
        "cpu_backend": result.cpu_backend,
        "gpu_available": result.gpu_available,
        "gpu_count": result.gpu_count,
        "max_single_gpu_speedup": result.max_single_gpu_speedup,
        "estimated_multi_gpu_speedup_2x": result.estimated_multi_gpu_speedup_2x,
        "estimated_multi_gpu_speedup_4x": result.estimated_multi_gpu_speedup_4x,
    }
