"""Performance benchmarking for backend and scaling diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from mc_pricer import (
    Backend,
    CUPY_AVAILABLE,
    DiscretizationScheme,
    MarketData,
    PayoffType,
    SimulationConfig,
    VarianceReduction,
    price_option,
)


@dataclass
class BackendBenchmarkResult:
    """One backend scaling measurement."""

    backend: str
    num_paths: int
    num_steps: int
    price: float
    std_error: float
    runtime_seconds: float


def benchmark_backend_scaling(
    *,
    num_paths_grid: Optional[List[int]] = None,
    num_steps: int = 120,
    seed: int = 42,
) -> List[BackendBenchmarkResult]:
    """Run scaling benchmark across available compute backends."""

    if num_paths_grid is None:
        num_paths_grid = [4000, 8000, 16000]

    market = MarketData(S0=100.0, r=0.03, q=0.0)
    backends = [Backend.NUMPY]
    if CUPY_AVAILABLE:
        backends.append(Backend.CUPY)

    results: List[BackendBenchmarkResult] = []
    for backend in backends:
        for i, num_paths in enumerate(num_paths_grid):
            cfg = SimulationConfig(
                num_paths=num_paths,
                num_steps=num_steps,
                backend=backend,
                scheme=DiscretizationScheme.QE,
                variance_reduction=VarianceReduction.ANTITHETIC,
                seed=seed + i,
            )
            out = price_option(
                market=market,
                K=100.0,
                T=1.0,
                payoff_type=PayoffType.EUROPEAN_CALL,
                sigma=0.2,
                config=cfg,
            )
            results.append(
                BackendBenchmarkResult(
                    backend=backend.value,
                    num_paths=num_paths,
                    num_steps=num_steps,
                    price=out.price,
                    std_error=out.std_error,
                    runtime_seconds=float(out.elapsed_time or 0.0),
                )
            )

    return results


def summarize_backend_speedup(results: List[BackendBenchmarkResult]) -> Dict[str, float]:
    """Compute average GPU/CPU speedup when GPU results are present."""

    cpu = [r.runtime_seconds for r in results if r.backend == "numpy"]
    gpu = [r.runtime_seconds for r in results if r.backend == "cupy"]

    if not cpu:
        return {"cpu_mean_runtime": float("nan"), "gpu_mean_runtime": float("nan"), "gpu_speedup": float("nan")}
    if not gpu:
        return {
            "cpu_mean_runtime": float(sum(cpu) / len(cpu)),
            "gpu_mean_runtime": float("nan"),
            "gpu_speedup": float("nan"),
        }

    cpu_mean = float(sum(cpu) / len(cpu))
    gpu_mean = float(sum(gpu) / len(gpu))
    return {
        "cpu_mean_runtime": cpu_mean,
        "gpu_mean_runtime": gpu_mean,
        "gpu_speedup": cpu_mean / max(gpu_mean, 1e-12),
    }


def performance_to_dict(results: List[BackendBenchmarkResult]) -> List[Dict[str, float]]:
    """Serialize performance benchmark results."""

    return [asdict(r) for r in results]
