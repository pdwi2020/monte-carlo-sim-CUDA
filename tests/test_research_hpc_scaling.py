"""Tests for HPC scaling diagnostics."""

from research.hpc_scaling import hpc_scaling_to_dict, run_hpc_scaling_study


def test_hpc_scaling_runs():
    out = run_hpc_scaling_study(problem_sizes=(50_000, 80_000), repeats=1, seed=68)
    assert len(out.rows) >= 2
    payload = hpc_scaling_to_dict(out)
    assert "max_single_gpu_speedup" in payload
