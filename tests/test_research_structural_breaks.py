"""Tests for structural-break diagnostics."""

from research.structural_breaks import run_structural_break_diagnostics, structural_breaks_to_dict


def test_structural_break_detects_shift():
    dates = [f"2024-01-{i:02d}" for i in range(1, 21)]
    series = [0.10] * 10 + [0.20] * 10
    out = run_structural_break_diagnostics(
        dates_by_series={"x": dates},
        values_by_series={"x": series},
        min_segment_size=3,
        n_bootstrap=120,
        seed=7,
    )
    assert out.num_series == 1
    assert out.entries[0].break_index >= 7
    assert out.entries[0].break_index <= 13
    payload = structural_breaks_to_dict(out)
    assert payload["entries"][0]["series_id"] == "x"
