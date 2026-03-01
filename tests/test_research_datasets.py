"""Tests for dataset builders/loaders."""

from research.datasets import build_multi_year_synthetic_dataset, panel_dataset_to_dict, resample_dataset


def test_multi_year_dataset_and_resample():
    ds = build_multi_year_synthetic_dataset(years=1, step_days=21, seed=61)
    assert ds.num_panels >= 8
    rs = resample_dataset(ds, every_n=2, max_panels=6)
    assert rs.num_panels <= 6
    payload = panel_dataset_to_dict(rs)
    assert payload["num_panels"] == rs.num_panels
