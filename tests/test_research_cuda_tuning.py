"""Tests for CUDA tuning harness."""

from research.cuda_tuning import cuda_tuning_to_dict, run_cuda_autotune


class _FakeCudaModule:
    def price_bates_full_sequence(self, *args):
        return 1.0

    def price_bates_full_sequence_with_config(self, *args):
        threads = int(args[-2])
        streams = int(args[-1])
        return 1.0 + 1e-6 * (threads + streams)

    def price_bates_full_sequence_mixed_precision_with_config(self, *args):
        threads = int(args[-2])
        streams = int(args[-1])
        return 1.0 + 1e-6 * (0.5 * threads + streams)


def test_cuda_tuning_not_available(monkeypatch):
    def fake_import(name):
        raise ImportError("no module")

    monkeypatch.setattr("importlib.import_module", fake_import)
    out = run_cuda_autotune(num_paths=1000, num_steps=10, repeats=1)
    assert out.status == "not_available"
    assert out.module_available is False


def test_cuda_tuning_with_fake_module():
    out = run_cuda_autotune(
        num_paths=1000,
        num_steps=10,
        repeats=1,
        threads_grid=(64, 128),
        streams_grid=(1, 2),
        module=_FakeCudaModule(),
    )
    assert out.status == "ok"
    assert out.module_available is True
    assert out.best_config is not None
    payload = cuda_tuning_to_dict(out)
    assert payload["num_candidates"] >= 4
