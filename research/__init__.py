"""Research pipeline package for doctoral-grade experimentation."""


def run_research_pipeline(*args, **kwargs):
    """Lazily import pipeline entrypoint to avoid module pre-import side effects."""

    from .pipeline import run_research_pipeline as _run_research_pipeline

    return _run_research_pipeline(*args, **kwargs)


__all__ = ["run_research_pipeline"]
