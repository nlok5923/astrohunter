"""astrohunt — a modern, CPU-first, cross-platform asteroid detection pipeline.

A first-principles successor to Astrometrica for IASC-style citizen-science
asteroid hunting. The MVP core is fully deterministic classical astrometry and
runs natively on Apple Silicon (arm64) with no GPU required.

Pipeline:
    ingest FITS -> extract sources -> LSPC astrometric solution
    -> static-source suppression -> tracklet linking
    -> known-object match -> ADES report

See RESEARCH.md for the grounding behind these design choices.
"""

__version__ = "0.1.0"

from .pipeline import run_pipeline, PipelineResult  # noqa: E402,F401

__all__ = ["run_pipeline", "PipelineResult", "__version__"]
