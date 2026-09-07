"""Compatibility entry point for the canonical cosmology bsel module."""
from pathlib import Path
import sys

_PIPELINES_DIR = str(Path(__file__).resolve().parents[1] / "src" / "pipelines")
if _PIPELINES_DIR not in sys.path:
    sys.path.insert(0, _PIPELINES_DIR)

from cosmology.bsel import *  # noqa: F401,F403
from cosmology.bsel import cleanup, execute, setup
