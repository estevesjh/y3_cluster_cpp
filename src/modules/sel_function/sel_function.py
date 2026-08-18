"""CosmoSIS entry point for the maintained shared selection module."""
from pathlib import Path
import sys

_PIPELINES_DIR = str(Path(__file__).resolve().parents[2] / "pipelines")
if _PIPELINES_DIR not in sys.path:
    sys.path.insert(0, _PIPELINES_DIR)

from shared.sel_function import *  # noqa: F401,F403
from shared.sel_function import cleanup, execute, setup
