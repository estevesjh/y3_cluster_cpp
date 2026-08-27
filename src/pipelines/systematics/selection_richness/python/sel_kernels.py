"""Reuse of the maintained selection kernels from sel_function.py.

The approved layout proposal requires new implementations to *reuse* the
maintained shared model layer instead of copying HOD / richness / photo-z
constructors. The Python side of that layer is
``src/pipelines/systematics/selection_richness/python/sel_function.py``
(which itself mirrors ``mor_hod_t.hh`` and ``richness_kernel_t.hh``); this
helper loads it once by path and caches it.

``src/modules/sel_function/sel_function.py`` is only the CosmoSIS module
shim. It imports this shared implementation, so there is one selection
kernel source for both the module entry point and offline consumers.

Also provides small Source-protocol equivalents of its DataBlock readers
so offline validators can replay a test-sampler dump.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def repo_root():
    """Locate the y3_cluster_cpp root from this file's position."""
    for p in Path(__file__).resolve().parents:
        if (p / "src" / "pipelines" / "systematics" / "selection_richness"
                / "python" / "sel_function.py").is_file():
            return p
    raise ImportError(
        "sel_kernels: could not locate the y3_cluster_cpp repository root")


def load():
    """Load (once) and return the maintained sel_function module."""
    name = "y3_des_sel_function_systematics"
    if name in sys.modules:
        return sys.modules[name]
    root = repo_root()
    # sel_function.py imports the canonical systematics prj_params module;
    # make sure src/pipelines is importable even when PYTHONPATH does not
    # already include it.
    pipelines_dir = root / "src" / "pipelines"
    if str(pipelines_dir) not in sys.path:
        sys.path.insert(0, str(pipelines_dir))
    path = (root / "src" / "pipelines" / "systematics"
            / "selection_richness" / "python" / "sel_function.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def mor_from_source(source):
    """Return the legacy MOR dictionary through the shared HOD adapter.

    The downstream full-ltmz reference functions still use dictionary
    indexing. Parameter normalization itself belongs to ``HODParameters`` so
    this compatibility function does not maintain a second datablock reader.
    """
    load()  # keep the historical side effect of making src/pipelines importable
    from shared import datablock_models as dm

    parameters = dm.HODParameters.from_source(source)
    return {
        name: getattr(parameters, name)
        for name in (
            "log10_Mmin", "log10_M1", "alpha", "epsilon",
            "sigma_lambda", "z_pivot",
        )
    }


def plob_splines_default():
    """The frozen Y3 EMG coefficient splines (PrjParams.default())."""
    load()  # ensures src/pipelines is on sys.path for the systematics import
    from systematics.selection_function.prj_params import PrjParams
    return PrjParams.default().splines()
