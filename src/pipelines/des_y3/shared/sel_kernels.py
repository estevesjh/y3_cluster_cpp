"""Reuse of the maintained selection kernels from sel_function.py.

The approved layout proposal requires new implementations to *reuse* the
maintained shared model layer instead of copying HOD / richness / photo-z
constructors. The Python side of that layer is
``src/pipelines/des_y3/shared/sel_function.py`` (which itself mirrors
``mor_hod_t.hh`` and ``richness_kernel_t.hh``); this helper loads it once
by path — it is a module file, not a package — and caches it.

``src/modules/sel_function/sel_function.py`` (the CosmoSIS module entry
point) is currently an exact copy of the same file, kept until
downstream consumers are confirmed migrated; this helper reads only the
shared-folder copy.

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
        if (p / "src" / "pipelines" / "des_y3" / "shared" / "sel_function.py").is_file():
            return p
    raise ImportError(
        "sel_kernels: could not locate the y3_cluster_cpp repository root")


def load():
    """Load (once) and return the maintained sel_function module."""
    name = "y3_des_sel_function"
    if name in sys.modules:
        return sys.modules[name]
    root = repo_root()
    # sel_function.py imports y3_buzzard.prj_params — make sure the repo
    # root is importable even when PYTHONPATH does not already include it.
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    path = root / "src" / "pipelines" / "des_y3" / "shared" / "sel_function.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def mor_from_source(source):
    """HOD parameter dict from a Source — mirror of sel_function._read_mor."""
    sf = load()
    log10_mmin = source.scalar("cluster_mor", "log10_Mmin")
    try:
        log10_m1 = log10_mmin + source.scalar("cluster_mor", "log10_ratio")
    except Exception:
        log10_m1 = source.scalar("cluster_mor", "log10_M1")
    try:
        z_pivot = source.scalar("cluster_mor", "z_pivot")
    except Exception:
        z_pivot = sf.Z_PIVOT_DEFAULT
    return dict(
        log10_Mmin=log10_mmin,
        log10_M1=log10_m1,
        alpha=source.scalar("cluster_mor", "alpha"),
        epsilon=source.scalar("cluster_mor", "epsilon"),
        sigma_lambda=source.scalar("cluster_mor", "sigma_lambda"),
        z_pivot=z_pivot,
    )


def plob_splines_default():
    """The frozen Y3 EMG coefficient splines (PrjParams.default())."""
    load()  # ensures repo root is on sys.path for the y3_buzzard import
    from y3_buzzard.prj_params import PrjParams
    return PrjParams.default().splines()
