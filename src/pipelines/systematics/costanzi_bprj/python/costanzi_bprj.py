"""Costanzi-2026 selection-bias correction B_prj(R) (arXiv:2604.05833, App. C).

The correction multiplies the max halo lensing model:

    Sigma_corr(R) = B_prj(R) Sigma_max(R),     Sigma_max = max(Sigma_1h, Sigma_2h)

    B_prj(R) = A (R/R0)^alpha [1 + (R/R0)^gamma]^((beta - alpha)/gamma) + 1
    R0       = R_lambda(lob) (1 + z)           comoving Mpc/h

``A`` sets the amplitude, ``alpha``/``beta`` the inner/outer slopes, ``R0``
the transition scale and ``gamma`` its smoothness.  ``R`` must be comoving
Mpc/h like ``R0``.  The same form describes the bias on DeltaSigma(R); only
the parameter values differ (see :meth:`CostanziBprj.sigma` /
:meth:`CostanziBprj.dsigma`).

Datablock contract (section ``costanzi_bprj``)
----------------------------------------------
    A, alpha, beta, gamma     values file of the pipeline (sampled or fixed)
    lob_centers (n_lambda,)   published by this file's CosmoSIS module
    zob_centers (n_z,)        published by this file's CosmoSIS module

:func:`bprj_wall` evaluates B on the stacked (bin, R) wall from that
section -- bins z-major with richness fastest, radius fastest within a bin,
the NumCounts/Shear1h2hMax order -- so a consumer only supplies the radii::

    from systematics.costanzi_bprj.python.costanzi_bprj import bprj_wall
    shear_theory *= bprj_wall(block, r_perp)

Pipeline ini (defaults are the DES Y3 wall)::

    [costanzi_bprj]
    file = ${Y3_CLUSTER_CPP_DIR}/src/pipelines/systematics/costanzi_bprj/python/costanzi_bprj.py
    lob_centers = 25.0 37.5 52.5 130.0
    zob_centers = 0.275 0.425 0.575

Values file::

    [costanzi_bprj]
    A = 0.12
    alpha = 4.11
    beta = 0.18
    gamma = 1.82

C++ twin: ``../cpp/costanzi_bprj_t.hh`` (``y3_cluster::CostanziBprj_t``).
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Annotated, Any

import numpy as np
from pydantic import Field
from pydantic.dataclasses import dataclass

for _parent in Path(__file__).resolve().parents:
    if ((_parent / "shared" / "datablock_models.py").is_file()
            and (_parent / "systematics").is_dir()):
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break
else:
    raise ImportError("could not locate src/pipelines")

from shared.datablock_models import DEFAULT_LOB_CENTERS, R_lambda  # noqa: E402

SECTION = "costanzi_bprj"
PARAM_NAMES = ("A", "alpha", "beta", "gamma")
WALL_KEYS = ("lob_centers", "zob_centers")
DEFAULT_ZOB_CENTERS = (0.275, 0.425, 0.575)  # midpoints of the DES Y3 z wall


@dataclass(frozen=True)
class CostanziBprj:
    """B_prj(R | lob, z) double power law with smooth transition at R0."""

    A: float
    alpha: float
    beta: float
    gamma: Annotated[float, Field(gt=0)]  # divides (beta - alpha); > 0 = smooth transition

    # Paper best fits.  NOTE: arXiv:2604.05833 App. C quotes alpha = 0.92 for
    # Sigma; alpha = 0.1 here follows the owner's spec (2026-09-01) -- confirm
    # against the published version before sampling around it.
    @classmethod
    def sigma(cls) -> "CostanziBprj":
        """Best fit for Sigma(R)."""
        return cls(A=0.10, alpha=0.1, beta=-0.53, gamma=4.1)

    @classmethod
    def dsigma(cls) -> "CostanziBprj":
        """Best fit for DeltaSigma(R)."""
        return cls(A=0.12, alpha=4.11, beta=0.18, gamma=1.82)

    @classmethod
    def from_datablock(cls, block: Any, section: str = SECTION) -> "CostanziBprj":
        """Read ``A, alpha, beta, gamma`` from ``block[section, name]``."""
        return cls(**{name: float(block[section, name]) for name in PARAM_NAMES})

    @staticmethod
    def r0(lob, z):
        """Transition scale R0 = R_lambda(lob) (1 + z) [comoving Mpc/h]."""
        return R_lambda(lob) * (1.0 + np.asarray(z, dtype=float))

    def __call__(self, R, lob, z):
        """B_prj at comoving radius ``R`` [Mpc/h] for a cluster of richness ``lob`` at ``z``."""
        x = np.asarray(R, dtype=float) / self.r0(lob, z)
        return (self.A * x**self.alpha
                * (1.0 + x**self.gamma) ** ((self.beta - self.alpha) / self.gamma)
                + 1.0)


def bprj_wall(block: Any, R, section: str = SECTION) -> np.ndarray:
    """B_prj on the stacked (bin, R) wall, everything but ``R`` from the datablock.

    Reads ``A/alpha/beta/gamma`` and the bin grid ``lob_centers`` (n_lambda,)
    / ``zob_centers`` (n_z,) from ``section``.  Returns shape
    ``(n_z * n_lambda * n_R,)`` ordered z-major, richness fastest, radius
    fastest within a bin: ``index = (iz * n_lambda + il) * n_R + ir``.
    """
    model = CostanziBprj.from_datablock(block, section)
    lob = np.asarray(block[section, "lob_centers"], dtype=float).ravel()
    zob = np.asarray(block[section, "zob_centers"], dtype=float).ravel()
    R = np.asarray(R, dtype=float).ravel()
    return model(R[None, None, :], lob[None, :, None], zob[:, None, None]).ravel()


# ---------------------------------------------------------------------------
# CosmoSIS module: publish the wall grid (lob_centers, zob_centers) into the
# costanzi_bprj section; A/alpha/beta/gamma arrive there from the values file.
# ---------------------------------------------------------------------------
def setup(options):
    from cosmosis.datablock import option_section

    def read(name, default):
        try:
            return np.asarray(options.get_double_array_1d(option_section, name),
                              dtype=float)
        except Exception:
            return np.asarray(default, dtype=float)

    return {"lob_centers": read("lob_centers", DEFAULT_LOB_CENTERS),
            "zob_centers": read("zob_centers", DEFAULT_ZOB_CENTERS)}


def execute(block, config):
    for key in WALL_KEYS:
        block[SECTION, key] = config[key]
    return 0


def cleanup(config):
    return 0
