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

In a pipeline the parameters live in the CosmoSIS values file and are read
with :meth:`CostanziBprj.from_datablock`::

    [costanzi_bprj]
    A = 0.10
    alpha = 0.1
    beta = -0.53
    gamma = 4.1

C++ twin: ``../cpp/costanzi_bprj_t.hh`` (``y3_cluster::CostanziBprj_t``).
"""
from __future__ import annotations

from typing import Annotated, Any

import numpy as np
from pydantic import Field
from pydantic.dataclasses import dataclass

from shared.datablock_models import R_lambda

PARAM_NAMES = ("A", "alpha", "beta", "gamma")


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
    def from_datablock(cls, block: Any, section: str = "costanzi_bprj") -> "CostanziBprj":
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
