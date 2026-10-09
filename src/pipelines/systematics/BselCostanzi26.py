"""Costanzi-2026 Bsel(R) kernel for the raw surface-density profile.

The model follows Appendix C of arXiv:2604.05833.  ``R``, ``R0`` and the
richness-radius convention are comoving Mpc/h.  The class is deliberately
separate from ``CostanziBprj``: this kernel supplies both ``Bsel`` and its
derivative to the non-local selected-profile calculation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


SECTION = "bsel_profile_costanzi26"
PARAM_NAMES = ("A", "alpha", "beta", "gamma")


def _r_lambda(lob):
    return (np.asarray(lob, dtype=float) / 100.0) ** 0.2


@dataclass(frozen=True)
class BselCostanzi26:
    """Smooth Costanzi-2026 selection-bias profile ``Bsel(R)``."""

    A: float
    alpha: float
    beta: float
    gamma: float

    def __post_init__(self):
        values = (self.A, self.alpha, self.beta, self.gamma)
        if not np.all(np.isfinite(values)):
            raise ValueError("BselCostanzi26 parameters must be finite")
        if self.gamma <= 0.0:
            raise ValueError("BselCostanzi26 gamma must be positive")

    @staticmethod
    def r0(lob, z):
        """Return ``R_lambda(lob) * (1 + z)`` in comoving Mpc/h."""
        lob, z = np.broadcast_arrays(np.asarray(lob, dtype=float),
                                     np.asarray(z, dtype=float))
        if (np.any(~np.isfinite(lob)) or np.any(~np.isfinite(z))
                or np.any(lob <= 0.0) or np.any(z <= -1.0)):
            raise ValueError("BselCostanzi26 requires lob > 0 and z > -1")
        return _r_lambda(lob) * (1.0 + z)

    def __call__(self, R, lob, z):
        """Evaluate ``Bsel(R | lob, z)`` on broadcastable inputs."""
        R, lob, z = np.broadcast_arrays(np.asarray(R, dtype=float),
                                        np.asarray(lob, dtype=float),
                                        np.asarray(z, dtype=float))
        if np.any(~np.isfinite(R)) or np.any(R < 0.0):
            raise ValueError("BselCostanzi26 requires R >= 0")
        r0 = self.r0(lob, z)
        x = np.divide(R, r0, out=np.zeros_like(R), where=r0 != 0.0)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            value = (1.0 + self.A * x**self.alpha
                     * (1.0 + x**self.gamma)
                     ** ((self.beta - self.alpha) / self.gamma))
        return np.where(R == 0.0, 1.0, value)

    def derivative(self, R, lob, z):
        """Evaluate the analytic derivative ``dBsel/dR`` for ``R > 0``."""
        R, lob, z = np.broadcast_arrays(np.asarray(R, dtype=float),
                                        np.asarray(lob, dtype=float),
                                        np.asarray(z, dtype=float))
        if np.any(~np.isfinite(R)) or np.any(R <= 0.0):
            raise ValueError("BselCostanzi26.derivative requires R > 0")
        r0 = self.r0(lob, z)
        x = R / r0
        B_minus_one = self(R, lob, z) - 1.0
        return B_minus_one / R * ((self.alpha + self.beta * x**self.gamma)
                                  / (1.0 + x**self.gamma))

    @classmethod
    def from_source(cls, source: Any, section: str = SECTION):
        """Read ``A``, ``alpha``, ``beta`` and ``gamma`` from a DataBlock."""
        return cls(**{name: float(source[section, name])
                      for name in PARAM_NAMES})

    @staticmethod
    def validate_wall(lob, z):
        """Validate richness and redshift wall vectors used by the kernel."""
        lob = np.asarray(lob, dtype=float).ravel()
        z = np.asarray(z, dtype=float).ravel()
        if lob.size == 0 or z.size == 0:
            raise ValueError("BselCostanzi26 wall vectors cannot be empty")
        if np.any(~np.isfinite(lob)) or np.any(lob <= 0.0):
            raise ValueError("BselCostanzi26 wall richness must be positive")
        if np.any(~np.isfinite(z)) or np.any(z <= -1.0):
            raise ValueError("BselCostanzi26 wall redshifts must exceed -1")
        return lob, z
