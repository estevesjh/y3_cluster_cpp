"""Sunayama-2023/Park selection-bias kernel ``1 + Pi(R)``.

The piecewise profile is calibrated per richness bin with a shared outer
slope.  The class uses the wall order supplied by the caller; ``lambda_bin``
is retained so a DataBlock can use explicit, non-contiguous bin labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


SECTION = "bsel_profile_sunayama23"


@dataclass(frozen=True)
class BselSunayama23:
    """Piecewise Sunayama-2023 multiplicative selection-bias profile."""

    pi0: np.ndarray
    r0: np.ndarray
    c: float
    lambda_bin: np.ndarray | None = None

    def __post_init__(self):
        pi0 = np.asarray(self.pi0, dtype=float).ravel()
        r0 = np.asarray(self.r0, dtype=float).ravel()
        if pi0.size == 0 or pi0.size != r0.size:
            raise ValueError("BselSunayama23 pi0/r0 vectors must have equal size")
        if np.any(~np.isfinite(pi0)) or np.any(~np.isfinite(r0)):
            raise ValueError("BselSunayama23 parameters must be finite")
        if np.any(r0 <= 0.0):
            raise ValueError("BselSunayama23 r0 values must be positive")
        if not np.isfinite(self.c):
            raise ValueError("BselSunayama23 c must be finite")
        bins = (np.arange(pi0.size, dtype=int) if self.lambda_bin is None
                else self._integer_labels(self.lambda_bin).ravel())
        if bins.size != pi0.size or np.unique(bins).size != bins.size:
            raise ValueError("BselSunayama23 lambda_bin must match pi0/r0")
        object.__setattr__(self, "pi0", pi0)
        object.__setattr__(self, "r0", r0)
        object.__setattr__(self, "lambda_bin", bins)

    @staticmethod
    def _integer_labels(values):
        labels = np.asarray(values, dtype=float)
        int_bounds = np.iinfo(np.intp)
        if (np.any(~np.isfinite(labels))
                or np.any(labels != np.trunc(labels))
                or np.any(labels < int_bounds.min)
                or np.any(labels >= -int_bounds.min)):
            raise ValueError("BselSunayama23 bin labels must be finite integers")
        return labels.astype(int)

    def _row(self, bin_index):
        requested = self._integer_labels(bin_index)
        positions = {int(value): i for i, value in enumerate(self.lambda_bin)}
        try:
            return np.asarray([positions[int(value)] for value in requested.ravel()],
                             dtype=int).reshape(requested.shape)
        except KeyError as exc:
            raise ValueError("BselSunayama23 bin is absent from lambda_bin") from exc

    def __call__(self, R, bin_index):
        """Evaluate the multiplicative selection factor ``1 + Pi(R)``."""
        R, bin_index = np.broadcast_arrays(np.asarray(R, dtype=float),
                                           np.asarray(bin_index))
        if np.any(~np.isfinite(R)) or np.any(R < 0.0):
            raise ValueError("BselSunayama23 requires R >= 0")
        row = self._row(bin_index)
        pi0 = self.pi0[row]
        r0 = self.r0[row]
        with np.errstate(divide="ignore", invalid="ignore"):
            outer = pi0 + self.c * np.log(R / r0)
        pi = np.where(R <= r0, pi0 * R / r0, outer)
        return 1.0 + pi

    def derivative(self, R, bin_index):
        """Evaluate the branch derivative for positive ``R``."""
        R, bin_index = np.broadcast_arrays(np.asarray(R, dtype=float),
                                           np.asarray(bin_index))
        if np.any(~np.isfinite(R)) or np.any(R <= 0.0):
            raise ValueError("BselSunayama23.derivative requires R > 0")
        row = self._row(bin_index)
        return np.where(R <= self.r0[row], self.pi0[row] / self.r0[row],
                        self.c / R)

    @classmethod
    def from_source(cls, source: Any, section: str = SECTION):
        """Read the per-bin calibration and shared ``c`` from a DataBlock."""
        try:
            bins = source[section, "lambda_bin"]
        except Exception:
            bins = None
        return cls(pi0=np.asarray(source[section, "pi0"], dtype=float),
                   r0=np.asarray(source[section, "r0"], dtype=float),
                   c=float(source[section, "c"]),
                   lambda_bin=bins)

    def validate_wall(self, bin_index):
        """Validate that every requested richness-bin label is available."""
        self._row(bin_index)
        return True
