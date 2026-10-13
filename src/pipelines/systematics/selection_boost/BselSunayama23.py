r"""Sunayama-2023 optical selection-bias correction kernel.

Sunayama et al. (2023), arXiv:2309.13025, Eq. (28), use the piecewise
contribution

.. math::

   \Pi(R) = \begin{cases}
       \Pi_0 R/R_0, & R \le R_0,\\
       \Pi_0 + c\ln(R/R_0), & R > R_0.
   \end{cases}

This implementation exposes the multiplicative correction
``B_sel(R) = 1 + Pi(R)`` used by the selected-profile calculation.  The
profile is calibrated per richness bin with a shared outer slope.  The class
uses the wall order supplied by the caller; ``lambda_bin`` is retained so a
DataBlock can use explicit, non-contiguous bin labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


SECTION = "bsel_profile_sunayama23"


@dataclass(frozen=True)
class BselSunayama23:
    r"""Evaluate the Sunayama et al. (2023) correction ``B_sel(R)``.

    .. math::

       \Pi(R) = \begin{cases}
           \Pi_0 R/R_0, & R \le R_0,\\
           \Pi_0 + c\ln(R/R_0), & R > R_0,
       \end{cases}
       \qquad \mathcal{B}_{\rm sel}(R) = 1 + \Pi(R).
    """

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
        has_explicit = self.lambda_bin is not None
        bins = (np.arange(pi0.size, dtype=int) if not has_explicit
                else self._integer_labels(self.lambda_bin).ravel())
        if bins.size != pi0.size or np.unique(bins).size != bins.size:
            raise ValueError("BselSunayama23 lambda_bin must match pi0/r0")
        object.__setattr__(self, "pi0", pi0)
        object.__setattr__(self, "r0", r0)
        object.__setattr__(self, "lambda_bin", bins)
        object.__setattr__(self, "_has_explicit_bins", has_explicit)

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
        if self.pi0.size == 1 and not getattr(self, "_has_explicit_bins", True):
            return np.zeros(requested.shape, dtype=int)
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
        """Read the calibration parameters and shared ``c`` from a DataBlock."""
        has_val = source.has if hasattr(source, "has") else (
            source.has_value if hasattr(source, "has_value") else (lambda s, k: (s, k) in source))

        if not has_val(section, "c") and not has_val(section, "pi0"):
            for cand in ("boost_selection_sunayama23", "bsel_profile_sunayama23"):
                if has_val(cand, "c") or has_val(cand, "pi0"):
                    section = cand
                    break

        def _get_scalar(k):
            if hasattr(source, "scalar"):
                return float(source.scalar(section, k))
            return float(source[section, k])

        def _get_arr_or_scalar(k):
            if has_val(section, k):
                if hasattr(source, "array") and hasattr(source, "scalar"):
                    try:
                        return np.atleast_1d(source.array(section, k))
                    except Exception:
                        return np.atleast_1d(source.scalar(section, k))
                val = source[section, k]
                return np.atleast_1d(np.asarray(val, dtype=float))
            vals = []
            b = 0
            while has_val(section, f"{k}_{b}"):
                vals.append(_get_scalar(f"{k}_{b}"))
                b += 1
            if vals:
                return np.asarray(vals, dtype=float)
            raise KeyError(f"Parameter '{k}' not found in section '{section}'")

        pi0 = _get_arr_or_scalar("pi0")
        r0 = _get_arr_or_scalar("r0")
        c = _get_scalar("c")
        bins = None
        if has_val(section, "lambda_bin"):
            if hasattr(source, "array"):
                bins = source.array(section, "lambda_bin")
            else:
                bins = np.asarray(source[section, "lambda_bin"], dtype=float)

        return cls(pi0=pi0, r0=r0, c=c, lambda_bin=bins)

    def validate_wall(self, bin_index):
        """Validate that every requested richness-bin label is available."""
        self._row(bin_index)
        return True
