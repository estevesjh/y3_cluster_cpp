"""Tagged selection-boost kernels and Shear1h2hMax consumers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .BselCostanzi26 import BselCostanzi26, SECTION as COSTANZI_SECTION
from .BselSunayama23 import BselSunayama23, SECTION as SUNAYAMA_SECTION


@dataclass(frozen=True)
class BselModels:
    """Dispatch one explicit Bsel tag to its validated kernel."""

    tag: str
    model: BselCostanzi26 | BselSunayama23

    def __post_init__(self):
        if self.tag not in ("Costanzi26", "Sunayama23"):
            raise ValueError("BselModels tag must be Costanzi26 or Sunayama23")
        if self.tag == "Costanzi26" and not isinstance(self.model, BselCostanzi26):
            raise TypeError("Costanzi26 requires a BselCostanzi26 model")
        if self.tag == "Sunayama23" and not isinstance(self.model, BselSunayama23):
            raise TypeError("Sunayama23 requires a BselSunayama23 model")

    @classmethod
    def from_source(cls, source: Any, tag: str, section: str | None = None):
        """Construct the tagged kernel from a CosmoSIS DataBlock section."""
        if tag == "Costanzi26":
            return cls(tag, BselCostanzi26.from_source(
                source, COSTANZI_SECTION if section is None else section))
        if tag == "Sunayama23":
            return cls(tag, BselSunayama23.from_source(
                source, SUNAYAMA_SECTION if section is None else section))
        raise ValueError("BselModels tag must be Costanzi26 or Sunayama23")

    def __call__(self, R, *, bin_index, lob=None, z=None):
        """Evaluate the selected Bsel model with an explicit wall context."""
        if self.tag == "Costanzi26":
            if lob is None or z is None:
                raise ValueError("Costanzi26 requires lob and z")
            return self.model(R, lob, z)
        return self.model(R, bin_index)

    def derivative(self, R, *, bin_index, lob=None, z=None):
        """Evaluate the selected model derivative."""
        if self.tag == "Costanzi26":
            if lob is None or z is None:
                raise ValueError("Costanzi26 requires lob and z")
            return self.model.derivative(R, lob, z)
        return self.model.derivative(R, bin_index)


@dataclass(frozen=True)
class BselDeltaSigmaOperator:
    r"""Non-local optical selection-bias operator on excess surface density.

    .. math::

       \Delta\Sigma_{\rm sel}(R) = \mathcal{B}_{\rm sel}(R)\,\Delta\Sigma(R)
           - \frac{M_{\rm sel}(R)}{\pi R^2}

    where the mass-weighted gradient integral :math:`M_{\rm sel}(R)` is:

    .. math::

       \frac{M_{\rm sel}(R)}{\pi R^2} = R \int_0^1 u^2\,\bar{\Sigma}(<u R)\,
           \left.\frac{d\mathcal{B}_{\rm sel}}{dR'}\right|_{R'=u R}\,du.
    """

    model: BselModels
    n_gl: int = 20

    def __post_init__(self):
        if self.n_gl < 1:
            raise ValueError("BselDeltaSigmaOperator n_gl must be >= 1")

    @property
    def u_nodes(self) -> np.ndarray:
        u_raw, _ = np.polynomial.legendre.leggauss(self.n_gl)
        return 0.5 * (u_raw + 1.0)

    @property
    def weights(self) -> np.ndarray:
        _, w_raw = np.polynomial.legendre.leggauss(self.n_gl)
        return 0.5 * w_raw

    @classmethod
    def from_source(cls, source: Any, tag: str, section: str | None = None, n_gl: int = 20):
        return cls(BselModels.from_source(source, tag, section), n_gl=n_gl)

    def apply(self, delta_sigma, mean_sigma_fn, R, *, bin_index, lob=None, z=None):
        R_arr = np.asarray(R, dtype=float)
        if np.any(R_arr <= 0.0):
            raise ValueError("BselDeltaSigmaOperator requires R > 0")

        b_val = self.model(R_arr, bin_index=bin_index, lob=lob, z=z)

        u_nodes = self.u_nodes
        weights = self.weights
        m_sel_over_pi_r2 = np.zeros_like(b_val, dtype=float)

        for ui, wi in zip(u_nodes, weights):
            r_prime = ui * R_arr
            d_bsel = self.model.derivative(r_prime, bin_index=bin_index, lob=lob, z=z)
            bar_sig = mean_sigma_fn(r_prime)
            m_sel_over_pi_r2 = m_sel_over_pi_r2 + (wi * (ui * ui)) * bar_sig * d_bsel

        m_sel_over_pi_r2 = m_sel_over_pi_r2 * R_arr
        return b_val * delta_sigma - m_sel_over_pi_r2

    def __call__(self, delta_sigma, R, *, bin_index, lob=None, z=None):
        """Multiplicative fallback if no mean_sigma_fn is supplied."""
        return delta_sigma * self.model(R, bin_index=bin_index, lob=lob, z=z)



@dataclass(frozen=True)
class Shear1h2hMaxSelCostanzi26:
    """Multiply a local max-model profile by the Costanzi26 selection boost."""

    model: BselCostanzi26

    def __call__(self, profile, R, *, bin_index, lob, z):
        del bin_index
        return profile * self.model(R, lob, z)


@dataclass(frozen=True)
class Shear1h2hMaxSelSunayama23:
    """Multiply a local max-model profile by the Sunayama23 selection boost."""

    model: BselSunayama23

    def __call__(self, profile, R, *, bin_index, lob=None, z=None):
        del lob, z
        return profile * self.model(R, bin_index)


def selected_shear1h2h_max_consumer(source: Any, tag: str,
                                     section: str | None = None):
    """Load the selected consumer without changing the max-model wall API."""
    model = BselModels.from_source(source, tag, section)
    if tag == "Costanzi26":
        return Shear1h2hMaxSelCostanzi26(model.model)
    return Shear1h2hMaxSelSunayama23(model.model)
