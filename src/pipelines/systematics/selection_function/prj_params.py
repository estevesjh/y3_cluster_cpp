"""Frozen Costanzi-2026 projection-effects EMG coefficients.

The 8 EMG coefficients (a_tau, b_tau, a_mu, b_mu, a_sig, b_sig, a_fprj,
b_fprj) live on a fixed 15-node z grid spanning [0.10, 0.80] in steps
of 0.05.  They are the source of truth for the P(lob | ltr, z) kernel
used by `bsel.py`, `sel_function.py`, and the C++ `PLOB_LTR_t` model.

Consumers should import :class:`PrjParams` rather than reload arrays
from disk:

    from systematics.selection_function.prj_params import PrjParams
    params = PrjParams.default()
    raw    = params.as_dict()           # legacy serialization helper
    splines = params.splines()          # sel_function.py-style ius dict
    density = params.p_lob_given_ltr(lob, ltr, z)
    cdf     = params.cdf_lob(lob, ltr, z)

A CosmoSIS module shim at the bottom (``setup``/``execute``/``cleanup``)
publishes the same arrays into the ``plob_ltr_params`` datablock section
so the C++ ``plob_ltr_t.hh`` consumers (and any other datablock reader)
keep working without an external loader.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from scipy.special import erf, erfc, erfcx

_Z = (
    0.1, 0.15000000000000002, 0.2, 0.25, 0.30000000000000004, 0.35, 0.4,
    0.45000000000000007, 0.5, 0.55, 0.6, 0.65, 0.7000000000000001, 0.75, 0.8,
)
_A_TAU  = (0.442166204, 0.518481914, 0.568603921, 0.477236476, 0.520278762,
           0.579677963, 0.555190684, 0.672770912, 0.478623398, 0.549620102,
           0.480748319, 0.578522658, 0.671535976, 0.505164749, 0.58955537)
_B_TAU  = (1.18217964, 1.47500549, 2.02559386, 1.38676832, 1.32616774,
           1.67202719, 1.51274293, 1.95006009, 1.18386488, 1.36976846,
           1.05592613, 1.30622288, 1.86155835, 1.11381729, 1.55322604)
_A_MU   = (-0.364318121, -0.23432317, -0.656963951, -0.922603116, -0.846333985,
           -1.08055911, -1.18280131, -1.06499383, -1.39776432, -1.73727975,
           -1.89795303, -2.06646039, -2.11221354, -2.33692459, -2.68990643)
_B_MU   = (1.05327406, 1.02863546, 1.00438937, 1.02999661, 1.07731829,
           1.00753429, 1.02754167, 0.984235605, 1.0482203, 1.04839327,
           1.0613957, 1.06299404, 1.07220168, 1.06769774, 1.17169173)
_A_SIG  = (0.426034095, 0.399889486, 0.36483048, 0.400337495, 0.412239723,
           0.352441888, 0.386464927, 0.346387159, 0.36257126, 0.414359103,
           0.396064175, 0.392899807, 0.412642261, 0.411313571, 0.516942252)
_B_SIG  = (1.03790465, 1.07103564, 1.19439168, 1.15625107, 1.21959668,
           1.36154101, 1.29767428, 1.38632059, 1.37829562, 1.1874275,
           1.33730986, 1.43217579, 1.36804085, 1.49303631, 1.25987859)
_A_FPRJ = (9.77919696, 9.84852044, 9.85900036, 9.66653259, 9.80163756,
           9.57966267, 9.89274466, 9.53433063, 8.20542109, 9.93404856,
           9.89089118, 7.86185779, 8.44425519, 8.33876275, 9.83602795)
_B_FPRJ = (0.259342564, 0.392345959, 0.938787218, 0.76558764, 0.438594201,
           0.987913546, 0.819593953, 0.996036691, 0.906686505, 0.997565086,
           0.99306453, 0.984970784, 0.951840743, 0.987248745, 0.832798309)


COEFF_NAMES = (
    "a_tau", "b_tau", "a_mu", "b_mu",
    "a_sig", "b_sig", "a_fprj", "b_fprj",
)


def _arr(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


@dataclass(frozen=True)
class PrjParams:
    """EMG coefficients and analytical PLOB_LTR density/CDF model."""

    z: np.ndarray
    a_tau: np.ndarray
    b_tau: np.ndarray
    a_mu: np.ndarray
    b_mu: np.ndarray
    a_sig: np.ndarray
    b_sig: np.ndarray
    a_fprj: np.ndarray
    b_fprj: np.ndarray

    # cache for splines() -- populated lazily; bypasses the frozen guard
    # via object.__setattr__ since dataclasses(frozen=True) won't let us
    # assign normally.
    _splines_cache: dict = field(default=None, repr=False, compare=False)

    @classmethod
    def default(cls) -> "PrjParams":
        return cls(
            z=_arr(_Z),
            a_tau=_arr(_A_TAU),   b_tau=_arr(_B_TAU),
            a_mu=_arr(_A_MU),     b_mu=_arr(_B_MU),
            a_sig=_arr(_A_SIG),   b_sig=_arr(_B_SIG),
            a_fprj=_arr(_A_FPRJ), b_fprj=_arr(_B_FPRJ),
        )

    @classmethod
    def from_dict(cls, d: Mapping) -> "PrjParams":
        return cls(
            z=_arr(d["z"]),
            **{k: _arr(d[k]) for k in COEFF_NAMES},
        )

    @classmethod
    def from_datablock(cls, block, section: str = "plob_ltr_params") -> "PrjParams":
        return cls(
            z=_arr(block[section, "z"]),
            **{k: _arr(block[section, k]) for k in COEFF_NAMES},
        )

    @classmethod
    def from_source(cls, source, section: str = "plob_ltr_params") -> "PrjParams":
        """Read the coefficient table through a shared datablock adapter.

        ``source`` may wrap either a live CosmoSIS datablock or a saved
        test-sampler dump. Keeping this constructor here lets callers pass a
        typed ``PrjParams`` object through the numerical pipeline instead of
        rebuilding the table as a string-keyed dictionary.
        """
        return cls(
            z=_arr(source.array(section, "z")),
            **{k: _arr(source.array(section, k)) for k in COEFF_NAMES},
        )

    @classmethod
    def from_source_or_default(
        cls, source, section: str = "plob_ltr_params"
    ) -> "PrjParams":
        """Read a complete datablock table, or use the frozen default.

        The normal pipeline may publish ``plob_ltr_params`` explicitly, while
        lightweight fixtures and older pipelines may omit the section. A
        partial table is not mixed with the default: it falls back as a whole
        so every EMG coefficient always comes from one consistent model.
        """
        required_keys = ("z",) + COEFF_NAMES
        if all(source.has(section, key) for key in required_keys):
            return cls.from_source(source, section)
        return cls.default()

    def as_dict(self) -> dict[str, np.ndarray]:
        out = {"z": self.z}
        for k in COEFF_NAMES:
            out[k] = getattr(self, k)
        return out

    def interp_linear(self, name: str, z_query) -> np.ndarray:
        if name not in COEFF_NAMES:
            raise KeyError(f"unknown coefficient {name!r}; expected one of {COEFF_NAMES}")
        query = np.asarray(z_query, dtype=float)
        values = np.interp(
            query.reshape(-1), self.z, getattr(self, name)
        )
        return values.reshape(query.shape)

    @staticmethod
    def _align_redshift(ltr, z):
        """Place a one-dimensional redshift grid on the true-richness axis.

        The fast selection layout is ``(lnM, z, ltr)``. NumPy would otherwise
        align a bare ``(n_z,)`` array with the last axis, so reshape it before
        the coefficient interpolation. Scalar redshifts need no adjustment.
        """
        if z.ndim != 1 or ltr.ndim <= 1:
            return z
        if ltr.shape[-2] == z.size:
            axis = ltr.ndim - 2
        else:
            matches = [index for index, size in enumerate(ltr.shape)
                       if size == z.size]
            axis = matches[0] if matches else None
        if axis is None:
            return z
        shape = tuple(z.size if index == axis else 1
                      for index in range(ltr.ndim))
        return z.reshape(shape)

    def emg_parameters(self, ltr, z):
        """Return ``(mu, sigma, tau, fprj)`` for true richness and redshift.

        These four equations are shared by the EMG density and its CDF. They
        live on ``PrjParams`` so callers do not each maintain a second copy of
        the projection calibration formulas.
        """
        ltr = np.asarray(ltr, dtype=float)
        z = np.asarray(z, dtype=float)
        z = self._align_redshift(ltr, z)
        safe_ltr = np.maximum(ltr, 1.0e-12)

        # The coefficient table is linear in redshift and clamped at its
        # endpoints, matching the C++ Interp1D convention.
        a_tau = self.interp_linear("a_tau", z)
        b_tau = self.interp_linear("b_tau", z)
        a_mu = self.interp_linear("a_mu", z)
        b_mu = self.interp_linear("b_mu", z)
        a_sig = self.interp_linear("a_sig", z)
        b_sig = self.interp_linear("b_sig", z)
        a_fprj = self.interp_linear("a_fprj", z)
        b_fprj = self.interp_linear("b_fprj", z)

        mu = a_mu + b_mu * ltr
        sigma = np.maximum(b_sig * safe_ltr**a_sig, 1.0e-12)
        tau = np.maximum(b_tau / safe_ltr**a_tau, 1.0e-12)
        fprj = np.clip(
            b_fprj / (1.0 + np.exp(-ltr))**a_fprj,
            0.0,
            1.0,
        )
        return mu, sigma, tau, fprj

    @staticmethod
    def cdf_from_parameters(lob, mu, sigma, tau, fprj):
        """Evaluate the analytical Gaussian/EMG mixture CDF.

        ``mu``, ``sigma``, ``tau``, and ``fprj`` are assumed to have already
        been evaluated at ``(ltr, z)``. Keeping this low-level form lets the
        optimized selection path reuse its own fast parameter arrays while
        the model formula remains defined in one place.

        For ``u = (tau*sigma - x)/sqrt(2)``, the EMG tail is evaluated as

        ``0.5 * erfcx(abs(u)) * exp(-x**2 / 2)`` for ``u >= 0`` and
        ``exp(A) - tail`` for ``u < 0``,

        where ``x = (lob - mu)/sigma`` and
        ``A = -tau*(lob - mu) + (tau*sigma)**2/2``.
        """
        lob = np.asarray(lob, dtype=float)
        mu = np.asarray(mu, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        tau = np.asarray(tau, dtype=float)
        fprj = np.asarray(fprj, dtype=float)

        standardized = (lob - mu) / sigma
        u = (tau * sigma - standardized) / np.sqrt(2.0)
        tail = (
            0.5 * erfcx(np.abs(u))
            * np.exp(-0.5 * standardized**2)
        )
        negative_u = u < 0.0
        exponent = (
            -tau * (lob - mu)
            + 0.5 * (tau * sigma)**2
        )
        np.clip(exponent, -700.0, 700.0, out=exponent)
        tail = np.where(negative_u, np.exp(exponent) - tail, tail)

        result = 0.5 * (1.0 + erf(standardized / np.sqrt(2.0)))
        result -= fprj * tail
        return np.clip(result, 0.0, 1.0)

    def cdf_lob(self, lob, ltr, z):
        """Evaluate ``P(lambda_ob <= lob | ltr, z)`` analytically."""
        mu, sigma, tau, fprj = self.emg_parameters(ltr, z)
        return self.cdf_from_parameters(lob, mu, sigma, tau, fprj)

    def p_lob_given_ltr(self, lob, ltr, z):
        """Evaluate the analytical EMG density ``P(lob | ltr, z)``.

        The coefficient table defines the four EMG parameters at true
        richness ``ltr`` and redshift ``z``:

        ``mu    = a_mu  + b_mu  * ltr``
        ``sigma = b_sig * ltr**a_sig``
        ``tau   = b_tau / ltr**a_tau``
        ``fprj  = b_fprj / (1 + exp(-ltr))**a_fprj``.

        The returned density is the analytical Gaussian/EMG mixture:

        ``P(lob | ltr, z) = (1 - fprj) * Gaussian(lob; mu, sigma)``
        ``                    + fprj * EMG(lob; mu, sigma, tau)``.

        All arguments broadcast through NumPy. In particular, ``lob`` and
        ``z`` may be scalars while ``ltr`` is a Gauss-Legendre vector.
        Redshift coefficients use the same linear, edge-clamped lookup as the
        C++ ``Interp1D`` convention.
        """
        lob = np.asarray(lob, dtype=float)
        mu, sigma, tau, fprj = self.emg_parameters(ltr, z)

        # Gaussian core of the projection model.
        standardized = (lob - mu) / sigma
        gaussian = (
            np.exp(-0.5 * standardized**2)
            / (sigma * np.sqrt(2.0 * np.pi))
        )

        # Exponentially modified Gaussian (EMG) tail. Clip only the exponent
        # used by exp so extreme valid inputs do not overflow the vectorized
        # calculation; erfc itself remains analytical and unapproximated.
        exp_argument = 0.5 * tau * (
            2.0 * mu + tau * sigma**2 - 2.0 * lob
        )
        erfc_argument = (
            mu + tau * sigma**2 - lob
        ) / (np.sqrt(2.0) * sigma)
        emg = (
            0.5 * tau
            * np.exp(np.clip(exp_argument, -700.0, 700.0))
            * erfc(erfc_argument)
        )
        return (1.0 - fprj) * gaussian + fprj * emg

    def splines(self) -> dict:
        """Lazily build and cache `InterpolatedUnivariateSpline(k=1, ext=3)`
        for each coefficient."""
        if self._splines_cache is None:
            from scipy.interpolate import InterpolatedUnivariateSpline as ius
            cache = {k: ius(self.z, getattr(self, k), k=1, ext=3)
                     for k in COEFF_NAMES}
            object.__setattr__(self, "_splines_cache", cache)
        return self._splines_cache


# ---------------------------------------------------------------------------
# CosmoSIS module shim: replaces src/modules/plob_ltr_loader/plob_ltr_loader.py
# ---------------------------------------------------------------------------
def setup(_options):
    return PrjParams.default()


def execute(block, params: PrjParams):
    block["plob_ltr_params", "z"] = params.z
    for k in COEFF_NAMES:
        block["plob_ltr_params", k] = getattr(params, k)
    return 0


def cleanup(_params):
    return 0
