"""The fixed unit-profile family behind the radial_series tables.

Conventions are pinned to ``src/models/nfw_dsigma_mis.hh`` exactly
(concentration c = 4, the UNIFIED rho_m reference density
``haloModel/rho_m_ref`` for BOTH the halo boundary and the amplitude —
2026-08-24 convention decision — Wright & Brainerd normalisation, gamma
miscentering kernel):

IMPORTANT MODEL LIMITATION: ``c = 4`` is fixed for every mass and redshift.
This family therefore has no concentration--mass or concentration--redshift
evolution.  In particular, ``r_s(M) = r_200(M) / 4`` and the dimensionless
shape ``u`` is reused for all M and z.  This is the assumption that makes an
offline table reusable, but it also means that the table cannot reproduce a
production profile built with a varying ``c(M, z)`` relation.  Exact redshift
weight contraction in the consumer does not remove this limitation.

    DSigma_cen(R, M)        = A0(y) * u_cen(x)
    DSigma_mis(R, r_mis, M) = A0(y) * u_mis(x, x_mis)

    y     = ln r_s(M),  r_s = r_200 / c,  r_200 = [3M/(800 pi rho_ref)]^(1/3)
    x     = R e^-y,     x_mis = r_mis e^-y
    A0(y) = 2 e^y delta_c rho_ref * 1e-12        [Msun/(h pc^2) per unit u]

``u_mis`` is the stored ``exp(ln u)`` of the production look-up table
``data/nfw_off_center/table_1000_1e-03_5e+03_log_deltasigma_gamma.txt``;
``u_cen(x) = g(x)/2`` with g the closed-form Wright & Brainerd (2000)
shape function, which is the table family's own centred limit (the study
docs/shear1h_radial_factorization.tex cross-checked g against an
independent line-of-sight projection to 1.6e-9).

Everything sample-dependent enters through ``rho_ref``
(haloModel/rho_m_ref, shared by BOTH components), the mixture weight
``f_mis``, and the query coordinates — never through the shape of
``u``; that separation is what licenses the offline U_ell tables
(src/pipelines/des_y3/README.md, "Offline unit-profile table").
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline

# Strong approximation shared by the offline table and C++ evaluator:
# concentration has no mass evolution and no redshift evolution.  Do not
# silently compare this fixed-c family with haloModel/dSigma_nfw, whose
# production concentration varies with M and z.
CONC = 4.0
RHOC = 2.77533742639e11
MPC2_TO_PC2 = 1.0e-12
DELTA_C = (200.0 * CONC**3 / 3.0) / (np.log(1.0 + CONC) - CONC / (1.0 + CONC))

GAMMA_TABLE_FILES = (
    "table_1000_1e-03_5e+03_gamma_logx.txt",
    "table_1000_1e-03_5e+03_gamma_logxmis.txt",
    "table_1000_1e-03_5e+03_log_deltasigma_gamma.txt",
)


def repo_root():
    for p in Path(__file__).resolve().parents:
        if (p / "data" / "nfw_off_center").is_dir():
            return p
    raise FileNotFoundError(
        "nfw_profile_family: cannot locate the repository data/ directory")


def r_s_of_lnM(lnM, rho_ref):
    """Scale radius r_s(M) [Mpc/h] at the fixed c=4 approximation.

    rho_ref = haloModel/rho_m_ref (unified rho_m convention, 2026-08-24):
    the same density the centred dSigma_nfw table is built with drives
    the boundary r_200 = [3M/(800 pi rho_ref)]^(1/3)."""
    return np.cbrt(3.0 * np.exp(np.asarray(lnM, dtype=float))
                   / (800.0 * np.pi * rho_ref)) / CONC


def y_of_lnM(lnM, rho_ref):
    return np.log(r_s_of_lnM(lnM, rho_ref))


def lnM_of_y(y, rho_ref):
    return np.log(800.0 * np.pi * rho_ref / 3.0) + 3.0 * (np.asarray(y) +
                                                          np.log(CONC))


def A0_of_y(y, rho_ref):
    """Fixed-c amplitude A0(y) = 2 e^y delta_c rho_ref * 1e-12."""
    return 2.0 * np.exp(np.asarray(y, dtype=float)) * DELTA_C * rho_ref \
        * MPC2_TO_PC2


# Taylor coefficients of the Sigma shape f(x) about x = 1 (mpmath, dps=60);
# used for |x-1| < 0.01 where the closed-form branches lose ~7 digits.
F_TAYLOR_1 = np.array([
    0.66666666666666667, -0.8, 0.74285714285714286, -0.63492063492063492,
    0.52813852813852814, -0.43822843822843823, 0.36705516705516706,
    -0.31197038255861785, 0.26945809608348308, -0.23638281223420542,
    0.21028668464393326])


def sigma_shape(x):
    """NFW surface-density shape f(x) = Sigma / (r_s delta_c rho_c).

    Wright & Brainerd (2000) closed form with a degree-10 Taylor branch
    at |x-1| < 0.01; relative accuracy <= ~1e-13 over x in [1e-5, 1e4].
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    near = np.abs(x - 1.0) < 0.01
    lo = (x < 1.0) & ~near
    hi = ~(lo | near)
    xl = x[lo]
    out[lo] = 2.0 / (xl**2 - 1.0) * (
        1.0 - 2.0 / np.sqrt(1.0 - xl**2)
        * np.arctanh(np.sqrt((1.0 - xl) / (1.0 + xl))))
    xh = x[hi]
    out[hi] = 2.0 / (xh**2 - 1.0) * (
        1.0 - 2.0 / np.sqrt(xh**2 - 1.0)
        * np.arctan(np.sqrt((xh - 1.0) / (xh + 1.0))))
    if near.any():
        d = x[near] - 1.0
        acc = np.zeros_like(d)
        for c in F_TAYLOR_1[::-1]:
            acc = c + d * acc
        out[near] = acc
    return out


def sigma_shape_mp(x, mp):
    """mpmath scalar f(x) for high-precision offline work."""
    x = mp.mpf(x)
    one = mp.mpf(1)
    if x == 1:
        return mp.mpf(2) / 3
    if x < 1:
        return 2 / (x * x - one) * (
            one - 2 / mp.sqrt(one - x * x)
            * mp.atanh(mp.sqrt((one - x) / (one + x))))
    return 2 / (x * x - one) * (
        one - 2 / mp.sqrt(x * x - one)
        * mp.atan(mp.sqrt((x - one) / (x + one))))


def u_cen_mp(x, mp):
    """mpmath scalar u_cen(x) = g(x)/2, stable at any x > 0.

    The float64 g_shape loses up to ~7 digits below x ~ 1e-3 through
    cancellation; the offline generator needs the centred profile on grid
    nodes to near machine precision, so it evaluates this instead.
    """
    return g_shape_mp(mp.mpf(x), mp) / 2


def g_shape(x):
    """Wright & Brainerd (2000) NFW Delta-Sigma shape function g(x).

    Same expression the radial-factorization study validated to 1.6e-9
    against a direct projection of the 3-D NFW density.
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    lo, hi = x < 1.0 - 1e-6, x > 1.0 + 1e-6
    mid = ~(lo | hi)
    xl = x[lo]
    yl = np.sqrt(1.0 - xl**2)
    al = np.arctanh(np.sqrt((1.0 - xl) / (1.0 + xl)))
    out[lo] = (8 * al / (xl**2 * yl) + 4 / xl**2 * np.log(xl / 2)
               - 2 / (xl**2 - 1) + 4 * al / ((xl**2 - 1) * yl))
    xh = x[hi]
    yh = np.sqrt(xh**2 - 1.0)
    ah = np.arctan(np.sqrt((xh - 1.0) / (1.0 + xh)))
    out[hi] = (8 * ah / (xh**2 * yh) + 4 / xh**2 * np.log(xh / 2)
               - 2 / (xh**2 - 1) + 4 * ah / (xh**2 - 1)**1.5)
    out[mid] = 10.0 / 3.0 + 4.0 * np.log(0.5)
    return out


def u_cen(x):
    """Centred unit profile: A0(y) u_cen = DSigma_NFW at fixed c=4."""
    return 0.5 * g_shape(x)


def g_shape_mp(x, mp):
    """mpmath scalar g(x) for the high-precision offline generator.

    Needs mp.dps well above 20 near x = 1: the branch formulas cancel
    ~2 digits per decade of |x-1|, and the constant fallback only
    covers |x-1| < 1e-20 (so mp.taylor's tiny finite-difference steps
    stay on the analytic branches).
    """
    one = mp.mpf(1)
    if abs(x - 1) < mp.mpf("1e-20"):
        return mp.mpf(10) / 3 + 4 * mp.log(mp.mpf(1) / 2)
    if x < 1:
        y = mp.sqrt(one - x * x)
        a = mp.atanh(mp.sqrt((one - x) / (one + x)))
        return (8 * a / (x * x * y) + 4 / (x * x) * mp.log(x / 2)
                - 2 / (x * x - one) + 4 * a / ((x * x - one) * y))
    y = mp.sqrt(x * x - one)
    a = mp.atan(mp.sqrt((x - one) / (one + x)))
    return (8 * a / (x * x * y) + 4 / (x * x) * mp.log(x / 2)
            - 2 / (x * x - one) + 4 * a / (x * x - one) ** mp.mpf(1.5))


class MisTable:
    """Smooth (quintic-spline) view of the gamma-kernel ln u source table.

    Axes are natural-log x and x_mis exactly as stored; queries outside
    the domain are clamped, matching Interp2D::clamp in the production
    reader. ``kx=ky=5`` so mixed partial derivatives through total order
    3 are smooth for the offline generator.
    """

    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = repo_root() / "data" / "nfw_off_center"
        self.dir = Path(data_dir)
        fx, fxm, fw = (self.dir / n for n in GAMMA_TABLE_FILES)
        self.lnx = np.loadtxt(fx)
        self.lnxm = np.loadtxt(fxm)
        self.w = np.loadtxt(fw)
        if self.w.shape != (self.lnxm.size, self.lnx.size):
            raise ValueError(
                f"gamma table shape {self.w.shape} does not match axes "
                f"({self.lnxm.size}, {self.lnx.size})")
        self._spl = RectBivariateSpline(self.lnxm, self.lnx, self.w,
                                        kx=5, ky=5, s=0)

    def sha256(self):
        out = {}
        for name in GAMMA_TABLE_FILES:
            h = hashlib.sha256()
            h.update((self.dir / name).read_bytes())
            out[name] = h.hexdigest()
        return out

    def _clamp(self, lnx, lnxm):
        return (np.clip(lnx, self.lnx[0], self.lnx[-1]),
                np.clip(lnxm, self.lnxm[0], self.lnxm[-1]))

    def ln_u(self, lnx, lnxm):
        lnx, lnxm = self._clamp(np.asarray(lnx, dtype=float),
                                np.asarray(lnxm, dtype=float))
        return self._spl.ev(lnxm, lnx)

    def u(self, lnx, lnxm):
        return np.exp(self.ln_u(lnx, lnxm))

    def partial_w(self, i_lnxm, j_lnx):
        """d^(i+j) ln u / d lnxm^i d lnx^j on the full source grid."""
        if i_lnxm == 0 and j_lnx == 0:
            return self.w.copy()
        d = self._spl.partial_derivative(i_lnxm, j_lnx)
        return d(self.lnxm, self.lnx)


def dsigma_cen(R, lnM, rho_ref):
    """Centred DSigma_NFW(R, M), fixed conventions [Msun/(h pc^2)]."""
    y = y_of_lnM(lnM, rho_ref)
    return A0_of_y(y, rho_ref) * u_cen(np.asarray(R, dtype=float) * np.exp(-y))


def make_dsigma_mis(table=None):
    """Return DSigma_mis(R, r_mis, lnM, rho_ref) over the smooth table."""
    tab = table if table is not None else MisTable()

    def dsigma_mis(R, r_mis, lnM, rho_ref):
        y = y_of_lnM(lnM, rho_ref)
        lnx = np.log(np.asarray(R, dtype=float)) - y
        lnxm = np.log(np.asarray(r_mis, dtype=float)) - y
        return A0_of_y(y, rho_ref) * tab.u(lnx, lnxm)

    return dsigma_mis
