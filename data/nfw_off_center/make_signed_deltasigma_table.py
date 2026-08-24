#!/usr/bin/env python
"""Regenerate the miscentered-NFW DeltaSigma "single" table as a SIGNED,
linear table (replacing the broken log_deltasigma_single table).

WHY (the bug this fixes)
------------------------
The shipped `table_1000_1e-03_5e+03_log_deltasigma_single.txt` stores
log(DeltaSigma_mis), read back as `norm * exp(spline(...))` in both
`nfw_dsigma_mis.hh` (C++) and `nfw.py` (RichnessSelection).  Because it is a
LOG table it is structurally >= 0, but the true miscentered contrast

    DeltaSigma_mis(R|R_mis) = Sigmabar_mis(<R|R_mis) - Sigma_mis(R|R_mis)

goes NEGATIVE for R_mis > R (the halo center lies outside the aperture, so the
enclosed mean Sigmabar(<R) drops below the local Sigma(R)).  The log table
zeroes that whole negative lobe.  In the shear projection the mean-field
("rnd") term is the offset sweep  int dR_mis DeltaSigma_mis * (mean weight);
the negative lobe is exactly what makes it cancel (a uniformly-distributed
halo population is a uniform mass sheet -> zero contrast).  With the lobe
zeroed, rnd never cancels.  So we must store DeltaSigma_mis SIGNED.

THE MATH (all from the KNOWN closed-form centered Sigma and Sigmabar)
--------------------------------------------------------------------
Dimensionless NFW (x = R/r_s, Sigma0 = 2 rho_s r_s):
  Sigma_NFW(x)  = Sigma0 * f(x)        (Wright & Brainerd 2000, piecewise x<1 / x>1)
  Sigmabar(<x)  = Sigma0 * (2/x^2) g(x)                     "  "

Miscentered mean via the APERTURE-MASS identity (single integral of known f,
NO cumulative double integral and NO R'=R_mis cusp):
  Sigmabar_mis(<R|R_mis) = (1/pi R^2) int_0^{R+R_mis} Sigma(u) u Lambda(u) du
  Lambda(u) = 2*pi                        u <= R-R_mis          (ring fully inside; only if R>R_mis)
            = 2*arccos((u^2+R_mis^2-R^2)/(2 u R_mis))   |R-R_mis| <= u <= R+R_mis
            = 0                            otherwise
The inner full-circle part is ANALYTIC via the known Sigmabar:
  int_0^a Sigma(u) u du = (a^2/2) Sigmabar(<a)   =>   inner = pi (R-R_mis)^2 Sigmabar(<R-R_mis).
Only the smooth "band" |R-R_mis|<u<R+R_mis needs quadrature (fixed Gauss-Legendre,
integrand smooth off the diagonal; integrable log at the diagonal endpoint).

Sigma_mis is the single azimuthal average of the known Sigma:
  Sigma_mis(R|R_mis) = (1/pi) int_0^pi Sigma(s) dtheta,  s = sqrt(R^2+R_mis^2-2 R R_mis cos theta).

Identity (cross-check, not used for storage):
  DeltaSigma_mis = -(R/2) d/dR Sigmabar_mis.

STORAGE / UNITS
---------------
Table value = dimensionless DeltaSigma_mis = Sigmabar_mis(x,xmis) - Sigma_mis(x,xmis)
computed with r_s=1, Sigma0=1 (pure NFW; no rho_s, no rho_mult).  The readers
multiply by  norm = 2 r_s delta_c rho_crit rho_mult  and 1e-12, exactly as for
the sigma table -- so the axes and the norm are unchanged; ONLY the value array
and the exp() are dropped.  Output is a plain signed float grid.

Axes are taken verbatim from the existing single_logx / single_logxmis files so
the new table is a drop-in (same shape, same clamp ranges).
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LOGX = os.path.join(HERE, "table_1000_1e-03_5e+03_single_logx.txt")
LOGXMIS = os.path.join(HERE, "table_1000_1e-03_5e+03_single_logxmis.txt")
OUT = os.path.join(HERE, "table_1000_1e-03_5e+03_deltasigma_signed_single.txt")
NG = 512  # Gauss-Legendre nodes for the smooth band integral


def _f(x):
    """Dimensionless centered NFW Sigma f(x) (Sigma0=1), piecewise x<1 / x>1."""
    x = np.abs(np.asarray(x, float))
    x = np.maximum(x, 1e-12)
    o = np.empty_like(x)
    lo = x < 1 - 1e-8
    hi = x > 1 + 1e-8
    eq = ~(lo | hi)
    xl = x[lo]
    o[lo] = 1 / (xl * xl - 1) * (
        1 - 2 / np.sqrt(1 - xl * xl) * np.arctanh(np.sqrt((1 - xl) / (1 + xl))))
    xh = x[hi]
    o[hi] = 1 / (xh * xh - 1) * (
        1 - 2 / np.sqrt(xh * xh - 1) * np.arctan(np.sqrt((xh - 1) / (1 + xh))))
    o[eq] = 1 / 3.
    return o


def _sbar(x):
    """Dimensionless centered mean Sigmabar(<x) = (2/x^2) g(x) (Sigma0=1)."""
    x = np.abs(np.asarray(x, float))
    x = np.maximum(x, 1e-12)
    g = np.empty_like(x)
    lo = x < 1 - 1e-8
    hi = x > 1 + 1e-8
    eq = ~(lo | hi)
    xl = x[lo]
    g[lo] = np.log(xl / 2) + np.arccosh(1 / xl) / np.sqrt(1 - xl * xl)
    xh = x[hi]
    g[hi] = np.log(xh / 2) + np.arccos(1 / xh) / np.sqrt(xh * xh - 1)
    g[eq] = 1 + np.log(0.5)
    return 2 / x**2 * g


_GX, _GW = np.polynomial.legendre.leggauss(NG)
_GX = 0.5 * (_GX + 1)
_GW = 0.5 * _GW


def sbar_mis(x, xmis):
    """Sigmabar_mis(<x|xmis): analytic inner (2pi) part + GL band part."""
    x = np.asarray(x, float)
    xmis = np.asarray(xmis, float)
    inner = np.where(x > xmis, np.pi * (x - xmis)**2 * _sbar(np.abs(x - xmis)), 0.0)
    a = np.abs(x - xmis)
    b = x + xmis
    u = a[..., None] + (b - a)[..., None] * _GX
    A = np.clip((u * u + xmis[..., None]**2 - x[..., None]**2)
                / (2 * u * xmis[..., None]), -1, 1)
    band = np.sum(_f(u) * u * (2 * np.arccos(A)) * _GW, axis=-1) * (b - a)
    return (inner + band) / (np.pi * x * x)


def sigma_mis(x, xmis):
    """Sigma_mis(x|xmis): azimuthal average of the known Sigma over [0,pi]."""
    x = np.asarray(x, float)
    xmis = np.asarray(xmis, float)
    th = np.pi * _GX
    s = np.sqrt(x[..., None]**2 + xmis[..., None]**2
                - 2 * x[..., None] * xmis[..., None] * np.cos(th))
    return np.sum(_f(s) * _GW, axis=-1)   # (1/pi) int_0^pi f dtheta


def main():
    lnx = np.loadtxt(LOGX)
    lnxm = np.loadtxt(LOGXMIS)
    x = np.exp(lnx)
    xmis = np.exp(lnxm)
    print(f"grid: x[{x.size}] in [{x.min():.2e},{x.max():.2e}], "
          f"xmis[{xmis.size}] in [{xmis.min():.2e},{xmis.max():.2e}]")
    out = np.empty((xmis.size, x.size))   # row-major (xmis, x): matches log_sigma table layout
    for i, xm in enumerate(xmis):
        xm_row = np.full_like(x, xm)
        out[i] = sbar_mis(x, xm_row) - sigma_mis(x, xm_row)
        if i % 100 == 0:
            print(f"  row {i}/{xmis.size}  xmis={xm:.3e}  "
                  f"dS range [{out[i].min():+.3e},{out[i].max():+.3e}]")
    np.savetxt(OUT, out, fmt="%.10e")
    neg = (out < 0).mean() * 100
    print(f"wrote {OUT}\n  shape {out.shape}  signed  ({neg:.1f}% negative entries)")


if __name__ == "__main__":
    main()
