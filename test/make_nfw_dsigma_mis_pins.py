#!/usr/bin/env python
"""Exact reference values for test/nfw_dsigma_mis.test.cu (gamma-kernel miscentred NFW DeltaSigma).

DeltaSigma_mis(R) = < Sigma_bar(<R | offset) - Sigma(R | offset) > over the gamma offset
distribution P(R_s) = R_s / R_mis^2 exp(-R_s / R_mis), with
  * Sigma(R | offset): the NFW Sigma (Wright & Brainerd 2000) averaged over the ring of radius R
    centred at the offset, and
  * Sigma_bar(<R | offset): the mean Sigma inside that offset circle, from the exact
    ring-fraction formula.
All integrals are scipy quad at rel. accuracy ~1e-9.  This replaces the earlier cluster_toolkit
recipe, whose radial grid stopped at 15 Mpc/h and truncated the offset integral (+0.49% at
R = 10, R_mis = 2; +0.05% for R >= 6).  Checks: reduces to the analytic centred profile as
R_mis -> 0 (1e-11); averaging DeltaSigma over offsets agrees with building Sigma_gamma first
(5e-15 at R = 10, R_mis = 2); cluster_toolkit with its grid extended to 30-100 Mpc/h agrees to 7e-5.

Fixture: c = 4, rho = 0.3 rho_crit (h^2 Msun/Mpc^3), lnM = 32.2361913, comoving Mpc/h.

    python test/make_nfw_dsigma_mis_pins.py      # prints the `answers` initializer (~1 min)
"""
import numpy as np
from scipy.integrate import quad

RHOC = 2.77533742639e11
def nfw_params(lnM, c=4.0, rho=0.3*RHOC):
    dc = (200*c**3/3)/(np.log1p(c)-c/(1+c))
    r200 = np.cbrt(3*np.exp(lnM)/(800*np.pi*rho))
    rs = r200/c
    return rs, 2*rs*dc*rho*1e-12   # amplitude: Sigma = A * f(x), f = WB00 bracket/(x^2-1)

def f_sigma(x):
    """dimensionless Sigma/(2 rs dc rho): 1/(x^2-1) * [1 - arccosh(1/x)/sqrt(1-x^2)]"""
    x = float(x)
    if abs(x-1) < 1e-4:
        e = x-1
        return 1/3 - 0.4*e + 0.3714285714285714*e*e
    if x < 1:
        return (1 - np.arccosh(1/x)/np.sqrt(1-x*x))/(x*x-1)
    return (1 - np.arccos(1/x)/np.sqrt(x*x-1))/(x*x-1)

def g_mean(x):
    """dimensionless mean Sigma inside x: 2/x^2 * [ln(x/2) + h(x)]"""
    x = float(x)
    if abs(x-1) < 1e-4:
        return 2*(np.log(0.5)+1) + 0  # approx; avoid by not hitting
    if x < 1:
        h = np.arccosh(1/x)/np.sqrt(1-x*x)
    else:
        h = np.arccos(1/x)/np.sqrt(x*x-1)
    return 2/(x*x)*(np.log(x/2)+h)

EPS = 1e-11
def sigma_single(x, xo):
    """Azimuthal average of Sigma around ring radius x at offset xo."""
    if xo == 0: return f_sigma(x)
    g = lambda t: f_sigma(np.sqrt(max(x*x+xo*xo+2*x*xo*np.cos(t), 1e-300)))
    pts = []
    v = quad(g, 0, np.pi, epsabs=0, epsrel=EPS, limit=400, points=[np.pi*0.999] if abs(x-xo)<0.05*max(x,xo) else None)[0]
    return v/np.pi

def mbar_single(x, xo):
    """Mean Sigma inside the offset circle of radius x (offset xo):
    (1/(pi x^2)) int 2 pi s Sigma(s) frac(s) ds, frac = fraction of ring s inside."""
    a, b = abs(x-xo), x+xo
    inner = 0.0
    if x > xo:
        # all of disk s < x - xo inside: mean* area
        inner = g_mean(a)*a*a/2 if a > 0 else 0.0   # int_0^a s f ds = a^2/2 * gmean(a)
    def ring(s):
        cth = (s*s+xo*xo-x*x)/(2*s*xo)
        cth = min(1, max(-1, cth))
        return s*f_sigma(s)*np.arccos(cth)/np.pi
    if x < xo:
        mid = quad(ring, a, b, epsabs=0, epsrel=EPS, limit=400)[0]
    else:
        mid = quad(ring, a, b, epsabs=0, epsrel=EPS, limit=400)[0]
    return 2*(inner+mid)/(x*x)

def dsig_single(x, xo):
    return mbar_single(x, xo) - sigma_single(x, xo)

def dsig_gamma(x, xm, kernel="gamma", upper=np.inf):
    """<DeltaSigma>(x) averaged over P(xo) = xo/xm^2 exp(-xo/xm) (gamma, k=2)."""
    if kernel == "gamma":
        P = lambda u: u/xm**2*np.exp(-u/xm)
    else:  # rayleigh
        P = lambda u: u/xm**2*np.exp(-0.5*(u/xm)**2)
    f = lambda u: P(u)*dsig_single(x, u)
    hi = min(upper, 60*xm)
    pts = [p for p in (x, xm) if 0 < p < hi]
    edges = sorted(set([0.0] + pts + [hi]))
    tot = 0.0
    for lo_, hi_ in zip(edges[:-1], edges[1:]):
        tot += quad(f, lo_, hi_, epsabs=0, epsrel=1e-9, limit=300)[0]
    return tot

def sigma_gamma(x, xm):
    P = lambda u: u/xm**2*np.exp(-u/xm)
    f = lambda u: P(u)*sigma_single(x, u)
    edges = sorted(set([0.0, x, xm, 60*xm]))
    return sum(quad(f, a, b, epsabs=0, epsrel=1e-9, limit=300)[0] for a, b in zip(edges[:-1], edges[1:]))


def main():
    lnM, conc = 32.2361913, 4.0
    rs, amp = nfw_params(lnM, c=conc)
    radii = np.logspace(np.log10(0.1), np.log10(10.0), 10)
    rmis = (0.1, 0.5, 2.0)
    print("  std::vector<double> answers{", end="")
    rows = []
    for r in radii:
        rows.append(", ".join(f"{amp * dsig_gamma(r / rs, rm / rs):.7f}" for rm in rmis))
    print((",\n" + " " * 30).join(rows) + "};")


if __name__ == "__main__":
    main()
