"""Closed-form density profiles for the transform-chain test bench.

Every profile provides the full analytic chain

    rho(r)  --3D FT-->  rho_tilde(k)  --inverse-->  rho(r)
    rho(r)  --Abel-->   Sigma(R)      --interior--> Sigma_bar(<R)
    DeltaSigma(R) = Sigma_bar(<R) - Sigma(R)

so each numerical transform (P(k)->xi, xi->Sigma, Sigma->DeltaSigma) can be
validated against an exact answer, and a failure localizes to one stage.

Conventions
-----------
- Fourier transform: rho_tilde(k) = int d^3r rho(r) e^{-i k.r}
                                  = 4 pi int r^2 rho(r) j0(kr) dr.
  This matches the mcfit/cluster_toolkit xi convention
  xi(r) = int dk k^2/(2 pi^2) P(k) j0(kr): feeding rho_tilde(k) as "P(k)"
  returns rho(r) as "xi(r)" exactly.
- Unit-agnostic: lengths in any unit L, rho0 in any unit U/L^3.
  Then [rho_tilde] = U, [Sigma] = U/L^2... times L (i.e. rho0*L).
  Callers own the unit bookkeeping (see the harness unit ledger).

Closed forms sourced from CLensPy docs (einasto_power_spectrum.tex,
einasto_proj_density.tex) and Wright & Brainerd (2000); all verified
numerically by the self-check in __main__.
"""

import numpy as np
from scipy.special import kv, sici


# --------------------------------------------------------------------------
# Gaussian = Einasto n=1/2 (alpha=2). Flat core.
# --------------------------------------------------------------------------
class GaussianProfile:
    """rho(r) = rho0 * exp(-(r/s)^2)."""

    name = "gaussian"

    def __init__(self, rho0=1.0, s=1.0):
        self.rho0, self.s = float(rho0), float(s)

    def rho(self, r):
        return self.rho0 * np.exp(-(np.asarray(r) / self.s) ** 2)

    def rho_tilde(self, k):
        k = np.asarray(k)
        return self.rho0 * np.pi ** 1.5 * self.s ** 3 * np.exp(-(k * self.s) ** 2 / 4.0)

    def sigma(self, R):
        x = np.asarray(R) / self.s
        return self.rho0 * np.sqrt(np.pi) * self.s * np.exp(-(x ** 2))

    def sigma_bar(self, R):
        x = np.asarray(R) / self.s
        return self.rho0 * np.sqrt(np.pi) * self.s * (1.0 - np.exp(-(x ** 2))) / x ** 2

    def delta_sigma(self, R):
        return self.sigma_bar(R) - self.sigma(R)

    def total_mass(self):
        # int d^3r rho = rho0 pi^{3/2} s^3
        return self.rho0 * np.pi ** 1.5 * self.s ** 3


# --------------------------------------------------------------------------
# Exponential = Einasto n=1 (alpha=1).
# --------------------------------------------------------------------------
class ExponentialProfile:
    """rho(r) = rho0 * exp(-r/h)."""

    name = "exponential"

    def __init__(self, rho0=1.0, h=1.0):
        self.rho0, self.h = float(rho0), float(h)

    def rho(self, r):
        return self.rho0 * np.exp(-np.asarray(r) / self.h)

    def rho_tilde(self, k):
        k = np.asarray(k)
        return 8.0 * np.pi * self.rho0 * self.h ** 3 / (1.0 + (k * self.h) ** 2) ** 2

    def sigma(self, R):
        x = np.asarray(R) / self.h
        return 2.0 * self.rho0 * self.h * x * kv(1, x)

    def sigma_bar(self, R):
        # M2D(R) = 4 pi rho0 h^3 [2 - x^2 K2(x)]  (from d/dx[x^2 K2] = -x^2 K1)
        x = np.asarray(R) / self.h
        return 4.0 * self.rho0 * self.h * (2.0 - x ** 2 * kv(2, x)) / x ** 2

    def delta_sigma(self, R):
        return self.sigma_bar(R) - self.sigma(R)

    def total_mass(self):
        return 8.0 * np.pi * self.rho0 * self.h ** 3


# --------------------------------------------------------------------------
# NFW. Cuspy core; the production-relevant case.
# --------------------------------------------------------------------------
class NFWProfile:
    """rho(r) = rho_s / [x (1+x)^2], x = r/r_s.

    rho_tilde: untruncated (log-divergent total mass, finite for k > 0)
    or truncated at r = c*r_s (finite mass; use this for FFTLog stages,
    the untruncated form diverges logarithmically as k -> 0).
    Sigma/DeltaSigma: Wright & Brainerd (2000), untruncated.
    """

    name = "nfw"

    def __init__(self, rho_s=1.0, r_s=1.0, c=5.0):
        self.rho_s, self.r_s, self.c = float(rho_s), float(r_s), float(c)

    def rho(self, r, truncated=False):
        x = np.asarray(r) / self.r_s
        out = self.rho_s / (x * (1.0 + x) ** 2)
        if truncated:
            out = np.where(x <= self.c, out, 0.0)
        return out

    def rho_tilde(self, k, truncated=False):
        kappa = np.asarray(k, dtype=float) * self.r_s
        P1 = 4.0 * np.pi * self.rho_s * self.r_s ** 3
        si_k, ci_k = sici(kappa)
        if truncated:
            c = self.c
            si_ck, ci_ck = sici((1.0 + c) * kappa)
            out = (np.sin(kappa) * (si_ck - si_k)
                   + np.cos(kappa) * (ci_ck - ci_k)
                   - np.sin(c * kappa) / ((1.0 + c) * kappa))
        else:
            out = np.sin(kappa) * (np.pi / 2.0 - si_k) - np.cos(kappa) * ci_k
        return P1 * out

    @staticmethod
    def _f_wb(x):
        """Sigma shape: Sigma = 2 r_s rho_s f(x)."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        f = np.full_like(x, 1.0 / 3.0)
        lo, hi = x < 1.0, x > 1.0
        xl, xh = x[lo], x[hi]
        f[lo] = (1.0 - 2.0 / np.sqrt(1.0 - xl ** 2)
                 * np.arctanh(np.sqrt((1.0 - xl) / (1.0 + xl)))) / (xl ** 2 - 1.0)
        f[hi] = (1.0 - 2.0 / np.sqrt(xh ** 2 - 1.0)
                 * np.arctan(np.sqrt((xh - 1.0) / (xh + 1.0)))) / (xh ** 2 - 1.0)
        return f

    @staticmethod
    def _g_wb(x):
        """DeltaSigma shape: DeltaSigma = r_s rho_s g(x)."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        g = np.full_like(x, 10.0 / 3.0 + 4.0 * np.log(0.5))
        lo, hi = x < 1.0, x > 1.0
        xl, xh = x[lo], x[hi]
        al = np.arctanh(np.sqrt((1.0 - xl) / (1.0 + xl)))
        g[lo] = (8.0 * al / (xl ** 2 * np.sqrt(1.0 - xl ** 2))
                 + 4.0 / xl ** 2 * np.log(xl / 2.0)
                 - 2.0 / (xl ** 2 - 1.0)
                 + 4.0 * al / ((xl ** 2 - 1.0) * np.sqrt(1.0 - xl ** 2)))
        ah = np.arctan(np.sqrt((xh - 1.0) / (xh + 1.0)))
        g[hi] = (8.0 * ah / (xh ** 2 * np.sqrt(xh ** 2 - 1.0))
                 + 4.0 / xh ** 2 * np.log(xh / 2.0)
                 - 2.0 / (xh ** 2 - 1.0)
                 + 4.0 * ah / ((xh ** 2 - 1.0) ** 1.5))
        return g

    def sigma(self, R):
        return 2.0 * self.r_s * self.rho_s * self._f_wb(np.asarray(R) / self.r_s)

    def delta_sigma(self, R):
        return self.r_s * self.rho_s * self._g_wb(np.asarray(R) / self.r_s)

    def sigma_bar(self, R):
        return self.delta_sigma(R) + self.sigma(R)

    def mass_3d(self, r):
        x = np.asarray(r) / self.r_s
        return 4.0 * np.pi * self.rho_s * self.r_s ** 3 * (np.log(1.0 + x) - x / (1.0 + x))


# --------------------------------------------------------------------------
# Numeric self-check: every closed form vs direct quadrature.
# Run:  python analytic_profiles.py
# --------------------------------------------------------------------------
def _scalar(v):
    """float() that also accepts size-1 arrays under numpy >= 2."""
    return float(np.asarray(v).reshape(-1)[0])


def _selfcheck(tol=5e-6):
    from scipy.integrate import quad

    failures = []

    def check(label, got, want, rtol):
        rel = abs(got / want - 1.0)
        status = "ok " if rel < rtol else "FAIL"
        if rel >= rtol:
            failures.append(label)
        print(f"  [{status}] {label}: closed={want:.6e} quad={got:.6e} rel={rel:.2e}")

    profiles = [
        ("gaussian s=1", GaussianProfile(rho0=2.0, s=1.0), {}),
        ("exponential h=1", ExponentialProfile(rho0=2.0, h=1.0), {}),
        ("nfw rs=0.5 c=5", NFWProfile(rho_s=2.0, r_s=0.5, c=5.0), {}),
    ]
    for name, p, _ in profiles:
        print(f"profile: {name}")
        # 1) rho_tilde vs 4 pi int r^2 rho j0(kr) dr.
        #    NFW: only the TRUNCATED form is checked (and only the truncated
        #    form is fed to any numerical transform in the bench -- the
        #    untruncated FT is k->0 log-divergent and its slowly decaying
        #    oscillatory tail defeats generic quadrature; it stays as a
        #    reference formula, unused by the bench).
        if isinstance(p, NFWProfile):
            for k in (0.3, 2.0, 8.0):
                num, _ = quad(lambda r: 4 * np.pi * r ** 2 * _scalar(p.rho(r)) * np.sinc(k * r / np.pi),
                              0, p.c * p.r_s, limit=400)
                check(f"rho_tilde_trunc(k={k})", num, _scalar(p.rho_tilde(k, truncated=True)), 1e-5)
            # k->0 anchor: truncated FT -> M3d(c r_s)
            check("rho_tilde_trunc(k->0)=M3d", _scalar(p.mass_3d(p.c * p.r_s)),
                  _scalar(p.rho_tilde(1e-6, truncated=True)), 1e-4)
        else:
            for k in (0.3, 2.0, 8.0):
                num, _ = quad(lambda r: 4 * np.pi * r ** 2 * _scalar(p.rho(r)) * np.sinc(k * r / np.pi),
                              0, np.inf, limit=400)
                check(f"rho_tilde(k={k})", num, _scalar(p.rho_tilde(k)), 1e-5)
            check("rho_tilde(k->0)=Mtot", p.total_mass(), _scalar(p.rho_tilde(1e-8)), 1e-6)
        # 2) Sigma vs Abel 2 int_R^inf rho r dr / sqrt(r^2 - R^2)
        for R in (0.2, 1.0, 3.0):
            num, _ = quad(lambda u: 2 * _scalar(p.rho(np.hypot(R, u))), 0, np.inf, limit=400)
            check(f"sigma(R={R})", num, _scalar(p.sigma(R)), 1e-5)
        # 3) sigma_bar vs (2/R^2) int_0^R Sigma R' dR'
        for R in (0.5, 2.0):
            num, _ = quad(lambda t: _scalar(p.sigma(t)) * t, 0, R, limit=400)
            check(f"sigma_bar(R={R})", 2 * num / R ** 2, _scalar(p.sigma_bar(R)), 1e-5)
        # 4) delta_sigma consistency
        for R in (0.5, 2.0):
            ds = _scalar(p.sigma_bar(R)) - _scalar(p.sigma(R))
            check(f"delta_sigma(R={R})", ds, _scalar(p.delta_sigma(R)), 1e-10)

    if failures:
        raise SystemExit(f"SELF-CHECK FAILED: {failures}")
    print("all self-checks passed")


if __name__ == "__main__":
    _selfcheck()
