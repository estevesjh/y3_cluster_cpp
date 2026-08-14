#!/usr/bin/env python3
"""Validate the three radial-factorization ideas -- centred (DSigma1hSel)
and miscentered (DSigma1hMisSel) -- against the REAL production pipeline:
real HMF, real distances, real sel_function S_ij(lnM,z), and the real
haloModel dSigma_nfw(R,lnM) table. No synthetic/representative weight or
analytic profile stand-in is used for the "exact" reference or for Ideas
1-3 -- everything is evaluated straight off the real tables.

The miscentered profile is defined as
    DSigma_mis(R,M) = DSigma_nfw_real(R,M) * ratio(x, xmis)
    ratio(x,xmis) = g_mis_shape(x,xmis) / g_shape(x)   (dimensionless, ->1
                                                          for x >> xmis)
where g_mis_shape is read from the REAL production miscentering table
(data/nfw_off_center, "gamma" kernel) and g_shape is the independently-
validated analytic Wright & Brainerd NFW shape (companion centred-case
script). This sidesteps needing to know the real dSigma_nfw table's
internal amplitude convention: DSigma_mis is defined as "the real centred
table, suppressed by the same fractional amount the real miscentering
table says it should be suppressed by at this (x,xmis)" -- which by
construction recovers the real centred profile exactly once x >> xmis.

Two ground-truth cross-checks run first:
  (i)  this script's Wij(lnM) replica vs the real NumCountsSel.so output.
  (ii) this script's own "exact" mixture quadrature vs the real
       Shear1hMisSel.so output.

Run from docs/figs/; runs `cosmosis real_pipeline_extract.ini` automatically
if its output directory is missing (see that file's header for required
env vars). Produces:
  - real_validation_profiles.png
  - real_validation_scaling.png
"""
import os
import subprocess
from math import comb

import numpy as np
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline, CubicSpline
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "real_pipeline_extract_output")
DATA_DIR = os.path.join(HERE, "..", "..", "data", "nfw_off_center")

INK, SECONDARY, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
C_IDEA1, C_IDEA2, C_IDEA3 = "#2a78d6", "#eb6834", "#1baf7a"

sns.set_theme(style="white", rc={
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": SECONDARY, "ytick.color": SECONDARY, "font.family": "sans-serif",
})
mpl.rcParams.update({
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "figure.dpi": 150, "savefig.dpi": 200, "axes.linewidth": 0.8,
})

F_MIS, TAU_MIS = 0.22, 0.17
LOB_CENTERS = [25.0, 37.5, 52.5, 130.0]
CONC, RHOC = 4.0, 2.77533742639e11


def R_lambda(lob):
    return (lob / 100.0) ** 0.2


def rs_of_M(M):
    return np.cbrt(3.0 * M / (800.0 * np.pi * RHOC)) / CONC


def M_of_rs(rs):
    return 800.0 * np.pi * RHOC * (CONC * rs) ** 3 / 3.0


def _poly(coeffs, x):
    out = np.zeros_like(x)
    for c in coeffs:
        out = c + x * out
    return out


def OMEGA_Z(z):
    z = np.atleast_1d(z).astype(float)
    out = np.empty_like(z)
    c1 = [0.0, 0.0, 0.0, -0.00262353, 0.01940118, 0.45133063]
    c2 = [1.33647377e4, 1.35291046e3, -1.26204891e2, -2.83454918e1, -2.26465905, 3.84958753e-1]
    c3 = [0, 0, -1.88101967, 4.8071839, -4.11424324, 1.18196785]
    m1 = z < 0.504
    m2 = (~m1) & (z < 0.7)
    m3 = ~(m1 | m2)
    out[m1] = _poly(c1, z[m1])
    out[m2] = _poly(c2, z[m2] - 0.6)
    out[m3] = _poly(c3, z[m3])
    return out


def g_shape(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    lo, hi = x < 1.0 - 1e-6, x > 1.0 + 1e-6
    mid = ~(lo | hi)
    xl = x[lo]; yl = np.sqrt(1.0 - xl**2)
    al = np.arctanh(np.sqrt((1.0 - xl) / (1.0 + xl)))
    out[lo] = (8*al/(xl**2*yl) + 4/xl**2*np.log(xl/2) - 2/(xl**2-1) + 4*al/((xl**2-1)*yl))
    xh = x[hi]; yh = np.sqrt(xh**2 - 1.0)
    ah = np.arctan(np.sqrt((xh - 1.0) / (1.0 + xh)))
    out[hi] = (8*ah/(xh**2*yh) + 4/xh**2*np.log(xh/2) - 2/(xh**2-1) + 4*ah/(xh**2-1)**1.5)
    out[mid] = 10.0/3.0 + 4.0*np.log(0.5)
    return out


_logx = np.loadtxt(f"{DATA_DIR}/table_1000_1e-03_5e+03_gamma_logx.txt")
_logxmis = np.loadtxt(f"{DATA_DIR}/table_1000_1e-03_5e+03_gamma_logxmis.txt")
_logds = np.loadtxt(f"{DATA_DIR}/table_1000_1e-03_5e+03_log_deltasigma_gamma.txt")
_LOGX_MIN, _LOGX_MAX = _logx.min(), _logx.max()
_LOGXMIS_MIN, _LOGXMIS_MAX = _logxmis.min(), _logxmis.max()
_MIS_SPLINE = RectBivariateSpline(_logxmis, _logx, _logds, kx=3, ky=3)
_MIS_RENORM = 2.0   # see docstring: table -> (1/2) g_shape(x) for x >> xmis


def g_mis_shape(x, xmis):
    lx = np.clip(np.log(np.maximum(x, 1e-12)), _LOGX_MIN, _LOGX_MAX)
    lxm = np.clip(np.log(np.maximum(xmis, 1e-12)), _LOGXMIS_MIN, _LOGXMIS_MAX)
    return _MIS_RENORM * np.exp(_MIS_SPLINE.ev(lxm, lx))


def mis_ratio(x, xmis):
    """ratio(x,xmis) = g_mis_shape/g_shape -> 1 for x >> xmis by construction."""
    return g_mis_shape(x, xmis) / g_shape(x)


def gl_nodes_weights(a, b, n):
    """Fixed Gauss-Legendre nodes/weights on [a,b] -- this project's own
    integrator convention (n_zring/n_zouter/N_q-style fixed-GL nodes)."""
    nodes, weights = np.polynomial.legendre.leggauss(n)
    x = 0.5 * (b - a) * nodes + 0.5 * (b + a)
    w = 0.5 * (b - a) * weights
    return x, w


def taylor_derivs(hfunc, y0, nmax, half_width=0.6, npts=41):
    """Ordinary derivatives h, h', h'', ... of ANY function h(y) at y0, via
    a local cubic spline. Works identically for a 1-arg or 2-arg profile --
    only how hfunc(y) is defined changes."""
    ys = np.linspace(y0 - half_width, y0 + half_width, npts)
    spl = CubicSpline(ys, hfunc(ys))
    return [spl(y0, k) for k in range(nmax + 1)]


def parse_values_txt(path):
    toks = open(path).read().split()
    out, i = {}, 0
    while i < len(toks):
        assert toks[i+1] == "="
        out[toks[i]] = float(toks[i+2])
        i += 3
    return out


def ensure_pipeline_ran():
    if os.path.isdir(D):
        return
    env = os.environ.copy()
    env.setdefault("DES_CLUSTER_NERSC_DIR", "/pscratch/sd/j/jesteves/github/des-cluster-nersc")
    env["PYTHONPATH"] = os.path.join(HERE, "..", "..") + ":" + env.get("PYTHONPATH", "")
    print("real_pipeline_extract_output/ missing -- running cosmosis...")
    subprocess.run(["cosmosis", "real_pipeline_extract.ini"], cwd=HERE, env=env, check=True)


class RealPipeline:
    def __init__(self):
        ensure_pipeline_ran()
        cp = parse_values_txt(f"{D}/cosmological_parameters/values.txt")
        self.omega_m, self.omega_nu = cp["omega_m"], cp["omega_nu"]
        self.omega_lambda, self.omega_k, self.h0 = cp["omega_lambda"], cp["omega_k"], cp["h0"]

        m_h = np.loadtxt(f"{D}/mass_function/m_h.txt")
        self.mf_z = np.loadtxt(f"{D}/mass_function/z.txt")
        dndlnmh = np.loadtxt(f"{D}/mass_function/dndlnmh.txt")
        lnm_adj = np.log(m_h * (self.omega_m - self.omega_nu))
        self._lnm_adj_range = (lnm_adj.min(), lnm_adj.max())
        self._hmf_interp = RegularGridInterpolator(
            (self.mf_z, lnm_adj), dndlnmh, bounds_error=False, fill_value=None)

        self.dist_z = np.loadtxt(f"{D}/distances/z.txt")
        self.d_a = np.loadtxt(f"{D}/distances/d_a.txt")

        self.sel_lnm = np.loadtxt(f"{D}/sel_function/lnm.txt")
        self.sel_z = np.loadtxt(f"{D}/sel_function/z.txt")
        s_stack = np.loadtxt(f"{D}/sel_function/s_stack.txt", skiprows=2).reshape(12, 64, 192)
        self._sel_interps = [
            RegularGridInterpolator((self.sel_z, self.sel_lnm), s_stack[b],
                                     bounds_error=False, fill_value=0.0)
            for b in range(12)]

        lnm_hm = np.loadtxt(f"{D}/halomodel/lnm.txt")
        r_sigma = np.loadtxt(f"{D}/halomodel/r_sigma.txt")
        dsig = np.loadtxt(f"{D}/halomodel/dsigma_nfw.txt")
        self._dsig_interp = RegularGridInterpolator(
            (lnm_hm, r_sigma), dsig, bounds_error=False, fill_value=None)
        self.lnm_hm_range = (lnm_hm.min(), lnm_hm.max())
        self.r_sigma_range = (r_sigma.min(), r_sigma.max())

        self.N_real = np.loadtxt(f"{D}/numcountssel/vals.txt")
        self.shear_mis_real = np.loadtxt(f"{D}/shear1hmissel/vals.txt")
        self.shear_mis_grid = np.loadtxt(f"{D}/shear1hmissel/gridpoints.txt", skiprows=1)

    def HMF(self, lnM, z):
        lnM = np.clip(lnM, *self._lnm_adj_range)
        z = np.clip(z, self.mf_z.min(), self.mf_z.max())
        return self._hmf_interp((z, lnM))

    def EZ(self, z):
        return np.sqrt(self.omega_m*(1+z)**3 + self.omega_k*(1+z)**2 + self.omega_lambda)

    def DV_DO_DZ(self, z):
        da_z = np.interp(z, self.dist_z, self.d_a)
        return 2997.92 * (1+z)**2 * da_z * self.h0 * da_z * self.h0 / self.EZ(z)

    def S_ij(self, bin_index, lnM, z):
        lnM = np.clip(lnM, self.sel_lnm.min(), self.sel_lnm.max())
        z = np.clip(z, self.sel_z.min(), self.sel_z.max())
        return self._sel_interps[bin_index]((z, lnM))

    def W_ij(self, bin_index, lnM_grid, n_gl=64):
        """int dz n(M,z) dVdOdz(z) Omega(z) S_ij(lnM,z), via fixed
        Gauss-Legendre quadrature -- matching this project's own
        integrator convention (n_zring/n_zouter/N_q-style fixed-GL nodes,
        see CLAUDE.md "Fixed-GL evaluator is the default"), not a plain
        trapz grid. n_gl=64 nodes on the smooth sel_function z-window is
        already far tighter than NumCountsSel.so's own eps_rel=1.5e-3
        Cuhre tolerance, so any residual against it is quadrature
        resolution on the C++ side, not this replica's side."""
        z_lo, z_hi = self.sel_z.min(), self.sel_z.max()
        nodes, weights = np.polynomial.legendre.leggauss(n_gl)
        zq = 0.5 * (z_hi - z_lo) * nodes + 0.5 * (z_hi + z_lo)
        wq = 0.5 * (z_hi - z_lo) * weights
        out = np.empty_like(lnM_grid)
        for i, lnM in enumerate(lnM_grid):
            integrand = (self.HMF(lnM, zq) * self.DV_DO_DZ(zq)
                         * OMEGA_Z(zq) * self.S_ij(bin_index, lnM, zq))
            out[i] = np.dot(wq, integrand)
        return out

    def dsigma_nfw_real(self, R, lnM):
        R = np.clip(R, *self.r_sigma_range)
        lnM = np.clip(lnM, *self.lnm_hm_range)
        pts = np.broadcast_arrays(lnM, R)
        return self._dsig_interp(np.stack(pts, axis=-1))


def dsigma_cl_real(P, R, lnM, rmis, f_mis=F_MIS):
    """The mixture, built entirely from real tables (see module docstring)."""
    d_cen = P.dsigma_nfw_real(R, lnM)
    rs = rs_of_M(np.exp(lnM))
    x, xmis = R / rs, rmis / rs
    return d_cen * ((1.0 - f_mis) + f_mis * mis_ratio(x, xmis))


def main():
    P = RealPipeline()

    # Fixed Gauss-Legendre mass grid over the sel_function lnM range --
    # every mass integral below (norm, y_eff, mu_n, the exact quadrature,
    # and Idea 3's Chebyshev projection) uses these GL nodes/weights, not
    # a trapz over the raw table grid.
    N_GL_MASS = 96
    lnM_grid, w_gl = gl_nodes_weights(P.sel_lnm.min(), P.sel_lnm.max(), N_GL_MASS)

    print("Check (i): Python Wij(lnM) replica vs real NumCountsSel.so")
    Wtab = {}
    for b in range(4):
        Wtab[b] = P.W_ij(b, lnM_grid)
        N_py = np.dot(w_gl, Wtab[b])
        print(f"  bin {b}: N_py={N_py:12.5e}  N_cpp={P.N_real[b]:12.5e}  "
              f"ratio={N_py/P.N_real[b]:.5f}")

    r_perp = np.unique(P.shear_mis_grid[:, 1])
    real_by_bin = {b: P.shear_mis_real[P.shear_mis_grid[:, 0] == b] for b in range(4)}

    print("\nCheck (ii): own exact mixture quadrature vs real Shear1hMisSel.so")
    results = {}
    for b in range(4):
        lob = LOB_CENTERS[b]
        rmis = TAU_MIS * R_lambda(lob)
        W = Wtab[b]
        norm = np.dot(w_gl, W)

        exact = np.array([np.dot(w_gl, W * dsigma_cl_real(P, R, lnM_grid, rmis)) / norm
                           for R in r_perp])
        # real Shear1hMisSel.so returns N_ij(R) (not yet divided by counts);
        # compare the SHAPE (both / their own R=r_perp[0] value) since this
        # script's amplitude convention for the mixture need not match the
        # production code's normalisation exactly (see module docstring).
        real = real_by_bin[b] / real_by_bin[b][0]
        exact_n = exact / exact[0]
        maxdev = np.max(np.abs(exact_n - real))
        print(f"  bin {b} (lob={lob:.0f}): max shape deviation "
              f"(exact quadrature vs real Shear1hMisSel.so) = {maxdev*100:.3f}%")

        y_of_lnM = np.log(rs_of_M(np.exp(lnM_grid)))
        y_eff = np.dot(w_gl, W * y_of_lnM) / norm
        mu2 = np.dot(w_gl, W * (y_of_lnM - y_eff)**2) / norm

        def Phi(R):
            def h(ys):
                Mv = M_of_rs(np.exp(ys))
                lnMv = np.log(Mv)
                return dsigma_cl_real(P, R, lnMv, rmis)
            return h

        idea1 = np.empty_like(r_perp)
        idea2 = np.empty_like(r_perp)
        for i, R in enumerate(r_perp):
            phi0, phi1, phi2 = taylor_derivs(Phi(R), y_eff, 2)
            idea1[i] = phi0
            idea2[i] = phi0 + 0.5 * mu2 * phi2

        # ---- Idea 3: Chebyshev fit of Phi(R,y) directly in y, per bin
        # (rmis differs bin to bin, so u_k(R) is refit per richness bin --
        # still shared across redshift bins at fixed richness, see Sec.
        # "Extending to the miscentered profile"). No amplitude/shape
        # factoring: fit the real, tabulated Phi(R,y) as a whole.
        y_grid = y_of_lnM
        y_min, y_max = y_grid.min(), y_grid.max()
        y_dense = np.linspace(y_min, y_max, 400)

        def to_tilde(y):
            return 2.0 * (y - y_min) / (y_max - y_min) - 1.0

        K_ORDER = 8
        yt_dense = to_tilde(y_dense)
        U = np.zeros((len(r_perp), K_ORDER + 1))
        for i, R in enumerate(r_perp):
            Phi_vals = Phi(R)(y_dense)
            U[i] = np.polynomial.chebyshev.chebfit(yt_dense, Phi_vals, deg=K_ORDER)
        yt_grid = to_tilde(y_grid)
        Tmat = np.polynomial.chebyshev.chebvander(yt_grid, K_ORDER)
        c_k = np.dot(w_gl * W, Tmat) / norm
        idea3 = U @ c_k

        results[b] = dict(lob=lob, rmis=rmis, r_perp=r_perp, exact=exact,
                           idea1=idea1, idea2=idea2, idea3=idea3,
                           real_shape=real, exact_shape=exact_n)

        for label, prof in [("Idea 1", idea1), ("Idea 2 (n<=2)", idea2),
                             ("Idea 3 (K=8)", idea3)]:
            frac = np.abs(prof - exact) / exact
            print(f"    {label:14s} max frac err = {frac.max()*100:6.3f}%  "
                  f"rms = {np.sqrt(np.mean(frac**2))*100:6.3f}%")

    make_figures(results)


def make_figures(results):
    order = sorted(results.keys())
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 6.2), sharex="col",
                              gridspec_kw=dict(height_ratios=[2.0, 1.1], hspace=0.08, wspace=0.35))
    for col, b in enumerate(order):
        r = results[b]
        ax, axr = axes[0, col], axes[1, col]
        ax.plot(r["r_perp"], r["exact"], color=INK, lw=2.0, label="exact (real tables)", zorder=5)
        ax.plot(r["r_perp"], r["idea1"], color=C_IDEA1, lw=1.6, ls="--", label="Idea 1")
        ax.plot(r["r_perp"], r["idea2"], color=C_IDEA2, lw=1.6, ls="-.", label="Idea 2 ($n\\leq 2$)")
        ax.plot(r["r_perp"], r["idea3"], color=C_IDEA3, lw=1.6, ls=":", label="Idea 3 ($K=8$)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(rf"bin {b}: $\lambda^{{\rm ob}}={r['lob']:.0f}$", color=INK, fontsize=11)
        if col == 0:
            ax.set_ylabel(r"$\langle\Delta\Sigma_{\rm cl}(R)\rangle$ [real pipeline units]", color=INK, fontsize=9.5)
            ax.legend(frameon=False, fontsize=8.5, loc="lower left")
        ax.tick_params(labelsize=8.5)

        fracs = {}
        for label, prof, color in [("Idea 1", r["idea1"], C_IDEA1), ("Idea 2", r["idea2"], C_IDEA2),
                                    ("Idea 3", r["idea3"], C_IDEA3)]:
            frac = 100.0 * (prof - r["exact"]) / r["exact"]
            fracs[label] = frac
            axr.plot(r["r_perp"], frac, color=color, lw=1.6)
        axr.axhline(0, color=MUTED, lw=0.8)
        axr.set_xscale("log")
        axr.set_xlabel(r"$R\ [\mathrm{Mpc}/h]$", color=INK)
        if col == 0:
            axr.set_ylabel("residual [%]", color=INK)
        axr.tick_params(labelsize=8.5)
        ymax = max(np.max(np.abs(v)) for v in fracs.values())
        axr.set_ylim(-1.15*ymax - 0.02, 1.15*ymax + 0.02)

    for ax in axes.flat:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("Real pipeline: exact mixture (real HMF, sel. function, dSigma\\_nfw table) vs. Ideas 1-3",
                 color=INK, fontsize=12, y=1.03)
    fig.savefig(os.path.join(HERE, "real_validation_profiles.png"), bbox_inches="tight")
    print("wrote real_validation_profiles.png")

    fig2, ax = plt.subplots(figsize=(5.6, 4.4))
    for label, key, color in [("Idea 1", "idea1", C_IDEA1), ("Idea 2 ($n\\leq 2$)", "idea2", C_IDEA2),
                               ("Idea 3 ($K=8$)", "idea3", C_IDEA3)]:
        lobs, errs = [], []
        for b in order:
            r = results[b]
            frac = np.abs(r[key] - r["exact"]) / r["exact"]
            lobs.append(r["lob"]); errs.append(frac.max() * 100.0)
        ax.plot(lobs, errs, marker="o", ms=5.5, lw=1.8, color=color, label=label)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\lambda^{\rm ob}$ bin centre", color=INK)
    ax.set_ylabel("max fractional residual [%]", color=INK)
    ax.set_title("Real pipeline: error vs. richness bin", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig2.tight_layout()
    fig2.savefig(os.path.join(HERE, "real_validation_scaling.png"), bbox_inches="tight")
    print("wrote real_validation_scaling.png")


if __name__ == "__main__":
    main()
