#!/usr/bin/env python3
"""Validate the three Shear1h/DSigma1h radial-factorization approaches
against a direct (exact) mass integral, and against an independent
numerical projection of the NFW profile.

Companion to ../shear1h_radial_factorization.tex (Sec. "Numerical
validation"). Produces:
  - shear1h_validation_profiles.png
  - shear1h_validation_scaling.png

The mass--redshift weight Wij(lnM) used here is a *representative*
construction (Tinker-like HMF power-law/exponential shape x lognormal
mass--richness kernel of specified sigma_lnM), NOT a literal read-out
of the production sel_function.py tables -- reproducing that requires
running the full CosmoSIS pipeline, which is out of scope for this
math check. What is tested is exactly the thing the three approaches
depend on: how the approximation error scales with the *width* and
*shape* of the mass weight, at the sigma_lnM values quoted in the note.
"""
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# ---------------------------------------------------------------------------
# Palette (validated categorical order; see dataviz skill references/palette.md)
# ---------------------------------------------------------------------------
INK        = "#0b0b0b"
SECONDARY  = "#52514e"
MUTED      = "#898781"
GRID       = "#e1e0d9"
C_IDEA1    = "#2a78d6"   # slot 1 - blue
C_IDEA2    = "#eb6834"   # slot 2 - orange
C_IDEA3    = "#1baf7a"   # slot 3 - aqua

sns.set_theme(style="white", rc={
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": SECONDARY,
    "ytick.color": SECONDARY,
    "font.family": "sans-serif",
})
mpl.rcParams["axes.grid"] = True
mpl.rcParams["grid.color"] = GRID
mpl.rcParams["grid.linewidth"] = 0.7
mpl.rcParams["figure.dpi"] = 150
mpl.rcParams["savefig.dpi"] = 200
mpl.rcParams["axes.linewidth"] = 0.8

# ---------------------------------------------------------------------------
# NFW conventions -- matches src/models/nfw_dsigma_mis.hh exactly
# ---------------------------------------------------------------------------
CONC = 4.0
RHOC = 2.77533742639e11        # Msun/h / (Mpc/h)^3


def delta_c(c):
    return (200.0 / 3.0) * c**3 / (np.log(1.0 + c) - c / (1.0 + c))


DELTA_C = delta_c(CONC)


def r200_of_M(M):
    return np.cbrt(3.0 * M / (800.0 * np.pi * RHOC))


def rs_of_M(M):
    return r200_of_M(M) / CONC


def amplitude_of_M(M):
    """A(M) such that DeltaSigma_NFW(R, M) = A(M) * g(R / rs(M))."""
    return 2.0 * rs_of_M(M) * DELTA_C * RHOC


# ---------------------------------------------------------------------------
# Wright & Brainerd (2000) NFW excess-surface-density shape function g(x)
# ---------------------------------------------------------------------------
def g_shape(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)

    lo = x < 1.0 - 1e-6
    hi = x > 1.0 + 1e-6
    mid = ~(lo | hi)

    xl = x[lo]
    yl = np.sqrt(1.0 - xl**2)
    al = np.arctanh(np.sqrt((1.0 - xl) / (1.0 + xl)))
    out[lo] = (8.0 * al / (xl**2 * yl) + 4.0 / xl**2 * np.log(xl / 2.0)
               - 2.0 / (xl**2 - 1.0) + 4.0 * al / ((xl**2 - 1.0) * yl))

    xh = x[hi]
    yh = np.sqrt(xh**2 - 1.0)
    ah = np.arctan(np.sqrt((xh - 1.0) / (1.0 + xh)))
    out[hi] = (8.0 * ah / (xh**2 * yh) + 4.0 / xh**2 * np.log(xh / 2.0)
               - 2.0 / (xh**2 - 1.0) + 4.0 * ah / (xh**2 - 1.0)**1.5)

    out[mid] = 10.0 / 3.0 + 4.0 * np.log(0.5)
    return out


def check_g_shape_against_projection():
    """Independent check: project the 3-D NFW density numerically
    (rs=1, rho_s=1) and compare its *shape* against g_shape(x)."""
    def rho(r):
        return 1.0 / (r * (1.0 + r)**2)

    def sigma(x):
        integrand = lambda z: rho(np.sqrt(x**2 + z**2))
        val, _ = integrate.quad(integrand, 0.0, np.inf, limit=200)
        return 2.0 * val

    def sigma_bar(x):
        integrand = lambda xp: xp * sigma(xp)
        val, _ = integrate.quad(integrand, 1e-8, x, limit=200)
        return 2.0 * val / x**2

    xs = np.array([0.05, 0.1, 0.3, 0.5, 0.8, 0.95, 1.0, 1.05, 1.5, 2.0, 5.0, 10.0])
    dsig_numeric = np.array([sigma_bar(x) - sigma(x) for x in xs])
    dsig_formula = g_shape(xs)
    ratio = dsig_numeric / dsig_formula
    print("g_shape() vs. independent numerical NFW projection:")
    print(f"  x        ratio (should be ~constant)")
    for x, r in zip(xs, ratio):
        print(f"  {x:6.3f}  {r:.6f}")
    rel_scatter = ratio.std() / ratio.mean()
    print(f"  relative scatter of ratio: {rel_scatter:.2e}  (constant => g_shape is correct)")
    assert rel_scatter < 1e-3, "g_shape() does not match the independent NFW projection"
    return ratio.mean()


# ---------------------------------------------------------------------------
# L = d/dlnx derivatives of g, via a dense spline in lnx (robust across x=1)
# ---------------------------------------------------------------------------
from scipy.interpolate import CubicSpline

_LNX_GRID = np.linspace(np.log(1e-3), np.log(1e3), 4000)
_G_GRID = g_shape(np.exp(_LNX_GRID))
_SPLINE = CubicSpline(_LNX_GRID, _G_GRID)


def L_derivs(x, nmax):
    """Return [g, Lg, L^2g, ..., L^nmax g] at x, L = d/dlnx."""
    lnx = np.log(x)
    out = [_SPLINE(lnx, k) for k in range(nmax + 1)]
    return out


def one_minus_L_pow(x, n):
    """[(1-L)^n g](x) via the binomial theorem in L-derivatives."""
    from math import comb
    Ls = L_derivs(x, n)
    return sum((-1)**k * comb(n, k) * Ls[k] for k in range(n + 1))


# ---------------------------------------------------------------------------
# Representative mass--redshift weight Wij(lnM): Tinker-like HMF shape x
# lognormal mass--richness kernel of width sigma_lnM, centered at M0.
# ---------------------------------------------------------------------------
LNM_GRID = np.linspace(np.log(5e12), np.log(3e15), 3000)
M_GRID = np.exp(LNM_GRID)


def hmf_shape(M, alpha=0.9, Mstar=2.0e14, beta=1.3):
    return (M / Mstar)**(-alpha) * np.exp(-(M / Mstar)**beta)


def weight_ij(lnM, M0, sigma_lnM):
    M = np.exp(lnM)
    sel = np.exp(-0.5 * (lnM - np.log(M0))**2 / sigma_lnM**2)
    return hmf_shape(M) * sel


BINS = {
    "narrow": dict(M0=3.0e14, sigma_lnM=0.30, color=C_IDEA1),
    "medium": dict(M0=1.2e14, sigma_lnM=0.45, color=C_IDEA2),
    "wide":   dict(M0=0.7e14, sigma_lnM=0.60, color=C_IDEA3),
}

RS_GRID = rs_of_M(M_GRID)
Y_GRID = np.log(RS_GRID)
A_GRID = amplitude_of_M(M_GRID)

R_GRID = np.logspace(np.log10(0.05), np.log10(20.0), 40)  # Mpc/h


def exact_profile(W):
    """N(R) = int dlnM W(lnM) A(M) g(R/rs(M)), on R_GRID."""
    out = np.empty_like(R_GRID)
    for i, R in enumerate(R_GRID):
        integrand = W * A_GRID * g_shape(R / RS_GRID)
        out[i] = np.trapz(integrand, LNM_GRID)
    return out


def idea1_params(W):
    """y_eff is the PLAIN mean of ln(rs) under the bin's own mass weight W
    (no amplitude weighting) -- this is what the exact Taylor identity
    (eq:deriv_identity) requires for the linear term to vanish. A_eff is
    the amplitude evaluated AT that mean scale radius, C*exp(y_eff), not
    a separately amplitude-weighted average (those differ by Jensen's
    inequality: <e^y> != e^<y>)."""
    norm = np.trapz(W, LNM_GRID)
    y_eff = np.trapz(W * Y_GRID, LNM_GRID) / norm
    C = 2.0 * DELTA_C * RHOC
    A_eff = C * np.exp(y_eff)
    return A_eff, y_eff


def moments(W, y_eff, nmax=4):
    """Plain central moments of ln(rs) about y_eff, under W alone."""
    norm = np.trapz(W, LNM_GRID)
    mus = {}
    for n in range(nmax + 1):
        mus[n] = np.trapz(W * (Y_GRID - y_eff)**n, LNM_GRID) / norm
    return mus


def idea1_profile(A_eff, y_eff):
    rs_eff = np.exp(y_eff)
    return A_eff * g_shape(R_GRID / rs_eff)


def idea2_profile(A_eff, y_eff, mus, nmax):
    from math import factorial
    rs_eff = np.exp(y_eff)
    x = R_GRID / rs_eff
    out = np.zeros_like(R_GRID)
    for n in range(0, nmax + 1):
        if n == 1:
            continue
        out += (mus[n] / factorial(n)) * one_minus_L_pow(x, n)
    return A_eff * out


# ---- Idea 3: Chebyshev expansion of the kernel K(R,y) in y, shared box ----
# LNM_GRID already spans 5e12-3e15 Msun/h, comfortably covering every bin's
# weight to well beyond +-4 sigma_lnM, so reuse its y-range as the shared box.
Y_MIN_GLOBAL = Y_GRID.min()
Y_MAX_GLOBAL = Y_GRID.max()
K_ORDER = 8


def to_tilde(y):
    return 2.0 * (y - Y_MIN_GLOBAL) / (Y_MAX_GLOBAL - Y_MIN_GLOBAL) - 1.0


def kernel_K(R, y):
    rs = np.exp(y)
    C = 2.0 * DELTA_C * RHOC
    return C * rs * g_shape(R / rs)


# u_k(R): Chebyshev coefficients of K(R, .) in ytilde, on a dense y grid
_Y_DENSE = np.linspace(Y_MIN_GLOBAL, Y_MAX_GLOBAL, 400)
_YT_DENSE = to_tilde(_Y_DENSE)
U_COEF = np.zeros((len(R_GRID), K_ORDER + 1))
for i, R in enumerate(R_GRID):
    Kvals = kernel_K(R, _Y_DENSE)
    U_COEF[i] = np.polynomial.chebyshev.chebfit(_YT_DENSE, Kvals, deg=K_ORDER)


def idea3_profile(W):
    yt = to_tilde(Y_GRID)
    Tmat = np.polynomial.chebyshev.chebvander(yt, K_ORDER)   # (n_lnM, K+1)
    c_k = np.trapz(W[:, None] * Tmat, LNM_GRID, axis=0)       # (K+1,)
    return U_COEF @ c_k


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def main():
    scale = check_g_shape_against_projection()

    results = {}
    for name, spec in BINS.items():
        W = weight_ij(LNM_GRID, spec["M0"], spec["sigma_lnM"])
        norm = np.trapz(W, LNM_GRID)
        W = W / norm  # normalize so N_counts = 1 (only shapes/ratios matter)

        exact = exact_profile(W)
        A_eff, y_eff = idea1_params(W)
        mus = moments(W, y_eff, nmax=4)

        i1 = idea1_profile(A_eff, y_eff)
        i2a = idea2_profile(A_eff, y_eff, mus, nmax=2)
        i2b = idea2_profile(A_eff, y_eff, mus, nmax=3)
        i3 = idea3_profile(W)

        results[name] = dict(
            spec=spec, exact=exact, idea1=i1, idea2_n2=i2a, idea2_n3=i2b,
            idea3=i3, mu2=mus[2], mu3=mus[3],
        )
        print(f"\nbin={name}: sigma_lnM={spec['sigma_lnM']:.2f}  "
              f"mu2={mus[2]:.4f}  mu3={mus[3]:.4f}  "
              f"rs_eff={np.exp(y_eff):.3f} Mpc/h")
        for label, prof in [("Idea 1", i1), ("Idea 2 (n<=2)", i2a),
                             ("Idea 2 (n<=3)", i2b), ("Idea 3", i3)]:
            frac = np.abs(prof - exact) / exact
            print(f"  {label:14s}  max frac err = {frac.max()*100:6.3f}%  "
                  f"rms = {np.sqrt(np.mean(frac**2))*100:6.3f}%")

    make_profile_figure(results)
    make_scaling_figure(results)


def make_profile_figure(results):
    order = ["narrow", "medium", "wide"]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.4), sharex="col",
                              gridspec_kw=dict(height_ratios=[2.0, 1.1], hspace=0.08, wspace=0.32))

    for col, name in enumerate(order):
        r = results[name]
        spec = r["spec"]
        ax, axr = axes[0, col], axes[1, col]

        ax.plot(R_GRID, r["exact"], color=INK, lw=2.0, label="exact", zorder=5)
        ax.plot(R_GRID, r["idea1"], color=C_IDEA1, lw=1.6, ls="--", label="Idea 1")
        ax.plot(R_GRID, r["idea2_n2"], color=C_IDEA2, lw=1.6, ls="-.", label="Idea 2 ($n\\leq 2$)")
        ax.plot(R_GRID, r["idea3"], color=C_IDEA3, lw=1.6, ls=":", label="Idea 3")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(rf"{name} bin: $\sigma_{{\ln M}}={spec['sigma_lnM']:.2f}$",
                      color=INK, fontsize=11)
        if col == 0:
            ax.set_ylabel(r"$N_{ij}(R)$ [arb. units]", color=INK)
            ax.legend(frameon=False, fontsize=8.5, loc="lower left")
        ax.tick_params(labelsize=8.5)

        for label, prof, color in [
            ("Idea 1", r["idea1"], C_IDEA1),
            ("Idea 2", r["idea2_n2"], C_IDEA2),
            ("Idea 3", r["idea3"], C_IDEA3),
        ]:
            frac = 100.0 * (prof - r["exact"]) / r["exact"]
            axr.plot(R_GRID, frac, color=color, lw=1.6)
        axr.axhline(0, color=MUTED, lw=0.8)
        axr.set_xscale("log")
        axr.set_xlabel(r"$R\ [\mathrm{Mpc}/h]$", color=INK)
        if col == 0:
            axr.set_ylabel("residual [%]", color=INK)
        axr.tick_params(labelsize=8.5)
        ymax = np.max(np.abs([100*(r["idea1"]-r["exact"])/r["exact"],
                              100*(r["idea2_n2"]-r["exact"])/r["exact"],
                              100*(r["idea3"]-r["exact"])/r["exact"]]))
        axr.set_ylim(-1.15*ymax - 0.05, 1.15*ymax + 0.05)

    for ax in axes.flat:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle(r"$\langle\Delta\Sigma_{1h}(R)\rangle_{ij}$: exact mass integral vs. the three radial-factorization ideas",
                 color=INK, fontsize=12, y=1.02)
    fig.savefig("shear1h_validation_profiles.png", bbox_inches="tight")
    print("wrote shear1h_validation_profiles.png")


def make_scaling_figure(results):
    order = ["narrow", "medium", "wide"]
    sigmas = np.array([results[n]["spec"]["sigma_lnM"] for n in order])

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for label, key, color in [
        ("Idea 1", "idea1", C_IDEA1),
        ("Idea 2 ($n\\leq 2$)", "idea2_n2", C_IDEA2),
        ("Idea 3", "idea3", C_IDEA3),
    ]:
        max_err = []
        for n in order:
            r = results[n]
            frac = np.abs(r[key] - r["exact"]) / r["exact"]
            max_err.append(frac.max() * 100.0)
        ax.plot(sigmas, max_err, marker="o", ms=5.5, lw=1.8, color=color, label=label)

    ax.set_xscale("linear")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\sigma_{\ln M}$ of the bin", color=INK)
    ax.set_ylabel("max fractional residual [%]", color=INK)
    ax.set_title("Error growth with mass-bin width", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig("shear1h_validation_scaling.png", bbox_inches="tight")
    print("wrote shear1h_validation_scaling.png")


if __name__ == "__main__":
    main()
