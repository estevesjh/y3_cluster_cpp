#!/usr/bin/env python
"""xi_NL consumers, current vs proposed fix.

'current'      = linear P(k) under the xi_nl name (the silent fallback,
                 all runs to date)
'proposed fix' = halofit matter_power_nl injected (common/inject_pk_nl.py),
                 everything downstream unchanged

Both arms are full pipeline runs of xinl_consumers.ini (cp_camb ->
halo_model -> b_sel_marg -> [wall-metadata shim] -> bsel ->
DSigmaPrjEvaluator -> ShearPrjFrozenPhysics). This script turns the two
dumps into the report's section-5 figures and macros:
  - b_sel(theta), marginalized over lambda_tr (the b_small/b_large closure
    + the shared sigmoid reconstruction used by every consumer)
  - DeltaSigma_prj(R) per richness/redshift bin, current vs proposed.

Run with the conda pipeline env after both cosmosis runs:
  python 08_xinl_consumers.py --current <dir> --proposed <dir>
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "report", "figs")
SCRATCH_DEFAULT = ("/private/tmp/claude-501/-Users-jesteves-Documents-Dev-"
                   "github-y3-cluster-cpp/4f385ffd-9da9-4451-a9e1-3234df5cfdfd"
                   "/scratchpad")

C = {"current": "#0072B2", "proposed": "#D55E00"}
LS = {"current": "--", "proposed": "-"}
plt.rcParams.update({
    "figure.dpi": 130, "font.size": 9, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
})

# the 180-point wall layout of xinl_consumers.ini ([dsigma_prj])
RADII = np.array([0.0426, 0.0669, 0.1045, 0.1652, 0.2607, 0.4117, 0.6505,
                  1.0257, 1.6181, 2.5537, 4.0265, 6.3490, 10.0107, 15.7832,
                  24.8771])
N_R = RADII.size
ZO_BINS = [(0.20, 0.35, 0.275), (0.35, 0.50, 0.425), (0.50, 0.65, 0.575)]
LOB_CENTRES = [25.0, 37.5, 52.5, 130.0]   # [bsel] lob


def bsel_row(dump, lam_bin, zob):
    lam = np.loadtxt(os.path.join(dump, "b_sel_marginalised", "lambda_bin.txt"))
    zob_v = np.loadtxt(os.path.join(dump, "b_sel_marginalised", "zob.txt"))
    bs = np.loadtxt(os.path.join(dump, "b_sel_marginalised", "b_small.txt"))
    bl = np.loadtxt(os.path.join(dump, "b_sel_marginalised", "b_large.txt"))
    lob = np.loadtxt(os.path.join(dump, "b_sel_marginalised", "lob.txt"))
    sel = (lam.astype(int) == lam_bin) & np.isclose(zob_v, zob)
    (i,) = np.flatnonzero(sel)
    return float(lob[i]), float(bs[i]), float(bl[i])


def bsel_of_theta(dump, lam_bin, zob, theta):
    """The shared sigmoid reconstruction every consumer applies
    (shear_prj_fast_mass.py / sigma_prj_t.hh convention)."""
    lob, bs, bl = bsel_row(dump, lam_bin, zob)
    z = np.loadtxt(os.path.join(dump, "distances", "z.txt"))
    d_c = np.loadtxt(os.path.join(dump, "distances", "d_c.txt"))
    h0 = 0.6766
    chi_o = np.interp(zob, z, d_c) * h0
    r_lam = (lob / 100.0) ** 0.2                # cMpc/h
    theta_lam = r_lam * (1.0 + zob) / chi_o
    k_sig, theta0 = 2.5 / theta_lam, 0.5 * theta_lam
    return bs + (bl - bs) / (1.0 + np.exp(-k_sig * (theta - theta0))), (bs, bl)


def prj_bin(dump, section, lam_bin, iz):
    vals = np.loadtxt(os.path.join(dump, section, "vals.txt"))
    start = iz * 4 * N_R + lam_bin * N_R
    return vals[start:start + N_R]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", default=os.path.join(SCRATCH_DEFAULT, "xinl_current"))
    ap.add_argument("--proposed", default=os.path.join(SCRATCH_DEFAULT, "xinl_proposed"))
    args = ap.parse_args()
    dumps = {"current": args.current, "proposed": args.proposed}
    os.makedirs(FIGS, exist_ok=True)
    macros = {}

    # ---------------- b_sel(theta), marginalized over lambda_tr ----------
    theta = np.logspace(-4.3, -1.9, 200)   # rad; spans the production grid
    iz_show, zob_show = 1, ZO_BINS[1][2]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, lam_bin in zip(axes, (0, 3)):
        for arm in ("current", "proposed"):
            b, (bs, bl) = bsel_of_theta(dumps[arm], lam_bin, zob_show, theta)
            label = (f"{'current (linear)' if arm == 'current' else 'proposed fix (halofit)'}"
                     f":  $b_s$={bs:.1f}, $b_\\ell$={bl:.2f}")
            ax.semilogx(theta, b, C[arm], ls=LS[arm], lw=1.8, label=label)
            if lam_bin == 0:
                macros[f"bSmall{arm.capitalize()}"] = f"{bs:.1f}"
                macros[f"bLarge{arm.capitalize()}"] = f"{bl:.2f}"
        lo, hi = (20, 30) if lam_bin == 0 else (60, 200)
        ax.set_title(rf"$\lambda^{{\rm ob}} \in [{lo}, {hi}]$, "
                     rf"$z^{{\rm ob}} = {zob_show}$", fontsize=8)
        ax.set_xlabel(r"$\theta$ [rad]")
        ax.legend(fontsize=6.5)
    axes[0].set_ylabel(r"$b_{\rm sel}(\theta)$ "
                       r"(marginalized over $\lambda^{\rm tr}$)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "bsel_theta_current_proposed.pdf"))
    plt.close(fig)
    bs_c = float(macros["bSmallCurrent"])
    bs_p = float(macros["bSmallProposed"])
    macros["bSmallShift"] = f"{bs_p / bs_c - 1:+.0%}"

    # ---------------- DeltaSigma_prj(R), current vs proposed --------------
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(4.8, 4.8), sharex=True,
                                  height_ratios=[2, 1])
    worst = 0.0
    for lam_bin, alpha in ((0, 1.0), (3, 0.55)):
        lo, hi = (20, 30) if lam_bin == 0 else (60, 200)
        for arm in ("current", "proposed"):
            v = prj_bin(dumps[arm], "dsigma_prj", lam_bin, iz_show)
            lab = (f"{'current' if arm == 'current' else 'proposed fix'}, "
                   rf"$\lambda^{{\rm ob}} \in [{lo}, {hi}]$")
            ax.loglog(RADII, v, C[arm], ls=LS[arm], lw=1.8, alpha=alpha,
                      label=lab)
        ratio = (prj_bin(dumps["proposed"], "dsigma_prj", lam_bin, iz_show)
                 / prj_bin(dumps["current"], "dsigma_prj", lam_bin, iz_show))
        worst = max(worst, float(np.max(np.abs(ratio - 1))))
        axr.semilogx(RADII, ratio - 1, C["proposed"], alpha=alpha, lw=1.8)
    ax.set_ylabel(r"$\Delta\Sigma^{\rm prj}(R)$ [$M_\odot h/\mathrm{pc}^2$]")
    ax.set_title(rf"$z^{{\rm ob}} \in [{ZO_BINS[1][0]}, {ZO_BINS[1][1]}]$",
                 fontsize=8)
    ax.legend(fontsize=6.5)
    axr.axhline(0, color="k", lw=0.5)
    axr.set_ylim(-0.15, 0.15)
    axr.set_xlabel(r"$R$ [cMpc/$h$, comoving]")
    axr.set_ylabel("proposed / current $-1$")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "dsigmaprj_current_proposed.pdf"))
    plt.close(fig)
    macros["dsPrjMaxShift"] = f"{worst:.0%}"

    # xi table sanity: proposed xi_nl must exceed current at small r
    xi_c = np.loadtxt(os.path.join(dumps["current"], "xi_nl", "xi_nl.txt"))
    xi_p = np.loadtxt(os.path.join(dumps["proposed"], "xi_nl", "xi_nl.txt"))
    r = np.loadtxt(os.path.join(dumps["current"], "xi_nl", "r.txt"))
    i1 = np.argmin(np.abs(r - 1.0))
    macros["xiTableNlLinAtROne"] = f"{xi_p[5, i1] / xi_c[5, i1]:.2f}"

    with open(os.path.join(HERE, "report", "values_consumers.tex"), "w") as f:
        f.write("% generated by 08_xinl_consumers.py -- do not edit\n")
        for k, v in macros.items():
            f.write(f"\\newcommand{{\\{k}}}{{{str(v).replace('%', chr(92)+'%')}}}\n")
    print("macros:", macros)
    print("wrote figures + report/values_consumers.tex")


if __name__ == "__main__":
    main()
