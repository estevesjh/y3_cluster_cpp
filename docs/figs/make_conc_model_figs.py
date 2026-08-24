#!/usr/bin/env python
"""Concentration-model comparison for the projection profile (issue #13).

Three panels (conc_model_comparison.png):

  Top    -- c(M) over the production mass range at z = 0.46 (the
            Buzzard one_halo_z): Child18 stacked_nfw (adopted),
            Duffy08 (the benchmark power law), legacy fixed c = 4.
  Middle -- profile-level impact: DeltaSigma_NFW(c(M)) / DeltaSigma(c=4)
            - 1 [%] vs R (cluster_toolkit, fiducial Omega_m) for both
            relations at three masses spanning the range.
  Bottom -- measured END-TO-END production impact: dsigma_prj per-radius
            mean shift (rnd / cl / vals) with use_halo_model_conc=T +
            one_halo_z=0.46 vs the legacy configuration, read from the
            scratch dumps of the validation runs (annotated as pending
            if those dumps are absent -- regenerate per the #13 issue
            comment).

Run:  python -B docs/figs/make_conc_model_figs.py
"""
from __future__ import annotations

import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "y3_buzzard"))

try:
    import seaborn as sns
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False

INK, SECONDARY, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
C_CHILD, C_DUFFY, C_FIX = "#2a78d6", "#eb6834", "#898781"

if _HAS_SNS:
    sns.set_theme(style="white", rc={
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": SECONDARY, "ytick.color": SECONDARY,
        "font.family": "sans-serif",
    })
mpl.rcParams.update({
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "figure.dpi": 150, "savefig.dpi": 200, "axes.linewidth": 0.8,
})

Z_EFF = 0.46
OMEGA_M = 0.311049
LNM_LO, LNM_HI = 29.9336, 35.6814          # production projection range
SCRATCH = os.environ.get(
    "CONC_DUMP_DIR",
    "/private/tmp/claude-501/-Users-jesteves-Documents-Dev-github-y3-cluster-cpp/"
    "e8de96a0-0204-461b-aeaf-6e931f305c1d/scratchpad")


def main():
    from haloModel import (child18_mass_concentration,
                           duffy_concentration_relation)
    import cluster_toolkit as ct

    lnM = np.linspace(LNM_LO, LNM_HI, 200)
    M = np.exp(lnM)
    c_child = child18_mass_concentration(M, Z_EFF, halo_sample="stacked_nfw")
    c_duffy = duffy_concentration_relation(M, z_eff=Z_EFF)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7.2, 10.0))

    # --- top: c(M) ---------------------------------------------------------
    ax1.plot(lnM, c_child, color=C_CHILD, lw=2.5,
             label="Child18 stacked_nfw (adopted, haloModel/concentration)")
    ax1.plot(lnM, c_duffy, color=C_DUFFY, lw=2.5,
             label="Duffy08 (benchmark power law)")
    ax1.axhline(4.0, color=C_FIX, lw=2.0, ls=(0, (5, 4)),
                label="legacy fixed c = 4 (NFW_DSIGMA_MIS)")
    ax1.set_xlabel(r"$\ln M\ [M_\odot/h]$")
    ax1.set_ylabel(r"$c(M,\,z=%.2f)$" % Z_EFF)
    ax1.legend(fontsize=8.5, frameon=False)

    # --- middle: profile-level DeltaSigma ratio ----------------------------
    R = np.logspace(np.log10(0.05), np.log10(25.0), 60)
    r3d = np.logspace(-3, 2.5, 800)
    for lnm_i, ls in zip((31.0, 33.5, 35.5),
                         ("-", (0, (5, 3)), (0, (1, 1.5)))):
        M_i = np.exp(lnm_i)
        ds4 = ct.deltasigma.DeltaSigma_at_R(
            R, r3d, ct.deltasigma.Sigma_nfw_at_R(r3d, M_i, 4.0, OMEGA_M),
            M_i, 4.0, OMEGA_M)
        for rel, cfun, color in (
                ("Child18", child18_mass_concentration, C_CHILD),
                ("Duffy08", duffy_concentration_relation, C_DUFFY)):
            c_i = (float(cfun(np.array([M_i]), Z_EFF,
                              halo_sample="stacked_nfw")[0])
                   if rel == "Child18"
                   else float(cfun(np.array([M_i]), z_eff=Z_EFF)[0]))
            ds = ct.deltasigma.DeltaSigma_at_R(
                R, r3d, ct.deltasigma.Sigma_nfw_at_R(r3d, M_i, c_i, OMEGA_M),
                M_i, c_i, OMEGA_M)
            ax2.plot(R, 100.0 * (ds / ds4 - 1.0), color=color, ls=ls, lw=2.0,
                     label=(f"{rel}, lnM={lnm_i:.1f} (c={c_i:.2f})"))
    ax2.axhline(0.0, color=MUTED, lw=0.8)
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$R$  [cMpc/$h$]")
    ax2.set_ylabel(r"$\Delta\Sigma(c_{\rm rel})/\Delta\Sigma(c{=}4)-1$  [%]")
    ax2.legend(fontsize=7.5, frameon=False, ncol=2)

    # --- bottom: measured end-to-end production shift ----------------------
    off, on = f"{SCRATCH}/dump_conc_off", f"{SCRATCH}/dump_conc_on"
    have = os.path.isdir(off) and os.path.isdir(on)
    if have:
        radii = sorted(set(np.loadtxt(
            os.path.join(HERE, "..", "..", "cosmosis-models",
                         "real_pipeline_extract_prj2h_output",
                         "shear_prj", "vals.txt")) * 0 + 1))  # placeholder
        # wall radii: 15 log radii repeated 12x -- reconstruct from ini order
        import re
        text = open(os.path.join(HERE, "..", "..", "cosmosis-models",
                                 "real_pipeline_extract_prj2h.ini")).read()
        sec = re.search(r"^\[dsigma_prj\]\s*$(.*?)(?:^\[|\Z)", text,
                        re.S | re.M).group(1)
        rw = np.array([float(v) for v in re.search(
            r"^radii\s*=\s*(.+)$", sec, re.M).group(1).split()])
        r_axis = rw.reshape(12, 15)[0]
        for ch, color, ls in (("rnd", C_DUFFY, (0, (5, 3))),
                              ("cl", C_CHILD, (0, (1, 1.5))),
                              ("vals", INK, "-")):
            o = np.loadtxt(f"{off}/dsigma_prj/{ch}.txt")
            n = np.loadtxt(f"{on}/dsigma_prj/{ch}.txt")
            shift = 100.0 * (n / o - 1.0).reshape(12, 15).mean(axis=0)
            ax3.plot(r_axis, shift, color=color, ls=ls, lw=2.0,
                     label=f"dsigma_prj/{ch}")
        ax3.axhline(0.0, color=MUTED, lw=0.8)
        ax3.set_xscale("log")
        ax3.set_ylabel("Child18@z=0.46 vs legacy  [%]")
        ax3.set_xlabel(r"$R$  [cMpc/$h$]")
        ax3.legend(fontsize=8.5, frameon=False)
        ax3.annotate("production ShearPrjCore, bin-averaged\n"
                     "(use_halo_model_conc=T + one_halo_z=0.46)",
                     xy=(0.97, 0.9), xycoords="axes fraction", ha="right",
                     fontsize=8.5, color=SECONDARY)
    else:
        ax3.annotate("end-to-end dumps not found under $CONC_DUMP_DIR --\n"
                     "rerun the two extract-pipeline configs from the #13\n"
                     "issue comment and re-run this script",
                     xy=(0.05, 0.5), xycoords="axes fraction", fontsize=9,
                     color=SECONDARY)
        ax3.set_xticks([]); ax3.set_yticks([])

    fig.suptitle("Projection-profile concentration: Child18 vs Duffy08 vs "
                 "fixed c=4 (issue #13)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(HERE, "conc_model_comparison.png")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
