#!/usr/bin/env python
"""Omega(z) validation figure for issue #8 / des-nersc-cluster-scripts#1.

Two-panel figure (omega_z_buzzard_validation.png):

  Top    -- Omega_Y1(z) in deg^2: the pipeline's hardcoded
            y3_cluster::OMEGA_Z_DES polynomial vs the copy in Tan Xing's
            MockDataVector.ipynb (cell 26) -- expected identical to
            double precision (the max |Delta| is printed and annotated) --
            plus her cell-27 volume-weighted per-bin averages <Omega>_bin
            for reference (the FINAL DV uses the per-halo weighting of
            cell 28, i.e. the smooth curve itself).

  Bottom -- the area fraction actually applied to the data,
            w(z) = Omega_Y1(z) / Omega_buzz, which is exactly the weight
            the model reproduces by carrying Omega_Y1(z) in-integral.
            If the NSIDE-robust footprint measurement has been run on the
            halo catalog (des-nersc-cluster-scripts
            validations/cache/buzzard_area_healpix.csv, issue nersc#3),
            its per-slice corrected footprint fraction is overlaid --
            flat slices mean Omega_buzz is a single number and the model
            treatment is complete; a trend would require Omega_buzz(z).

Buzzard box seam [0.33, 0.37] and the observed-z bin edges
(0.20/0.33, 0.37/0.50, 0.50/0.65 -- xtang126 npz z_bin_min/max) are
marked on both panels.

Run:  python -B docs/figs/make_omega_z_validation_figs.py
"""
from __future__ import annotations

import csv
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False

HERE = os.path.dirname(os.path.abspath(__file__))
AREA_CSV = os.path.expanduser(
    "~/Documents/Dev/github/des-nersc-cluster-scripts/validations/cache/"
    "buzzard_area_healpix.csv")

INK, SECONDARY, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
C_CODE, C_NB, C_BIN = "#2a78d6", "#eb6834", "#1baf7a"

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

RAD2_TO_DEG2 = (180.0 / np.pi) ** 2
OMEGA_BUZZ_DEG2 = 5025.6      # Tan Xing's NSIDE=256 raw value (cell 27);
                              # refine with the corrected plateau (nersc#3)
SEAM = (0.33, 0.37)
ZBIN_LO, ZBIN_HI = (0.20, 0.37, 0.50), (0.33, 0.50, 0.65)
H0_BUZZ, OM0_BUZZ = 70.0, 0.286


def omega_z_code(z):
    """y3_cluster::OMEGA_Z_DES (src/models/omega_z_des.hh), rad^2."""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    coef1 = [0.0, 0.0, 0.0, -0.00262353, 0.01940118, 0.45133063]
    coef2 = [1.33647377e4, 1.35291046e3, -1.26204891e2,
             -2.83454918e1, -2.26465905, 3.84958753e-1]
    coef3 = [0.0, 0.0, -1.88101967, 4.8071839, -4.11424324, 1.18196785]
    out = np.empty_like(z)
    m1, m2 = z < 0.504, (z >= 0.504) & (z < 0.7)
    out[m1] = np.polyval(coef1, z[m1])
    out[m2] = np.polyval(coef2, z[m2] - 0.6)
    out[~m1 & ~m2] = np.polyval(coef3, z[~m1 & ~m2])
    return out


def omega_z_notebook(z):
    """Tan Xing's omega_z_des_y1 (MockDataVector.ipynb cell 26), rad^2.
    Re-typed from the notebook, NOT an alias of omega_z_code -- the whole
    point is to compare the two sources."""
    z_in = np.asarray(z, dtype=float)
    z = np.atleast_1d(z_in)
    omega = np.empty_like(z)
    coef1 = [0.0, 0.0, 0.0, -0.00262353, 0.01940118, 0.45133063]
    coef2 = [1.33647377e+4, 1.35291046e+3, -1.26204891e+2,
             -2.83454918e+1, -2.26465905, 3.84958753e-1]
    coef3 = [0.0, 0.0, -1.88101967, 4.8071839, -4.11424324, 1.18196785]
    m1 = z < 0.504
    m2 = (z >= 0.504) & (z < 0.7)
    m3 = z >= 0.7
    omega[m1] = np.polyval(coef1, z[m1])
    omega[m2] = np.polyval(coef2, z[m2] - 0.6)
    omega[m3] = np.polyval(coef3, z[m3])
    return omega


def vol_weighted_averages(edges):
    """dV/dOmega/dz-weighted <Omega_Y1> per [edges] interval (Buzzard
    cosmology) -- the cell-27 method, at whatever slice width is asked."""
    from astropy.cosmology import FlatLambdaCDM
    from scipy.integrate import quad
    cosmo = FlatLambdaCDM(H0=H0_BUZZ, Om0=OM0_BUZZ)

    def dV(z):
        return cosmo.differential_comoving_volume(z).to('Mpc^3/sr').value

    out = []
    for zl, zh in zip(edges[:-1], edges[1:]):
        n, _ = quad(lambda z: float(omega_z_notebook([z])[0]) * dV(z),
                    zl, zh)
        d, _ = quad(dV, zl, zh)
        out.append(n / d)
    return np.array(out)


DZ_SLICE = 0.01
SLICE_EDGES = np.arange(0.20, 0.65 + 1e-9, DZ_SLICE)


def read_slice_fractions():
    """(z_mid, corrected_area_deg2) from the nersc#3 area tool, if run."""
    if not os.path.exists(AREA_CSV):
        return None
    zs, areas = [], []
    with open(AREA_CSV) as f:
        for row in csv.DictReader(f):
            if row.get("kind") == "slice":
                zs.append(float(row["z_mid"]))
                areas.append(float(row["corrected_deg2"]))
    if not zs:
        return None
    return np.array(zs), np.array(areas)


def main():
    z = np.linspace(0.15, 0.70, 800)
    om_code = omega_z_code(z) * RAD2_TO_DEG2
    om_nb = omega_z_notebook(z) * RAD2_TO_DEG2
    max_dev = float(np.max(np.abs(om_nb - om_code)))
    om_sl = vol_weighted_averages(SLICE_EDGES) * RAD2_TO_DEG2

    fig, (ax1, axr, ax2) = plt.subplots(
        3, 1, figsize=(7.2, 9.0), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.4, 2.6]})

    for ax in (ax1, axr, ax2):
        ax.axvspan(*SEAM, color=MUTED, alpha=0.18, lw=0, zorder=0)
        for zl, zh in zip(ZBIN_LO, ZBIN_HI):
            ax.axvline(zl, color=GRID, lw=0.8, zorder=0)
            ax.axvline(zh, color=GRID, lw=0.8, zorder=0)

    # --- top: Omega_Y1(z), code vs notebook -------------------------------
    ax1.plot(z, om_code, color=C_CODE, lw=2.5,
             label="pipeline code: OMEGA_Z_DES (omega_z_des.hh)")
    ax1.plot(z, om_nb, color=C_NB, lw=2.5, ls=(0, (5, 4)),
             label="mock: MockDataVector.ipynb cell 26")
    for i, (zl, zh, ob) in enumerate(zip(SLICE_EDGES[:-1], SLICE_EDGES[1:],
                                         om_sl)):
        ax1.hlines(ob, zl, zh, color=C_BIN, lw=2.0,
                   label=(r"$\langle\Omega\rangle$ per $\Delta z=%.2f$"
                          r" slice (cell-27 method)" % DZ_SLICE
                          if i == 0 else None))
    ax1.annotate(f"max |code $-$ notebook| = {max_dev:.2e} deg$^2$\n"
                 "(identical polynomial, verified)",
                 xy=(0.03, 0.08), xycoords="axes fraction", fontsize=9,
                 color=SECONDARY)
    ax1.annotate("box seam\n[0.33, 0.37]", xy=(0.35, 0.55),
                 xycoords=("data", "axes fraction"), fontsize=8,
                 color=SECONDARY, ha="center")
    ax1.set_ylabel(r"$\Omega_{\rm Y1}(z)$  [deg$^2$]")
    ax1.set_ylim(0, 1900)
    ax1.legend(loc="lower left", fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.02, 0.22))

    # --- middle: relative error in percent ---------------------------------
    rel_nb = 100.0 * (om_nb / om_code - 1.0)
    axr.plot(z, rel_nb, color=C_NB, lw=2.0,
             label="mock cell-26 / code $-$ 1")
    worst = 0.0
    for i, (zl, zh, ob) in enumerate(zip(SLICE_EDGES[:-1], SLICE_EDGES[1:],
                                         om_sl)):
        m = (z >= zl) & (z < zh)
        rel_sl = 100.0 * (ob / om_code[m] - 1.0)
        axr.plot(z[m], rel_sl, color=C_BIN, lw=2.0,
                 label=(r"$\langle\Omega\rangle_{\Delta z=%.2f}\,/\,"
                        r"\Omega_{\rm Y1}(z) - 1$" % DZ_SLICE
                        if i == 0 else None))
        worst = max(worst, float(np.max(np.abs(rel_sl))))
        print(f"slice [{zl:.2f},{zh:.2f}): <Omega>/Omega(z)-1 in "
              f"[{rel_sl.min():+.2f}%, {rel_sl.max():+.2f}%]")
    print(f"worst |<Omega>_slice/Omega(z)-1| over dz={DZ_SLICE}: "
          f"{worst:.2f}%")
    axr.axhline(0.0, color=MUTED, lw=0.8)
    axr.set_ylabel("rel. error [%]")
    axr.set_yscale("symlog", linthresh=1.0)
    axr.set_yticks([-10, -1, 0, 1, 10, 100])
    axr.legend(loc="upper left", fontsize=8.5, frameon=False)
    axr.annotate("code vs mock: 0% everywhere (identical polynomial)",
                 xy=(0.03, 0.06), xycoords="axes fraction", fontsize=8.5,
                 color=SECONDARY)

    # --- bottom: the applied area fraction --------------------------------
    w = om_code / OMEGA_BUZZ_DEG2
    ax2.plot(z, w, color=C_CODE, lw=2.5,
             label=(r"applied weight $w(z)=\Omega_{\rm Y1}(z)/"
                    r"\Omega_{\rm buzz}$"
                    f"  ($\\Omega_{{\\rm buzz}}$={OMEGA_BUZZ_DEG2:.0f} deg$^2$,"
                    " NSIDE=256 raw)"))
    sl = read_slice_fractions()
    if sl is not None:
        zs, areas = sl
        ax2.plot(zs, areas / areas.mean(), color=C_BIN, lw=0, marker="o",
                 ms=5, label="Buzzard footprint fraction per 120-Mpc slice\n"
                             "(occupancy-corrected, nersc#3 tool)")
    else:
        ax2.annotate("Buzzard per-slice footprint measurement pending:\n"
                     "run des-nersc validations/buzzard_area_healpix.py on\n"
                     "the halo catalog (NERSC) and re-run this script",
                     xy=(0.03, 0.72), xycoords="axes fraction", fontsize=8.5,
                     color=SECONDARY)
    ax2.set_xlabel(r"redshift $z$")
    ax2.set_ylabel(r"area fraction")
    ax2.set_xlim(0.15, 0.70)
    ax2.legend(loc="lower left", fontsize=8.5, frameon=False)

    fig.suptitle("Buzzard survey-area model: code vs mock (issues y3#8, "
                 "nersc#1/#3)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(HERE, "omega_z_buzzard_validation.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    print(f"max |code - notebook| = {max_dev:.3e} deg^2")
    print(f"<Omega> per dz={DZ_SLICE} slice [deg^2]:",
          " ".join(f"{v:.1f}" for v in om_sl))


if __name__ == "__main__":
    main()
