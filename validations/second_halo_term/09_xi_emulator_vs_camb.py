#!/usr/bin/env python
"""Validate the xi produced from cp_camb's emulator P(k) against xi from
real CAMB, across several cosmologies.

The production xi_nl table is the cluster_toolkit Hankel transform of
whatever P(k) cp_camb publishes. cp_camb's linear emulator
(camb_linear_s8_v3c, z=0-only; P(k,z) = D(z)^2 P(k,0) downstream) is the
ONLY P(k) source in production, so its error propagates directly into
xi_NL, b_sel and shearPrj. This script measures that error at the xi
level: for each test cosmology, xi[emulator P] / xi[CAMB P] - 1 at z=0
(the growth scaling is applied identically to both downstream, so z=0
isolates the emulator residual; massless neutrinos make the growth
scale-independent).

Run with the conda pipeline env:
  /opt/homebrew/Caskroom/miniforge/base/envs/y3cl_je_macos/bin/python 09_xi_emulator_vs_camb.py
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "report", "figs")
EMU_REPO = "/Users/jesteves/Documents/Dev/github/camb-emulator/camb-for-cp"
EMU_NPZ = os.path.join(EMU_REPO, "models", "camb_linear_s8_v3c_emulator.npz")
sys.path.insert(0, EMU_REPO)

import camb  # noqa: E402
import cluster_toolkit as ct  # noqa: E402
from cp_numpy import CosmoPowerNumpyNN  # noqa: E402

FID = dict(h0=0.6766, omega_m=0.311049, omega_b=0.048975,
           n_s=0.9665, sigma8=0.8238, mnu=0.0)

# some cosmological values: the fiducial and one-at-a-time variations,
# all inside the emulator training box
COSMOS = [
    ("fiducial", dict(FID)),
    (r"$\Omega_m = 0.26$", dict(FID, omega_m=0.26)),
    (r"$\Omega_m = 0.36$", dict(FID, omega_m=0.36)),
    (r"$\sigma_8 = 0.75$", dict(FID, sigma8=0.75)),
    (r"$\sigma_8 = 0.90$", dict(FID, sigma8=0.90)),
    (r"$h = 0.72$", dict(FID, h0=0.72)),
]

R_XI = np.logspace(np.log10(0.3), np.log10(80.0), 90)   # cMpc/h
RFIX = np.logspace(-3, 3, 400)

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 9, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
})
# CVD-validated categorical order, fixed per cosmology
COLORS = ["#000000", "#0072B2", "#56B4E9", "#009E73", "#CC79A7", "#D55E00"]


def camb_pk_z0(c, k_h):
    """CAMB linear P(k, z=0) on k_h, sigma8-matched by As iteration."""
    As = 2.0e-9
    for _ in range(4):
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=100 * c["h0"],
                           ombh2=c["omega_b"] * c["h0"] ** 2,
                           omch2=(c["omega_m"] - c["omega_b"]) * c["h0"] ** 2,
                           mnu=0.0, num_massive_neutrinos=0, omk=0.0,
                           tau=0.0544)
        pars.InitPower.set_params(As=As, ns=c["n_s"])
        pars.set_matter_power(redshifts=[0.0], kmax=60.0 / c["h0"])
        res = camb.get_results(pars)
        s8 = float(res.get_sigma8_0())
        if abs(s8 - c["sigma8"]) < 1e-4:
            break
        As *= (c["sigma8"] / s8) ** 2
    pk = camb.get_matter_power_interpolator(
        pars, nonlinear=False, zmin=0.0, zmax=0.1, nz_step=3,
        kmax=60.0 / c["h0"], k_hunit=True, hubble_units=True,
        extrap_kmax=200.0)
    return pk.P(0.0, k_h), s8


def main():
    os.makedirs(FIGS, exist_ok=True)
    emu = CosmoPowerNumpyNN(EMU_NPZ)
    k_h = emu.modes

    fig, (axp, axx) = plt.subplots(2, 1, figsize=(5.2, 5.4), sharex=False)
    worst = 0.0
    for (label, c), col in zip(COSMOS, COLORS):
        # CosmoPower convention: the NN emits log10 P(k)
        p_emu = 10.0 ** emu.predictions_np(
            {k: np.atleast_1d(v) for k, v in c.items()})[0]
        p_camb, s8 = camb_pk_z0(c, k_h)
        xi_emu = ct.xi.xi_mm_at_r(R_XI, k_h, p_emu)
        xi_camb = ct.xi.xi_mm_at_r(R_XI, k_h, p_camb)
        res_p = p_emu / p_camb - 1.0
        res_x = xi_emu / xi_camb - 1.0
        # xi residual metric away from the zero crossing (~120 cMpc/h)
        m = R_XI <= 50.0
        worst = max(worst, float(np.max(np.abs(res_x[m]))))
        axp.semilogx(k_h, res_p, color=col, lw=1.4, label=label)
        axx.semilogx(R_XI, res_x, color=col, lw=1.4, label=label)
        print(f"{label:16s} sigma8_camb={s8:.4f} "
              f"max|P res|={np.max(np.abs(res_p)):.3%} "
              f"max|xi res| (r<=50)={np.max(np.abs(res_x[m])):.3%}")

    axp.axhline(0, color="k", lw=0.5)
    axp.set_xlabel(r"$k$ [$h$/Mpc]")
    axp.set_ylabel("emulator / CAMB $-1$\n(linear $P(k)$, $z=0$)")
    axp.set_ylim(-0.004, 0.004)
    axx.axhline(0, color="k", lw=0.5)
    axx.set_xlabel(r"$r$ [cMpc/$h$, comoving]")
    axx.set_ylabel("emulator / CAMB $-1$\n" r"($\xi_{mm}(r)$, $z=0$)")
    axx.set_ylim(-0.004, 0.004)
    axx.legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "xi_emulator_vs_camb.pdf"))
    plt.close(fig)

    with open(os.path.join(HERE, "report", "values_emulator.tex"), "w") as f:
        f.write("% generated by 09_xi_emulator_vs_camb.py -- do not edit\n")
        f.write(f"\\newcommand{{\\xiEmuWorst}}{{{worst:.1%}}}\n"
                .replace("%", "\\%"))
    print(f"worst xi residual across cosmologies (r <= 50): {worst:.2%}")
    print("wrote figure + report/values_emulator.tex")


if __name__ == "__main__":
    main()
