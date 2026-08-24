#!/usr/bin/env python
"""Fiducial-P(k) cluster_toolkit reference for the two-halo term.

Run with the conda pipeline env:
  /opt/homebrew/Caskroom/miniforge/base/envs/y3cl_je_macos/bin/python 04_reference_ct.py

The CORRECT per-z recipe (what the fixed production code must produce),
on converged grids:
  xi_mm(r, z)   = ct.xi.xi_mm_at_r(r, k, P[iz])          per z slice
  Sigma_2h(R,z) = ct.deltasigma.Sigma_at_R (b=1)          dense 400-pt Rfix
  DeltaSigma:
    - anchor  : dense-extended-grid cumtrapz interior mean (converged)
    - sandwich: repaired consistent-Md NFW sandwich on the r_sigma table
    - direct  : extended-grid cumtrapz, table from xi (production candidate)
Also the production-fidelity variants (Rfix = 50 pts as ct_2hTerm.NSIZE)
to isolate the NSIZE=50 grid error.

Units: r/R Mpc/h comoving, Sigma/DeltaSigma Msun h/pc^2 comoving, b=1.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "y3_buzzard"))

import cluster_toolkit as ct  # noqa: E402
from nfwModel import sigmaNFW_Analytical, deltaSigmaNFW_Analytical  # noqa: E402

RHO_C = 2.77533742639e11
OMEGA_M = 0.311049
RHO_M = OMEGA_M * RHO_C
MD, CD = 1e14, 5.0

R_SIGMA = np.logspace(np.log10(0.1), np.log10(20.0), 128)
RFIX_DENSE = np.logspace(-3, 3, 400)
RFIX_PROD = np.logspace(-3, 3, 50)          # ct_2hTerm.NSIZE = 50
R_EXT = np.logspace(-3, np.log10(20.0), 300)
IZ_SLICES = [0, 3, 5, 8]


def dsig_from_sigma_grid(r_grid, sig_grid, r_out):
    lnr = np.log(r_grid)
    integrand = sig_grid * r_grid ** 2
    cum = np.concatenate([[0.0], np.cumsum(
        0.5 * (integrand[1:] + integrand[:-1]) * np.diff(lnr))])
    cum += 0.5 * sig_grid[0] * r_grid[0] ** 2   # flat inner disc
    sig_bar = 2.0 * cum / r_grid ** 2
    return np.interp(np.log(r_out), lnr, sig_bar - sig_grid)


def dsig_sandwich(sig_table, md=MD, cd=CD):
    sig_nfw = sigmaNFW_Analytical(R_SIGMA, md, cd, rho_c=RHO_M) / 1e12
    dsig_nfw = deltaSigmaNFW_Analytical(R_SIGMA, md, cd, rho_c=RHO_M) / 1e12
    ds = ct.deltasigma.DeltaSigma_at_R(R_SIGMA, R_SIGMA, sig_table + sig_nfw,
                                       md, cd, OMEGA_M)
    return ds - dsig_nfw


def per_z(k_h, p_row):
    """All reference quantities for one z slice."""
    out = {}
    xi_dense = ct.xi.xi_mm_at_r(RFIX_DENSE, k_h, p_row)
    out["xi_dense"] = xi_dense
    out["xi_prod"] = ct.xi.xi_mm_at_r(RFIX_PROD, k_h, p_row)

    sig = ct.deltasigma.Sigma_at_R(R_SIGMA, RFIX_DENSE, xi_dense, MD, CD, OMEGA_M)
    out["sigma"] = sig
    out["sigma_prodgrid"] = ct.deltasigma.Sigma_at_R(
        R_SIGMA, RFIX_PROD, out["xi_prod"], MD, CD, OMEGA_M)

    # converged anchor: Sigma on a dense grid extended well below r_sigma_min
    r_anchor = np.logspace(-3, np.log10(20.0), 600)
    sig_anchor = ct.deltasigma.Sigma_at_R(r_anchor, RFIX_DENSE, xi_dense,
                                          MD, CD, OMEGA_M)
    out["dsigma_anchor"] = dsig_from_sigma_grid(r_anchor, sig_anchor, R_SIGMA)

    # method candidates fed from the same xi
    out["dsigma_sandwich"] = dsig_sandwich(sig)
    out["dsigma_sandwich_md13"] = dsig_sandwich(sig, md=1e13, cd=4.0)
    sig_ext = ct.deltasigma.Sigma_at_R(R_EXT, RFIX_DENSE, xi_dense, MD, CD, OMEGA_M)
    out["dsigma_direct"] = dsig_from_sigma_grid(R_EXT, sig_ext, R_SIGMA)
    return out


def main():
    d = np.load(os.path.join(HERE, "outputs", "pk_camb.npz"), allow_pickle=False)
    k_h, z = d["k_h"], d["z"]

    results = {"r_sigma": R_SIGMA, "rfix_dense": RFIX_DENSE,
               "rfix_prod": RFIX_PROD, "z": z,
               "iz_slices": np.array(IZ_SLICES)}
    for pk_name, tag in (("p_k_lin", "lin"), ("p_k_nl", "nl")):
        p = d[pk_name]
        for iz in IZ_SLICES:
            out = per_z(k_h, p[iz])
            for key, val in out.items():
                results[f"{tag}_iz{iz}_{key}"] = val
            print(f"[{tag}] iz={iz} z={z[iz]:.3f}: "
                  f"xi(r=1)={np.interp(0.0, np.log(RFIX_DENSE), out['xi_dense']):.4f} "
                  f"Sigma(R=3)={np.interp(np.log(3), np.log(R_SIGMA), out['sigma']):.4f} "
                  f"DSanchor(R=3)={np.interp(np.log(3), np.log(R_SIGMA), out['dsigma_anchor']):.4f}")
        # z-variation curve: xi(r=1, z) and Sigma(R=3, z) over the full z grid
        xi_r1 = np.empty(z.size)
        sig_r3 = np.empty(z.size)
        for i in range(z.size):
            xi_i = ct.xi.xi_mm_at_r(RFIX_DENSE, k_h, p[i])
            xi_r1[i] = np.interp(0.0, np.log(RFIX_DENSE), xi_i)
            sig_i = ct.deltasigma.Sigma_at_R(np.array([3.0]), RFIX_DENSE, xi_i,
                                             MD, CD, OMEGA_M)
            sig_r3[i] = sig_i[0] if np.ndim(sig_i) else float(sig_i)
        results[f"{tag}_xi_r1_of_z"] = xi_r1
        results[f"{tag}_sigma_r3_of_z"] = sig_r3
        print(f"[{tag}] xi(r=1): z=0 -> {xi_r1[0]:.4f}, z=4 -> {xi_r1[-1]:.4f} "
              f"(ratio {xi_r1[0]/xi_r1[-1]:.2f})")

    results["units_r"] = "Mpc/h comoving"
    results["units_sigma"] = "Msun h/pc^2 comoving, b=1"
    results["rho_convention"] = f"rho_m = Omega_m*RHO_C = {RHO_M:.6e} Msun h^2/Mpc^3"
    out_path = os.path.join(HERE, "outputs", "ref_ct.npz")
    np.savez(out_path, **results)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
