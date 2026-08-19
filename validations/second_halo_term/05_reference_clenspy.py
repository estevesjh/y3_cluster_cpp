#!/usr/bin/env python
"""Fiducial-P(k) references from CLensPy, pyccl, and clmm.

Run with the CLensPy venv:
  /Users/jesteves/Documents/Dev/github/CLensPy/.venv/bin/python 05_reference_clenspy.py

- CLensPy TwoHaloTerm fed the SAME CAMB P(k,z) as everything else
  (k_phys = k_h*h, P_phys = p_k/h^3), evaluated at the comparison slices.
  Unit recipe proven in test/halo_model.test.py:415-455:
    Sigma [Msun h/pc^2 comoving] = sigma(R_h/h, z) * (Om*RHO_C*h^2)/h/1e12
- pyccl xi_mm via ccl.correlation_3d at matched cosmology (independent
  Boltzmann path -- NOT fed our npz; cross-checks the cosmology ledger).
- clmm 2-halo excess surface density (ccl backend) with the
  comoving/physical bridge derived here:
    Sigma_com = Sigma_phys / (1+z)^2 at R_phys = R_com/(1+z)
  (their own test used a single power of (1+z); the 4-way comparison
  arbitrates -- both conversions are stored.)

Everything CLensPy-side uses TwoHaloTerm directly (LensingProfile has known
rho_m and 1e12 unit bugs -- see the report's CLensPy issue list).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "common"))

from clenspy.halo.twohalo import TwoHaloTerm  # noqa: E402

RHO_C = 2.77533742639e11
OMEGA_M = 0.311049
OMEGA_B = 0.048975
SIGMA8 = 0.8238
NS = 0.9665

R_SIGMA = np.logspace(np.log10(0.1), np.log10(20.0), 128)   # Mpc/h comoving
R_XI = np.logspace(-1, np.log10(120.0), 96)                 # Mpc/h comoving
R_HUNIT_CLUSTER = np.array([3.0, 4.38, 6.41, 9.36, 13.69, 20.0])
IZ_SLICES = [0, 3, 5, 8]


def clenspy_block(k_h, p_k, z, h, tag, results):
    to_hunit = (OMEGA_M * RHO_C * h ** 2) / h / 1e12
    two = TwoHaloTerm(k_h * h, p_k / h ** 3, zvec=z)
    for iz in IZ_SLICES:
        zi = float(z[iz])
        xi = np.asarray(two.xi(R_XI / h, zi)).reshape(-1)
        sig = np.asarray(two.sigma(R_SIGMA / h, zi)).reshape(-1) * to_hunit
        ds = np.asarray(two.deltasigma(R_SIGMA / h, zi)).reshape(-1) * to_hunit
        results[f"{tag}_iz{iz}_xi"] = xi
        results[f"{tag}_iz{iz}_sigma"] = sig
        results[f"{tag}_iz{iz}_dsigma"] = ds
        print(f"[clenspy {tag}] iz={iz} z={zi:.3f}: "
              f"xi(r=1)={np.interp(0.0, np.log(R_XI), xi):.4f} "
              f"Sigma(R=3)={np.interp(np.log(3), np.log(R_SIGMA), sig):.4f} "
              f"DS(R=3)={np.interp(np.log(3), np.log(R_SIGMA), ds):.4f}")
    # benchmark constants at the pinned cluster radii, iz=5
    zi = float(z[5])
    sig_pin = np.asarray(two.sigma(R_HUNIT_CLUSTER / h, zi)).reshape(-1) * to_hunit
    ds_pin = np.asarray(two.deltasigma(R_HUNIT_CLUSTER / h, zi)).reshape(-1) * to_hunit
    xi_pin = np.asarray(two.xi(R_HUNIT_CLUSTER / h, zi)).reshape(-1)
    results[f"{tag}_pin_sigma_z_cluster"] = sig_pin
    results[f"{tag}_pin_dsigma_z_cluster"] = ds_pin
    results[f"{tag}_pin_xi_z_cluster"] = xi_pin
    print(f"[clenspy {tag}] pins at z={zi:.3f}, R={R_HUNIT_CLUSTER.tolist()}:")
    print(f"   SIGMA_BENCHMARK  = {np.array2string(sig_pin, precision=6)}")
    print(f"   DSIGMA_BENCHMARK = {np.array2string(ds_pin, precision=6)}")


def pyccl_block(z, h, results):
    import pyccl as ccl

    for tag, mps in (("lin", "linear"), ("nl", "halofit")):
        cosmo = ccl.Cosmology(Omega_c=OMEGA_M - OMEGA_B, Omega_b=OMEGA_B,
                              h=h, sigma8=SIGMA8, n_s=NS,
                              transfer_function="boltzmann_camb",
                              matter_power_spectrum=mps)
        for iz in IZ_SLICES:
            a = 1.0 / (1.0 + float(z[iz]))
            xi = ccl.correlation_3d(cosmo, r=R_XI / h, a=a)
            results[f"ccl_{tag}_iz{iz}_xi"] = np.asarray(xi)
        print(f"[pyccl {tag}] xi(r=1) at slices: "
              + ", ".join(f"z={z[iz]:.2f}: "
                          f"{np.interp(0.0, np.log(R_XI), results[f'ccl_{tag}_iz{iz}_xi']):.4f}"
                          for iz in IZ_SLICES))


def clmm_block(z, h, results):
    try:
        import clmm
    except ImportError:
        print("[clmm] not importable; skipped")
        return
    cosmo = clmm.Cosmology(H0=h * 100.0, Omega_dm0=OMEGA_M - OMEGA_B,
                           Omega_b0=OMEGA_B, Omega_k0=0.0)
    from clmm.theory import compute_excess_surface_density_2h
    for iz in IZ_SLICES[1:]:  # z=0 degenerate for lensing bridges
        zi = float(z[iz])
        r_phys = (R_SIGMA / h) / (1.0 + zi)      # Mpc physical
        ds_phys = compute_excess_surface_density_2h(r_phys, zi, cosmo,
                                                    halobias=1.0)
        ds_phys = np.asarray(ds_phys).reshape(-1)
        # comoving conversions, both candidate powers of (1+z); to h-units:
        # Msun/Mpc^2 (physical) -> Msun h/pc^2 comoving: /(1+z)^p /1e12 /h
        results[f"clmm_iz{iz}_dsigma_p2"] = ds_phys / (1.0 + zi) ** 2 / 1e12 / h
        results[f"clmm_iz{iz}_dsigma_p1"] = ds_phys / (1.0 + zi) ** 1 / 1e12 / h
        print(f"[clmm] iz={iz} z={zi:.3f}: DS_com(R=3, (1+z)^-2) = "
              f"{np.interp(np.log(3), np.log(R_SIGMA), results[f'clmm_iz{iz}_dsigma_p2']):.4f}"
              f"  ((1+z)^-1: "
              f"{np.interp(np.log(3), np.log(R_SIGMA), results[f'clmm_iz{iz}_dsigma_p1']):.4f})")


def main():
    d = np.load(os.path.join(HERE, "outputs", "pk_camb.npz"), allow_pickle=False)
    k_h, z, h = d["k_h"], d["z"], float(d["h"])

    results = {"r_sigma": R_SIGMA, "r_xi": R_XI, "z": z,
               "iz_slices": np.array(IZ_SLICES),
               "r_pin": R_HUNIT_CLUSTER}
    clenspy_block(k_h, d["p_k_lin"], z, h, "lin", results)
    clenspy_block(k_h, d["p_k_nl"], z, h, "nl", results)
    pyccl_block(z, h, results)
    clmm_block(z, h, results)

    results["units_r"] = "Mpc/h comoving"
    results["units_sigma"] = "Msun h/pc^2 comoving, b=1"
    results["rho_convention"] = ("Sigma = rho_m,com * dimensionless; "
                                 "rho_m = Omega_m*RHO_C*h^2 Msun/Mpc^3 physical-units path")
    out_path = os.path.join(HERE, "outputs", "ref_clenspy.npz")
    np.savez(out_path, **results)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
