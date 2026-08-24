#!/usr/bin/env python
"""Analytic-chain test bench: CLensPy transforms, validated stage by stage.

Run with the CLensPy venv:
  /Users/jesteves/Documents/Dev/github/CLensPy/.venv/bin/python 03_chain_bench_clenspy.py

Stages (closed forms from common/analytic_profiles.py):
  P->xi     : raw mcfit.P2xi (the FFTLog engine) and clenspy's
              pk_to_xi_fftlog wrapper (adds the positive-masked log-log
              interpolator) must return rho(r) when fed rho_tilde(k)
  xi->Sigma : clenspy compute_sigma_grid (trapz backend; n_points 150 as
              TwoHaloTerm uses, and a 400-pt convergence variant)
  Sigma->DS : clenspy sigma_to_deltasigma_cumtrapz on
              (i) the production-truncated table [0.1, 20] only and
              (ii) an extended grid logspace(-3, log10 20) (inner-boundary
              term is dropped by the implementation -- (i) measures that)
  full chain: TwoHaloTerm(k, rho_tilde) .xi/.sigma/.deltasigma vs closed forms

CLensPy is UNDER TEST here (not a reference): known internals measured by
this bench include the negative-dropping log interpolator, the power-law
k-extrapolation onto the internal _kfine grid, and the missing
inner-boundary term in the cumtrapz DeltaSigma.

Unit ledger: unit-agnostic math; we feed lengths in Mpc/h and rho in
Msun h^2/Mpc^3 so Sigma comes out in Msun h/Mpc^2; /1e12 -> Msun h/pc^2.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "common"))

import mcfit  # noqa: E402
from clenspy.halo.twohalo import TwoHaloTerm  # noqa: E402
from clenspy.utils.integrate import (compute_sigma_grid,  # noqa: E402
                                     pk_to_xi_fftlog,
                                     sigma_to_deltasigma_cumtrapz)
from analytic_profiles import (GaussianProfile, ExponentialProfile,  # noqa: E402
                               NFWProfile)

RHO_C = 2.77533742639e11
OMEGA_M = 0.311049
RHO_M = OMEGA_M * RHO_C

R_SIGMA = np.logspace(np.log10(0.1), np.log10(20.0), 128)
K_GRID = np.logspace(-4, 3, 1200)


def nfw_from_mass(m200m=1e14, c=5.0):
    r200 = (3.0 * m200m / (4.0 * np.pi * 200.0 * RHO_M)) ** (1.0 / 3.0)
    r_s = r200 / c
    delta_c = (200.0 / 3.0) * c ** 3 / (np.log(1.0 + c) - c / (1.0 + c))
    return NFWProfile(rho_s=delta_c * RHO_M, r_s=r_s, c=c)


def frac_err(got, want, mask=None):
    got, want = np.asarray(got), np.asarray(want)
    if mask is None:
        mask = np.ones_like(want, dtype=bool)
    e = np.abs(got[mask] / want[mask] - 1.0)
    return float(np.nanmax(e)), float(np.nanmedian(e))


def support_mask(p, r):
    if isinstance(p, GaussianProfile):
        return r < 3.5 * p.s
    if isinstance(p, ExponentialProfile):
        return r < 8.0 * p.h
    return (r > 0.02) & (r < 30.0)  # NFW untruncated FT window


def main():
    results = {}
    profiles = {
        "gaussian_s05": GaussianProfile(rho0=5.0 * RHO_M, s=0.5),
        "gaussian_s1": GaussianProfile(rho0=5.0 * RHO_M, s=1.0),
        "gaussian_s2": GaussianProfile(rho0=5.0 * RHO_M, s=2.0),
        "exponential_h1": ExponentialProfile(rho0=5.0 * RHO_M, h=1.0),
        "nfw_m14_c5": nfw_from_mass(1e14, 5.0),
    }

    r_eval = np.logspace(-2, 1.8, 80)

    print("=== stage P->xi (raw mcfit.P2xi and clenspy pk_to_xi_fftlog) ===")
    for name, p in profiles.items():
        p_in = p.rho_tilde(K_GRID)
        # raw mcfit: native output grid, no interpolation layer
        r_fft, xi_fft = mcfit.P2xi(K_GRID, lowring=True)(p_in)
        m_fft = support_mask(p, r_fft) & (r_fft > r_eval[0])
        mx_r, md_r = frac_err(xi_fft, p.rho(r_fft), m_fft)
        # clenspy wrapper (positive-masked log-log interpolation onto r_eval)
        xi_wrap = pk_to_xi_fftlog(K_GRID, p_in, r_eval)
        m_ev = support_mask(p, r_eval)
        mx_w, md_w = frac_err(xi_wrap, p.rho(r_eval), m_ev)
        results[f"{name}__p2xi_raw_err"] = np.array([mx_r, md_r])
        results[f"{name}__p2xi_wrap_err"] = np.array([mx_w, md_w])
        results[f"{name}__p2xi_wrap_out"] = xi_wrap
        results[f"{name}__p2xi_true"] = p.rho(r_eval)
        print(f"  {name:18s} raw mcfit: max={mx_r:.3e} med={md_r:.3e} | "
              f"wrapper: max={mx_w:.3e} med={md_w:.3e}")

    print("=== stage xi->Sigma (clenspy compute_sigma_grid, trapz) ===")
    for name, p in profiles.items():
        sig_true = p.sigma(R_SIGMA) / 1e12
        mask = sig_true > 1e-12 * np.nanmax(sig_true)
        for n_pts, tag in ((150, "n150"), (400, "n400")):
            sig = compute_sigma_grid(lambda r, z: p.rho(r), R_SIGMA,
                                     np.array([0.0]), method="trapz",
                                     rmax_integral=300.0, n_points=n_pts)
            sig = np.asarray(sig).reshape(-1) / 1e12
            mx, md = frac_err(sig, sig_true, mask)
            results[f"{name}__x2s_{tag}_err"] = np.array([mx, md])
            if tag == "n150":
                results[f"{name}__x2s_out"] = sig
                results[f"{name}__x2s_true"] = sig_true
            print(f"  {name:18s} {tag}: max={mx:.3e} med={md:.3e}")

    print("=== stage Sigma->DeltaSigma (clenspy cumtrapz) ===")
    for name, p in profiles.items():
        ds_true = p.delta_sigma(R_SIGMA) / 1e12
        ds_scale = np.nanmax(np.abs(ds_true))
        mask_r05 = R_SIGMA >= 0.5
        # (i) production-truncated table only
        ds_tab = sigma_to_deltasigma_cumtrapz(R_SIGMA, p.sigma(R_SIGMA) / 1e12)
        # (ii) extended grid
        r_ext = np.logspace(-3, np.log10(20.0), 300)
        ds_ext_grid = sigma_to_deltasigma_cumtrapz(r_ext, p.sigma(r_ext) / 1e12)
        ds_ext = np.interp(np.log(R_SIGMA), np.log(r_ext), np.asarray(ds_ext_grid).reshape(-1))
        for tag, ds in (("tableonly", np.asarray(ds_tab).reshape(-1)),
                        ("extended", ds_ext)):
            mx, md = frac_err(ds, ds_true, mask_r05)
            lo = float(np.nanmax(np.abs((ds - ds_true)[~mask_r05])) / ds_scale)
            results[f"{name}__s2d_{tag}_err"] = np.array([mx, md, lo])
            results[f"{name}__s2d_{tag}_out"] = ds
            print(f"  {name:18s} {tag:9s}: R>=0.5 max={mx:.3e} med={md:.3e} | "
                  f"R<0.5 abs/peak: {lo:.3e}")
        results[f"{name}__s2d_true"] = ds_true

    print("=== full chain (TwoHaloTerm: rho_tilde -> xi -> Sigma -> DS) ===")
    for name, p in profiles.items():
        two = TwoHaloTerm(K_GRID, p.rho_tilde(K_GRID))
        z0 = 0.0
        xi_c = np.asarray(two.xi(r_eval, z0)).reshape(-1)
        m_ev = support_mask(p, r_eval)
        mx_xi, md_xi = frac_err(xi_c, p.rho(r_eval), m_ev)
        sig_c = np.asarray(two.sigma(R_SIGMA, z0)).reshape(-1) / 1e12
        sig_true = p.sigma(R_SIGMA) / 1e12
        # tighter mask than the stage test: the full chain's xi-extrapolation
        # garbage beyond the profile support is a *finding* (reported via the
        # stored arrays), not something to fold into the in-support metric
        m_sig = sig_true > 1e-6 * np.nanmax(sig_true)
        mx_s, md_s = frac_err(sig_c, sig_true, m_sig)
        ds_c = np.asarray(two.deltasigma(R_SIGMA, z0)).reshape(-1) / 1e12
        ds_true = p.delta_sigma(R_SIGMA) / 1e12
        mx_d, md_d = frac_err(ds_c, ds_true, R_SIGMA >= 0.5)
        results[f"{name}__chain_err"] = np.array([mx_xi, md_xi, mx_s, md_s, mx_d, md_d])
        results[f"{name}__chain_xi"] = xi_c
        results[f"{name}__chain_sigma"] = sig_c
        results[f"{name}__chain_ds"] = ds_c
        print(f"  {name:18s} xi: max={mx_xi:.3e} med={md_xi:.3e} | "
              f"Sigma: max={mx_s:.3e} med={md_s:.3e} | "
              f"DS(R>=0.5): max={mx_d:.3e} med={md_d:.3e}")

    results["r_sigma"] = R_SIGMA
    results["r_eval"] = r_eval
    results["units_r"] = "Mpc/h comoving"
    results["units_sigma"] = "Msun h/pc^2 comoving"
    results["rho_convention"] = f"rho_m = Omega_m*RHO_C = {RHO_M:.6e} Msun h^2/Mpc^3"
    out_path = os.path.join(HERE, "outputs", "chain_clenspy.npz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **results)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
