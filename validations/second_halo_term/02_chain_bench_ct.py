#!/usr/bin/env python
"""Analytic-chain test bench: cluster_toolkit transforms + candidate
DeltaSigma-from-Sigma methods, validated stage by stage against closed forms.

Run with the conda pipeline env:
  /opt/homebrew/Caskroom/miniforge/base/envs/y3cl_je_macos/bin/python 02_chain_bench_ct.py

Stages (each has an independent closed-form truth from
common/analytic_profiles.py):
  P->xi     : ct.xi.xi_mm_at_r fed rho_tilde(k) must return rho(r)
  xi->Sigma : ct.deltasigma.Sigma_at_R fed rho/rho_m must return Sigma/1e12
  Sigma->DS : candidate methods fed Sigma tabulated ONLY on the production
              r_sigma grid [0.1, 20] cMpc/h (128 pts) must return DeltaSigma
     (a)  repaired NFW sandwich (consistent Md)          [production candidate]
     (a-) sandwich with Md/10 in DeltaSigma_at_R         [current bug, replicated]
     (b1) direct cumtrapz, table-only + power-law inner extrapolation
     (b2) direct cumtrapz on extended exact Sigma (mimics production, which
          can evaluate Sigma below r_sigma_min from xi)  [production candidate]
     (c)  bare ct.DeltaSigma_at_R, no sandwich           [what the sandwich buys]

Unit ledger: lengths in Mpc/h comoving; rho in Msun h^2 / Mpc^3 comoving;
Sigma/DeltaSigma in Msun h / pc^2 comoving (i.e. closed forms / 1e12);
rho_m = Omega_m * RHO_C. The transforms are unit-agnostic; the ledger is
enforced when comparing (everything is converted to Msun h/pc^2 first).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "common"))
sys.path.insert(0, os.path.join(REPO, "y3_buzzard"))

import cluster_toolkit as ct  # noqa: E402
from analytic_profiles import (GaussianProfile, ExponentialProfile,  # noqa: E402
                               NFWProfile)
from nfwModel import sigmaNFW_Analytical, deltaSigmaNFW_Analytical  # noqa: E402

RHO_C = 2.77533742639e11  # Msun/Mpc^3/h^2
OMEGA_M = 0.311049
RHO_M = OMEGA_M * RHO_C   # Msun h^2 / Mpc^3 comoving

R_SIGMA = np.logspace(np.log10(0.1), np.log10(20.0), 128)  # production grid
MD, CD = 1e14, 5.0        # production dummy-halo values
K_GRID = np.logspace(-4, 3, 1200)   # h/Mpc


def nfw_from_mass(m200m=1e14, c=5.0):
    """NFWProfile in ct-native units for a 200-mean-density halo."""
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


# --------------------------------------------------------------------------
# stage 1: P -> xi
# --------------------------------------------------------------------------
def stage_p_to_xi(profile, r_eval, mask, truncated=False):
    if truncated:
        p_in = profile.rho_tilde(K_GRID, truncated=True)
        rho_true = profile.rho(r_eval, truncated=True)
    else:
        p_in = profile.rho_tilde(K_GRID)
        rho_true = profile.rho(r_eval)
    # floor: a Gaussian FT underflows to exactly 0 for k >~ 60/s, and
    # cluster_toolkit's internal log-spline of P NaNs on log(0)
    p_in = np.maximum(p_in, 1e-140 * p_in.max())
    xi_out = ct.xi.xi_mm_at_r(r_eval, K_GRID, p_in)
    n_bad = int(np.sum(~np.isfinite(xi_out)))
    mx, md = frac_err(xi_out, rho_true, mask)
    return xi_out, rho_true, mx, md, n_bad


# --------------------------------------------------------------------------
# stage 2: xi -> Sigma
# --------------------------------------------------------------------------
def stage_xi_to_sigma(profile):
    r_xi = np.logspace(-3, 3, 1000)
    # floor: exp-profile tails underflow to exactly 0, which NaNs
    # cluster_toolkit's internal log-space handling
    xi_in = np.maximum(profile.rho(r_xi) / RHO_M, 1e-140)
    sig_out = ct.deltasigma.Sigma_at_R(R_SIGMA, r_xi, xi_in, MD, CD, OMEGA_M)
    sig_true = profile.sigma(R_SIGMA) / 1e12     # Msun h/pc^2
    # fractional error only where Sigma_true has not underflowed to the
    # floor (Gaussian tails vanish within the R range)
    mask = sig_true > 1e-12 * np.nanmax(sig_true)
    mx, md = frac_err(sig_out, sig_true, mask)
    return sig_out, sig_true, mx, md


# --------------------------------------------------------------------------
# stage 3: Sigma -> DeltaSigma (methods)
# --------------------------------------------------------------------------
def dsig_sandwich(sig_table, md=MD, cd=CD, md_ds=None):
    """Add analytic Sigma_NFW(md), ct DeltaSigma_at_R, subtract analytic
    DeltaSigma_NFW(md). A consistent sandwich uses md everywhere; md_ds
    overrides only the mass passed to DeltaSigma_at_R (the current
    production bug passes md/10 there)."""
    if md_ds is None:
        md_ds = md
    sig_nfw = sigmaNFW_Analytical(R_SIGMA, md, cd, rho_c=RHO_M) / 1e12
    dsig_nfw = deltaSigmaNFW_Analytical(R_SIGMA, md, cd, rho_c=RHO_M) / 1e12
    ds = ct.deltasigma.DeltaSigma_at_R(R_SIGMA, R_SIGMA, sig_table + sig_nfw,
                                       md_ds, cd, OMEGA_M)
    return ds - dsig_nfw


def dsig_direct(r_grid, sig_grid, r_out):
    """Sigma_bar(<R) by cumulative trapezoid of Sigma R dR (log-spaced grid),
    DeltaSigma = Sigma_bar - Sigma, interpolated onto r_out."""
    lnr = np.log(r_grid)
    integrand = sig_grid * r_grid ** 2           # Sigma R dR = Sigma R^2 dlnR
    cum = np.concatenate([[0.0], np.cumsum(
        0.5 * (integrand[1:] + integrand[:-1]) * np.diff(lnr))])
    # inner disc [0, r_grid[0]]: Sigma ~ flat at the first point
    cum += 0.5 * sig_grid[0] * r_grid[0] ** 2
    sig_bar = 2.0 * cum / r_grid ** 2
    ds = sig_bar - sig_grid
    return np.interp(np.log(r_out), lnr, ds)


def dsig_direct_table_only(sig_table):
    """Method (b1): only the truncated table; extend Sigma below r_min by a
    power law fit to the innermost decade."""
    n_fit = np.searchsorted(R_SIGMA, R_SIGMA[0] * 10.0)
    slope, lnA = np.polyfit(np.log(R_SIGMA[:n_fit]), np.log(sig_table[:n_fit]), 1)
    r_in = np.logspace(-3, np.log10(R_SIGMA[0]), 60, endpoint=False)
    sig_in = np.exp(lnA) * r_in ** slope
    r_full = np.concatenate([r_in, R_SIGMA])
    sig_full = np.concatenate([sig_in, sig_table])
    return dsig_direct(r_full, sig_full, R_SIGMA)


def dsig_direct_extended(profile):
    """Method (b2): exact Sigma on an extended grid (production analogue:
    Sigma is computable below r_sigma_min from xi)."""
    r_ext = np.logspace(-3, np.log10(20.0), 300)
    sig_ext = profile.sigma(r_ext) / 1e12
    return dsig_direct(r_ext, sig_ext, R_SIGMA)


def stage_sigma_to_dsigma(profile):
    sig_table = profile.sigma(R_SIGMA) / 1e12
    ds_true = profile.delta_sigma(R_SIGMA) / 1e12
    out = {"ds_true": ds_true, "sig_table": sig_table}
    methods = {
        "sandwich": lambda: dsig_sandwich(sig_table),
        "sandwich_md10": lambda: dsig_sandwich(sig_table, md_ds=MD / 10.0),
        "sandwich_md13": lambda: dsig_sandwich(sig_table, md=1e13, cd=4.0),
        "direct_tableonly": lambda: dsig_direct_table_only(sig_table),
        "direct_extended": lambda: dsig_direct_extended(profile),
        "bare_ct": lambda: ct.deltasigma.DeltaSigma_at_R(
            R_SIGMA, R_SIGMA, sig_table, MD, CD, OMEGA_M),
    }
    # fractional error where the truth is not vanishing (R >= 0.5, per the
    # decision criterion); absolute error, scaled to the profile's DS peak,
    # below that (DS -> 0 there, fractional is ill-posed)
    mask_r05 = R_SIGMA >= 0.5
    ds_scale = np.nanmax(np.abs(ds_true))
    for name, fn in methods.items():
        ds = fn()
        out[f"ds_{name}"] = ds
        out[f"err_{name}_r05"] = frac_err(ds, ds_true, mask_r05)
        out[f"abserr_{name}_lo"] = float(
            np.nanmax(np.abs((ds - ds_true)[~mask_r05])) / ds_scale)
    out["ds_scale"] = np.float64(ds_scale)
    return out


def main():
    results = {}
    profiles = {
        "gaussian_s05": (GaussianProfile(rho0=5.0 * RHO_M, s=0.5), None),
        "gaussian_s1": (GaussianProfile(rho0=5.0 * RHO_M, s=1.0), None),
        "gaussian_s2": (GaussianProfile(rho0=5.0 * RHO_M, s=2.0), None),
        "exponential_h1": (ExponentialProfile(rho0=5.0 * RHO_M, h=1.0), None),
        "nfw_m14_c5": (nfw_from_mass(1e14, 5.0), None),
    }

    r_eval = np.logspace(-2, 1.8, 80)
    print("=== stage P->xi (ct.xi.xi_mm_at_r) ===")
    # Masks: cluster_toolkit's fixed-cycle Hankel quadrature is accurate for
    # power-law-tailed P (the cosmological case; NFW below) but genuinely
    # degrades at r << s for flat-cored profile FTs (85% at r = 0.01 s,
    # k_max-independent -- mcfit's FFTLog handles the same input at 1e-8;
    # see 03). Flat-core masks therefore start at 0.3 of the core scale;
    # the full unmasked error profile is stored for the report figure.
    for name, (p, _) in profiles.items():
        if isinstance(p, GaussianProfile):
            mask = (r_eval > 0.3 * p.s) & (r_eval < 3.5 * p.s)
        elif isinstance(p, ExponentialProfile):
            mask = (r_eval > 0.3 * p.h) & (r_eval < 8.0 * p.h)
        else:
            mask = r_eval < 0.9 * p.c * p.r_s
        trunc = isinstance(p, NFWProfile)
        xi_out, rho_true, mx, md, n_bad = stage_p_to_xi(p, r_eval, mask,
                                                        truncated=trunc)
        results[f"{name}__p2xi_out"] = xi_out
        results[f"{name}__p2xi_true"] = rho_true
        results[f"{name}__p2xi_mask"] = mask
        results[f"{name}__p2xi_err"] = np.array([mx, md])
        results[f"{name}__p2xi_nan"] = np.int64(n_bad)
        print(f"  {name:18s} max={mx:.3e} median={md:.3e} nonfinite={n_bad}"
              f"{'  (truncated FT, r < 0.9 c r_s)' if trunc else ''}")
        if trunc:
            # untruncated variant: smooth FT on the finite k window (this is
            # the shape actually resembling a real P(k): no truncation ringing)
            mask_u = (r_eval > 0.02) & (r_eval < 30.0)
            xi_u, rho_u, mxu, mdu, n_bad_u = stage_p_to_xi(p, r_eval, mask_u,
                                                           truncated=False)
            results[f"{name}__p2xi_untrunc_out"] = xi_u
            results[f"{name}__p2xi_untrunc_true"] = rho_u
            results[f"{name}__p2xi_untrunc_err"] = np.array([mxu, mdu])
            results[f"{name}__p2xi_untrunc_nan"] = np.int64(n_bad_u)
            print(f"  {name:18s} max={mxu:.3e} median={mdu:.3e} "
                  f"nonfinite={n_bad_u}  (untruncated FT)")

    print("=== stage xi->Sigma (ct.deltasigma.Sigma_at_R) ===")
    for name, (p, _) in profiles.items():
        sig_out, sig_true, mx, md = stage_xi_to_sigma(p)
        results[f"{name}__x2s_out"] = sig_out
        results[f"{name}__x2s_true"] = sig_true
        results[f"{name}__x2s_err"] = np.array([mx, md])
        print(f"  {name:18s} max={mx:.3e} median={md:.3e}")

    print("=== stage Sigma->DeltaSigma (methods, table = r_sigma grid only) ===")
    for name, (p, _) in profiles.items():
        out = stage_sigma_to_dsigma(p)
        for k, v in out.items():
            results[f"{name}__s2d_{k}"] = np.asarray(v)
        print(f"  {name}:  (DS peak scale = {out['ds_scale']:.3e} Msun h/pc^2)")
        for m in ("sandwich", "sandwich_md10", "direct_tableonly",
                  "direct_extended", "bare_ct"):
            r5, r5med = out[f"err_{m}_r05"]
            lo = out[f"abserr_{m}_lo"]
            print(f"    {m:18s} R>=0.5 frac: max={r5:.3e} med={r5med:.3e} | "
                  f"R<0.5 abs/peak: {lo:.3e}")
        # cancellation residual: two consistent dummy-halo choices must give
        # the same answer; normalize the difference by the DS peak scale
        diff = np.abs(out["ds_sandwich"] - out["ds_sandwich_md13"])
        canc = float(np.nanmax(diff[R_SIGMA >= 0.5]) / out["ds_scale"])
        results[f"{name}__s2d_cancellation"] = np.float64(canc)
        print(f"    cancellation residual |DS(1e14,c5)-DS(1e13,c4)|/peak, R>=0.5: {canc:.3e}")

    results["r_sigma"] = R_SIGMA
    results["r_eval"] = r_eval
    results["units_r"] = "Mpc/h comoving"
    results["units_sigma"] = "Msun h/pc^2 comoving"
    results["rho_convention"] = f"rho_m = Omega_m*RHO_C = {RHO_M:.6e} Msun h^2/Mpc^3"
    out_path = os.path.join(HERE, "outputs", "chain_ct.npz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **results)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
