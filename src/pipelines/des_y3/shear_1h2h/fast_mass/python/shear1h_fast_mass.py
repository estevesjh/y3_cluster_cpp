"""Miscentred one-halo shear via the fast_mass strategy — Python.

The exact-redshift-contraction fast path for the shear observable,
extending the plan's `fast_mass` strategy from counts to shear at the
plan owner's direction: the z integral is done exactly, on fixed GL
nodes, OUTSIDE the radial operator —

    W_ij(lnM) = int dz  n(M,z) dV/dOmega/dz(z) Omega(z)
                        Sigma_crit_inv(z) S_ij(lnM,z)
    O_ij(R)   = int dlnM  W_ij(lnM) Phi_i(R, lnM)

with Phi_i the production miscentred mixture
(1 - f_mis) DSigma_nfw + f_mis DSigma_mis. This is exactly what the
production Shear1hMisSel.so `method = exact` computes
(Shear1hMisSelGL); here it is expressed through the shared Python
replicas — MassZWeights for W_ij (SelGLCore twin) and
lensing_profiles.MisMixtureProfile for Phi (haloModel bilinear +
NFW_DSIGMA_MIS gamma table, interpolation-exact) — so the backend
agrees with production to near machine precision (see README.md).

Unlike radial_series (whose per-(bin,R) work is three table lookups),
this pays the full n_lnm-node mass sum per (bin, R) point; unlike
full_ltmz, it consumes the tabulated S_ij rather than re-evaluating the
selection kernels.

DataBlock contract
------------------
Reads (options): bin_index, r_perp (cartesian, bin slow / R fast),
    lob_centers (default 25 37.5 52.5 130),
    zt_low, zt_high, lnm_low, lnm_high (required), n_lnm (96), n_z (64).
Reads (datablock): the full Shear1hMisSel contract —
    sel_function/{lnM,z,S_stack}, mass_function/*, cluster_abundance/*,
    distances/{z,d_a}, average_sigma_crit_inv/{zlense,sci_average},
    halomodel/{r_sigma,lnM,dSigma_nfw}, miscentering/{f_mis,tau_mis}
    (optional; defaults 0.22/0.17), cosmological_parameters/*,
    data/nfw_off_center/*gamma* (fixed tables, loaded at import).
Writes: shear1h_fast_mass/vals  (n_bins * n_r,)   [hardcoded section]

Status: reference re-expression of the production algorithm.
Production remains Shear1hMisSel.so.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

for _p in Path(__file__).resolve().parents:
    if (_p / "shared" / "datablock_models.py").is_file():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

from shared import datablock_models as dm
from shared import lensing_profiles as lp
from systematics.selection_richness.python import sel_kernels

OUTPUT_SECTION = "shear1h_fast_mass"


def compute_shear(weights, profile, bin_index, r_perp):
    """O(R) for the requested bins: sum_k w_k W[b,k] Phi_b(R, lnM_k)."""
    n_r = len(r_perp)
    vals = np.empty(len(bin_index) * n_r)
    lnm = weights.lnm_x
    for i, b in enumerate(bin_index):
        phi = profile(b, np.asarray(r_perp)[:, None], lnm[None, :])
        vals[i * n_r:(i + 1) * n_r] = phi @ (weights.lnm_w * weights.W[b])
    return vals


def setup(options):
    from cosmosis.datablock import option_section
    sf = sel_kernels.load()
    cfg = {}
    cfg["bin_index"] = sf._read_array(options, option_section,
                                      "bin_index").astype(int)
    cfg["r_perp"] = sf._read_array(options, option_section, "r_perp")
    try:
        cfg["lob_centers"] = sf._read_array(options, option_section,
                                            "lob_centers")
    except Exception:
        cfg["lob_centers"] = np.asarray(dm.DEFAULT_LOB_CENTERS)
    for key in ("zt_low", "zt_high", "lnm_low", "lnm_high"):
        cfg[key] = float(options.get_double(option_section, key))
    for key, default in (("n_lnm", 96), ("n_z", 64)):
        try:
            cfg[key] = int(options.get_int(option_section, key))
        except Exception:
            cfg[key] = default
    return cfg


def execute(block, cfg):
    t0 = time.perf_counter()
    source = dm.DataBlockSource(block)
    weights = dm.MassZWeights(
        source, n_lnm=cfg["n_lnm"], n_z=cfg["n_z"],
        zt_lo=cfg["zt_low"], zt_hi=cfg["zt_high"],
        lnm_lo=cfg["lnm_low"], lnm_hi=cfg["lnm_high"], include_sci=True)
    profile = lp.MisMixtureProfile(
        source, lob_centers=cfg["lob_centers"],
        f_mis=dm.read_mis_param(source, "f_mis", dm.F_MIS_DEFAULT),
        tau_mis=dm.read_mis_param(source, "tau_mis", dm.TAU_MIS_DEFAULT),
        omega_m=source.scalar("cosmological_parameters", "omega_m"))

    block[OUTPUT_SECTION, "vals"] = compute_shear(
        weights, profile, cfg["bin_index"], cfg["r_perp"])
    dt_ms = 1000.0 * (time.perf_counter() - t0)
    print(f"[shear1h_fast_mass] {cfg['bin_index'].size} bins x "
          f"{cfg['r_perp'].size} radii — {dt_ms:.0f} ms", flush=True)
    return 0


def cleanup(config):
    return 0
