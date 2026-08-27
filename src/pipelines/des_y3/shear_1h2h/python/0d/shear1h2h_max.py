"""Traditional 1h+2h shear via the max model — Python (fast path).

The traditional-pipeline counterpart of the projection treatment (plan
owner, 2026-08-12: "two pipelines — one with traditional shear and one
with prj"). The radial operator is the Y1-era SIG_MAX/GAMMA_MAX
composition on the modern haloModel tables:

    DSigma_max(R, lnM, z | bin) = max( DSigma_cl(R, lnM | bin),
                                       bias(lnM, z) * dSigma_hh(R, z) )

with DSigma_cl the production miscentred 1-halo mixture (optionally
pure centred via include_miscentering = F) and the biased two-halo term
from haloModel (requires halo_model to run with
compute_lensing_2h = T). The observable is the count-weighted stack

    O_ij(R) = int dz int dlnM  n dV/dOmegadz Omega Sigma_crit^-1
              S_ij(lnM, z) DSigma_max(R, lnM, z)

**The two-halo term is z-dependent, so — unlike the 1h-only fixed-GL —
the redshift integral cannot be contracted past the profile.** The fast
path therefore keeps the z-resolved tabulated weight
W2d(lnM, z) = zfac * hmf * S_stack (the S_ij tabulation is still the
S_ij-tabulation hallmark) and performs the double fixed-GL contraction
sum_kq W2d * DSigma_max per (bin, R).

DataBlock contract
------------------
Reads (options): bin_index, r_perp (cartesian, bin slow / R fast),
    lob_centers (default 25 37.5 52.5 130), zt_low/zt_high/
    lnm_low/lnm_high (required), n_lnm (96), n_z (64),
    include_miscentering (default T).
Reads (datablock): the Shear1hMisSel contract plus
    halomodel/{r_sigma, z, dSigma_hh, bias}  (compute_lensing_2h = T).
Writes: shear1h2h_max/vals  (n_bins * n_r,)   [hardcoded section]

Status: reference implementation of the traditional-shear model
(validated against its explicit-3d z-resolved reference; see README.md).
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

OUTPUT_SECTION = "shear1h2h_max"


def z_resolved_weights(source, *, n_lnm, n_z, zt_lo, zt_hi, lnm_lo, lnm_hi):
    """W2d[b, k, q] = zfac_q sci_q hmf(k,q) S_stack(b; k, q) on GL nodes.

    The z-resolved sibling of MassZWeights (which contracts z because
    the 1h profile is z-free); summing over q with the lnM weights left
    out reproduces MassZWeights.W exactly.
    """
    lnm_x, lnm_w = dm.gl_nodes(lnm_lo, lnm_hi, n_lnm)
    z_x, z_w = dm.gl_nodes(zt_lo, zt_hi, n_z)
    hmf = dm.HMF(source)
    dv = dm.DVDoDz(source)
    sel = dm.SelStack(source)
    zfac = z_w * dv(z_x) * dm.omega_z_des(z_x) * dm.SigmaCritInv(source)(z_x)
    base = hmf(lnm_x[:, None], z_x[None, :]) * zfac[None, :]
    w2d = np.empty((sel.n_bins, n_lnm, n_z))
    for b in range(sel.n_bins):
        w2d[b] = base * sel(b, lnm_x[:, None], z_x[None, :])
    return lnm_x, lnm_w, z_x, w2d


def compute_shear_max(profile, lnm_x, lnm_w, z_x, w2d, bin_index, r_perp,
                      physical=False):
    """O(R) = sum_kq lnm_w_k W2d[b,k,q] DSigma_max(b, R, lnM_k, z_q).

    physical: fold one_halo_physical_density's exact identity
    DSigma_phys(R|z) = (1+z)^2 DSigma_frozen(R(1+z)) into the 1-halo
    term. Unlike the z-contracted 0d Shear1hGl (where the (1+z)^2 must
    ride in the z-integration WEIGHT because the redshift sum happens
    before the profile is known), this model's z-axis is already
    resolved -- the 2-halo term forces it (see module docstring). So
    the 1-halo term just needs to be evaluated PER z-node with q=1+z
    instead of once at q=1; no weight restructuring needed.
    """
    r_perp = np.asarray(r_perp, dtype=float)
    n_r = r_perp.size
    n_z = z_x.size
    vals = np.empty(len(bin_index) * n_r)
    for i, b in enumerate(bin_index):
        if physical:
            DSigma_1h = np.empty((n_r, lnm_x.size, n_z))
            for iq, z in enumerate(z_x):
                qf = 1.0 + z
                DSigma_1h[:, :, iq] = profile._one(
                    b, r_perp[:, None], lnm_x[None, :], q=qf) * qf**2
        else:
            DSigma_1h = profile._one(
                b, r_perp[:, None], lnm_x[None, :])[:, :, None]
        DSigma_2h = (profile._bias(lnm_x[:, None], z_x[None, :])[None, :, :]
                    * profile._hh(r_perp[:, None], z_x[None, :])[:, None, :])
        DSigma_max = np.maximum(DSigma_1h, DSigma_2h)             # (r, k, q)
        vals[i * n_r:(i + 1) * n_r] = np.einsum(
            "rkq,kq->r", DSigma_max, w2d[b] * lnm_w[:, None])
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
    try:
        cfg["include_miscentering"] = bool(
            options.get_bool(option_section, "include_miscentering"))
    except Exception:
        cfg["include_miscentering"] = True
    return cfg


def execute(block, cfg):
    t0 = time.perf_counter()
    source = dm.DataBlockSource(block)
    physical = dm.physical_density_flag(source)
    profile = lp.MaxMixtureProfile(
        source, lob_centers=cfg["lob_centers"],
        # Required: no fallback to the fiducial defaults — a pipeline
        # missing the miscentering section must fail loudly.
        f_mis=source.scalar("miscentering", "f_mis"),
        tau_mis=source.scalar("miscentering", "tau_mis"),
        omega_m=source.scalar("cosmological_parameters", "omega_m"),
        include_miscentering=cfg["include_miscentering"])
    lnm_x, lnm_w, z_x, w2d = z_resolved_weights(
        source, n_lnm=cfg["n_lnm"], n_z=cfg["n_z"],
        zt_lo=cfg["zt_low"], zt_hi=cfg["zt_high"],
        lnm_lo=cfg["lnm_low"], lnm_hi=cfg["lnm_high"])
    block[OUTPUT_SECTION, "vals"] = compute_shear_max(
        profile, lnm_x, lnm_w, z_x, w2d, cfg["bin_index"], cfg["r_perp"],
        physical=physical)
    dt_ms = 1000.0 * (time.perf_counter() - t0)
    print(f"[shear1h2h_max] {cfg['bin_index'].size} bins x "
          f"{cfg['r_perp'].size} radii (max 1h/2h) — {dt_ms:.0f} ms",
          flush=True)
    return 0


def cleanup(config):
    return 0
