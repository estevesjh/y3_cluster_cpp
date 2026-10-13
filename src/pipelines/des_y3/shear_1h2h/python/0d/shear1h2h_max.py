"""Traditional 1h+2h excess surface density via ``Shear1h2hMax``.

The traditional-pipeline counterpart of the projection treatment (plan
owner, 2026-08-12: "two pipelines — one with traditional shear and one
with prj"). The radial profile is formulated via 3D halo exclusion and 3-pillar
smooth bridging:

1. In 3D real space, halo exclusion is defined on the correlation function:
       xi_hm(r) = max(xi_1h(r), b(M, z) * xi_mm(r, z))
   with transition radius r_trans defined by xi_1h(r_trans) = b(M, z) * xi_mm(r_trans, z).
2. The projected excess surface density DeltaSigma_max is assembled via the 3-pillar bridging operators:
   - For R < r_trans:
       DeltaSigma_1h(R) + Delta_gap^1 * (R / r_trans)^2
   - For R >= r_trans:
       b * DeltaSigma_hh(R) + (C_0 / R^2) * 1e-12
   where C_0 = Delta M_3D(r_trans) / pi (in Msun/h) and
   Delta_gap^1 = DeltaSigma_max(r_trans) - DeltaSigma_1h(r_trans).

DataBlock contract
------------------
Reads (options): bin_index, r_perp (cartesian, bin slow / R fast),
    lob_centers (default 25 37.5 52.5 130), zt_low/zt_high/
    lnm_low/lnm_high (required), n_lnm (96), n_z (64),
    n_gl (20), n_root (96) for the opt-in max_xi path,
    n_offset (24), n_phi (64), n_aperture (8), q_max (12),
    n_residual (512) for the signed residual convolution,
    max_xi (default F; applies max operator on xi in 3D before projection),
    miscentering_method (default ``table``; ``table`` keeps the DES NFW
        gamma table plus residual correction, ``direct`` convolves the full
        centered Sigma_max and then applies the aperture operation),
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
        if hasattr(profile, "excess_surface_density_grid"):
            envelope = profile.excess_surface_density_grid(
                b, r_perp, lnm_x, z_x, physical=physical)
            vals[i * n_r:(i + 1) * n_r] = np.einsum(
                "rkq,kq->r", envelope, w2d[b] * lnm_w[:, None])
            continue
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
        if getattr(profile, "_has_max1h2h", False):
            rt = profile._r_trans(lnm_x[:, None], z_x[None, :])[None, :, :]
            c0 = profile._c0_trans(lnm_x[:, None], z_x[None, :])[None, :, :]
            gap1 = profile._delta_gap_1(lnm_x[:, None], z_x[None, :])[None, :, :]
            r_arr = r_perp[:, None, None]
            c0_term = (c0 / (r_arr * r_arr)) * 1.0e-12
            outer = DSigma_2h + c0_term
            inner = DSigma_1h + gap1 * (r_arr / rt)**2
            DSigma_max = np.where(r_arr < rt, inner, outer)
        else:
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
    for key, default in (("n_lnm", 96), ("n_z", 64),
                         ("n_gl", 20), ("n_root", 96),
                         ("n_offset", 24), ("n_phi", 64),
                         ("n_aperture", 8), ("n_residual", 512)):
        try:
            cfg[key] = int(options.get_int(option_section, key))
        except Exception:
            cfg[key] = default
    try:
        cfg["q_max"] = float(options.get_double(option_section, "q_max"))
    except Exception:
        cfg["q_max"] = 12.0
    try:
        cfg["max_xi"] = bool(
            options.get_bool(option_section, "max_xi"))
    except Exception:
        cfg["max_xi"] = False
    try:
        cfg["use_nfw_table_residual"] = bool(
            options.get_bool(option_section, "use_nfw_table_residual"))
    except Exception:
        cfg["use_nfw_table_residual"] = True
    try:
        cfg["miscentering_method"] = options.get_string(
            option_section, "miscentering_method").lower()
    except Exception:
        cfg["miscentering_method"] = "table"
    if cfg["miscentering_method"] not in ("table", "direct"):
        raise ValueError(
            "miscentering_method must be 'table' or 'direct'")
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
    from cosmology.shear1h2h_max_profile import Shear1h2hMaxProfile
    profile_type = (Shear1h2hMaxProfile
                    if cfg["max_xi"] else lp.MaxMixtureProfile)
    profile_kwargs = dict(
        source=source, lob_centers=cfg["lob_centers"],
        # Required: no fallback to the fiducial defaults — a pipeline
        # missing the miscentering section must fail loudly.
        f_mis=source.scalar("miscentering", "f_mis"),
        tau_mis=source.scalar("miscentering", "tau_mis"),
        omega_m=source.scalar("cosmological_parameters", "omega_m"),
        include_miscentering=cfg["include_miscentering"])
    if cfg["max_xi"]:
        profile_kwargs.update(
            n_gl=cfg["n_gl"], n_root=cfg["n_root"],
            n_offset=cfg["n_offset"], n_phi=cfg["n_phi"],
            n_aperture=cfg["n_aperture"], q_max=cfg["q_max"],
            n_residual=cfg["n_residual"],
            use_nfw_table_residual=cfg["use_nfw_table_residual"],
            miscentering_method=cfg["miscentering_method"])
    profile = profile_type(**profile_kwargs)
    lnm_x, lnm_w, z_x, w2d = z_resolved_weights(
        source, n_lnm=cfg["n_lnm"], n_z=cfg["n_z"],
        zt_lo=cfg["zt_low"], zt_hi=cfg["zt_high"],
        lnm_lo=cfg["lnm_low"], lnm_hi=cfg["lnm_high"])
    block[OUTPUT_SECTION, "vals"] = compute_shear_max(
        profile, lnm_x, lnm_w, z_x, w2d, cfg["bin_index"], cfg["r_perp"],
        physical=physical)
    dt_ms = 1000.0 * (time.perf_counter() - t0)
    label = "max_xi" if cfg["max_xi"] else "max 1h/2h"
    print(f"[shear1h2h_max] {cfg['bin_index'].size} bins x "
          f"{cfg['r_perp'].size} radii ({label}) — {dt_ms:.0f} ms",
          flush=True)
    return 0


def cleanup(config):
    return 0
