"""Traditional 1h+2h max-model shear — full (lambda_true, lnM, z) reference.

The `explicit-3d` reference for the traditional-shear observable: the
explicit composition

    O_ij(R) = int dz int dlnM int dlt  n(M,z) dV/dOmega/dz(z) Omega(z)
              Sigma_crit_inv(z) S_j(z) S_i(lt, z) P_HOD(lt | M, z)
              DSigma_max(R, lnM, z | bin)

    DSigma_max = max( DSigma_cl(R, lnM | bin),
                 bias(lnM, z) * dSigma_hh(R, z) )

with every selection kernel evaluated at the quadrature nodes (no S_ij
tabulation, no interpolation of the selection). Unlike the 1-halo
explicit-3d backend, DSigma_max is z-dependent (the biased two-halo term) and
the max is nonlinear, so the z integral cannot be contracted past the
profile: this module uses the z-RESOLVED explicit-3d weight
(shared.explicit_grid_core.explicit_mass_z_weights, Sigma_crit_inv folded
into the z factors) and performs the double fixed-GL contraction
sum_kq W2d * DSigma_max per (bin, R) — the same contraction as the fixed-GL
shear1h2h_max backend, whose S_ij-tabulation error this reference
isolates.

DataBlock contract
------------------
Reads (options): the union of the 3d counts options
    (lam_min/lam_max/zob_min/zob_max/sigma_z per bin, zt/lnm envelope,
    n_lnm/n_z/n_q/l_lam) and the shear grid (bin_index, r_perp,
    lob_centers), plus include_miscentering (default T).
Reads (datablock): the counts 3d contract plus
    average_sigma_crit_inv/{zlense,sci_average},
    halomodel/{r_sigma, z, lnM, dSigma_nfw, dSigma_hh, bias}
    (compute_lensing_2h = T), miscentering/{f_mis,tau_mis} (REQUIRED —
    no in-code default fallback), and the fixed gamma miscentring
    tables.
Writes: shear1h2h_max_explicit_gl/vals  (n_bins * n_r,)  [hardcoded section]

Status: reference implementation. The fixed-GL shear1h2h_max backends
validate against this module.
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
from shared import explicit_grid_core
from shared import lensing_profiles as lp
from systematics.selection_richness.python import sel_kernels

OUTPUT_SECTION = "shear1h2h_max_explicit_gl"


def compute_shear_max(profile, lnm_x, lnm_w, z_x, w2d, bin_index, r_perp,
                      physical=False):
    """O(R) = sum_kq lnm_w_k W2d[b,k,q] DSigma_max(b, R, lnM_k, z_q).

    physical: fold one_halo_physical_density's exact identity
    DSigma_phys(R|z) = (1+z)^2 DSigma_frozen(R(1+z)) into the 1-halo
    term. The z-axis is already resolved here (the 2-halo term forces
    it), so this just evaluates the 1-halo mixture PER z-node with
    q=1+z instead of once at q=1 -- no weight restructuring needed.
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
    for key in ("lam_min", "lam_max", "zob_min", "zob_max", "sigma_z"):
        cfg[key] = sf._read_array(options, option_section, key)
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
    for key, default in (("n_lnm", 96), ("n_z", 64), ("n_q", 32)):
        try:
            cfg[key] = int(options.get_int(option_section, key))
        except Exception:
            cfg[key] = default
    try:
        cfg["l_lam"] = float(options.get_double(option_section, "l_lam"))
    except Exception:
        cfg["l_lam"] = 6.0
    try:
        cfg["include_miscentering"] = bool(
            options.get_bool(option_section, "include_miscentering"))
    except Exception:
        cfg["include_miscentering"] = True
    return cfg


def execute(block, cfg):
    t0 = time.perf_counter()
    sf = sel_kernels.load()
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

    lnm_x, lnm_w, z_x, w2d = explicit_grid_core.explicit_mass_z_weights(
        cfg, sf._read_mor(block), sf._make_plob_splines(block),
        dm.HMF(source), dm.DVDoDz(source), sci=dm.SigmaCritInv(source),
        zt_low=cfg["zt_low"], zt_high=cfg["zt_high"],
        lnm_low=cfg["lnm_low"], lnm_high=cfg["lnm_high"],
        n_lnm=cfg["n_lnm"], n_z=cfg["n_z"], n_q=cfg["n_q"],
        l_lam=cfg["l_lam"])

    block[OUTPUT_SECTION, "vals"] = compute_shear_max(
        profile, lnm_x, lnm_w, z_x, w2d, cfg["bin_index"], cfg["r_perp"],
        physical=physical)
    dt_ms = 1000.0 * (time.perf_counter() - t0)
    print(f"[shear1h2h_max_explicit_gl] {cfg['bin_index'].size} bins x "
          f"{cfg['r_perp'].size} radii "
          f"({cfg['n_lnm']}x{cfg['n_z']}x{cfg['n_q']} GL, max 1h/2h) — "
          f"{dt_ms:.0f} ms", flush=True)
    return 0


def cleanup(config):
    return 0
