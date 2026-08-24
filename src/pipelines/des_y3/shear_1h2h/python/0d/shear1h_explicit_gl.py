"""Miscentred one-halo shear — full (lambda_true, lnM, z) reference.

The `explicit-3d` reference for the shear observable: the explicit
quadruple composition

    O_ij(R) = int dz int dlnM int dlt  n(M,z) dV/dOmega/dz(z) Omega(z)
              Sigma_crit_inv(z) S_j(z) S_i(lt, z) P_HOD(lt | M, z)
              DSigma_i(R, lnM)

with every selection kernel evaluated at the quadrature nodes (no
S_ij tabulation, no interpolation) — the same shared explicit-3d
contraction the counts reference uses
(shared.explicit_grid_core.explicit_mass_weights, here with
Sigma_crit_inv folded into the z factors), contracted against the
production miscentred mixture DSigma_i (shared lensing_profiles, the
interpolation-exact haloModel + gamma-table pair). Because DSigma is
z-free (fixed concentration and reference density — the property the
radial-factorization study established), the lt and z integrals
commute past it exactly; the "quadruple" integral is the counts
triple integral weighted by DSigma in the final mass sum.

Difference vs the fixed-GL backend is precisely the production
S_ij-tabulation error; difference vs radial_series is tabulation +
truncation + the centred-profile convention (see the radial_series
README).

DataBlock contract
------------------
Reads (options): the union of the 3d counts options
    (lam_min/lam_max/zob_min/zob_max/sigma_z per bin, zt/lnm envelope,
    n_lnm/n_z/n_q/l_lam) and the shear grid (bin_index, r_perp,
    lob_centers).
Reads (datablock): the counts 3d contract plus
    average_sigma_crit_inv/{zlense,sci_average},
    halomodel/{r_sigma,lnM,dSigma_nfw}, miscentering/{f_mis,tau_mis},
    and the fixed gamma miscentring tables.
Writes: shear1h_explicit_gl/vals  (n_bins * n_r,)   [hardcoded section]

Status: reference implementation. Production remains Shear1hMisSel.so.
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

OUTPUT_SECTION = "shear1h_explicit_gl"


def compute_shear(bins, mor, plob_splines, hmf, dv, sci, profile,
                  bin_index, r_perp, *, zt_low, zt_high, lnm_low, lnm_high,
                  n_lnm=96, n_z=64, n_q=32, l_lam=6.0):
    """Full-reference O(R) for the requested bins."""
    lnm_x, lnm_w, weights = explicit_grid_core.explicit_mass_weights(
        bins, mor, plob_splines, hmf, dv, sci=sci,
        zt_low=zt_low, zt_high=zt_high,
        lnm_low=lnm_low, lnm_high=lnm_high,
        n_lnm=n_lnm, n_z=n_z, n_q=n_q, l_lam=l_lam)
    n_r = len(r_perp)
    vals = np.empty(len(bin_index) * n_r)
    for i, b in enumerate(bin_index):
        DSigma = profile(b, np.asarray(r_perp)[:, None], lnm_x[None, :])
        vals[i * n_r:(i + 1) * n_r] = DSigma @ (lnm_w * weights[b])
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
    return cfg


def execute(block, cfg):
    t0 = time.perf_counter()
    sf = sel_kernels.load()
    source = dm.DataBlockSource(block)
    profile = lp.MisMixtureProfile(
        source, lob_centers=cfg["lob_centers"],
        # Required: no fallback to the fiducial defaults — a pipeline
        # missing the miscentering section must fail loudly.
        f_mis=source.scalar("miscentering", "f_mis"),
        tau_mis=source.scalar("miscentering", "tau_mis"),
        omega_m=source.scalar("cosmological_parameters", "omega_m"))

    block[OUTPUT_SECTION, "vals"] = compute_shear(
        cfg, sf._read_mor(block), sf._make_plob_splines(block),
        dm.HMF(source), dm.DVDoDz(source), dm.SigmaCritInv(source),
        profile, cfg["bin_index"], cfg["r_perp"],
        zt_low=cfg["zt_low"], zt_high=cfg["zt_high"],
        lnm_low=cfg["lnm_low"], lnm_high=cfg["lnm_high"],
        n_lnm=cfg["n_lnm"], n_z=cfg["n_z"], n_q=cfg["n_q"],
        l_lam=cfg["l_lam"])
    dt_ms = 1000.0 * (time.perf_counter() - t0)
    print(f"[shear1h_explicit_gl] {cfg['bin_index'].size} bins x "
          f"{cfg['r_perp'].size} radii "
          f"({cfg['n_lnm']}x{cfg['n_z']}x{cfg['n_q']} GL) — {dt_ms:.0f} ms",
          flush=True)
    return 0


def cleanup(config):
    return 0
