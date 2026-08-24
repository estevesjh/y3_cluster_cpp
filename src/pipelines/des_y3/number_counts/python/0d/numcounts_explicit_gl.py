"""Full (lambda_true, lnM, z) reference number counts — `explicit-3d`.

The readable reference calculation for the DES Y3 cluster counts
(src/pipelines/des_y3/README.md, strategy `explicit-3d`): the expected
count in richness bin i and photo-z bin j is the explicit triple integral

    N_ij = int dz int dlnM int dlambda_tr
           n(M,z) dV/dOmega/dz(z) Omega(z)
           K_j(z) S_i(lambda_tr, z) P_HOD(lambda_tr | M, z)

with every kernel evaluated *at the quadrature nodes* — no intermediate
S_ij(lnM, z) tabulation and no bilinear interpolation. The production
fast path (`fixed-GL`: sel_function.py -> NumCountsSel.so) computes the
same quantity by tabulating S_ij once on a fixed (lnM, z) grid and
contracting redshift first; the difference between the two is exactly the
production tabulation/interpolation error, which validation bounds below
0.5% per bin (see README.md and validate_vs_production.py).

Integration variables and quadrature (all fixed Gauss-Legendre, this
project's integrator convention):

    z          GL on [zt_low, zt_high]         n_z    nodes (default 64)
    lnM        GL on [lnm_low, lnm_high]       n_lnm  nodes (default 96)
    lambda_tr  GL on the per-(M,z) HOD bracket
               [max(0, mu_eff - L_lam sig_eff), mu_eff + L_lam sig_eff]
                                                N_q    nodes (default 32)

The lambda_tr bracket, the shifted-Poisson HOD form, the EMG observed-
richness kernel S_i (CDF differencing at the unique bin edges) and the
Gaussian photo-z kernel K_j are *reused* from the maintained
sel_function.py — not reimplemented — so this reference and the
production selection stage share one set of kernels by construction.

DataBlock contract
------------------
Reads (ini options, sel_function conventions):
    lam_min, lam_max, zob_min, zob_max, sigma_z   per-bin arrays (len n_bins)
    zt_low, zt_high, lnm_low, lnm_high            integration envelope
    n_lnm, n_z, n_q, l_lam                        quadrature knobs
Reads (datablock):
    cluster_mor/{log10_Mmin, log10_ratio | log10_M1, alpha, epsilon,
                 sigma_lambda, z_pivot?}
    plob_ltr_params/*                             optional EMG table
    mass_function/{m_h, z, dndlnmh}
    cluster_abundance/{hmf_s, hmf_q}
    distances/{z, d_a}
    cosmological_parameters/{h0, omega_m, omega_nu, omega_lambda, omega_k}
Writes:
    numcounts_explicit_gl/vals    expected counts, (n_bins,)

The output section is hardcoded (not an ini knob) for the same reason the
fixed-GL C++ evaluators hardcode theirs: CosmoSIS [DEFAULT] blocks
propagate keys like output_section into every module section.

Status: reference implementation (validated 2026-08-11 against the pinned
production pipeline; see README.md). Not a production entry point.
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
from systematics.selection_richness.python import sel_kernels

OUTPUT_SECTION = "numcounts_explicit_gl"


def compute_counts(bins, mor, plob_splines, hmf, dv, *,
                   zt_low, zt_high, lnm_low, lnm_high,
                   n_lnm=96, n_z=64, n_q=32, l_lam=6.0):
    """The explicit-3d triple integral for every configured bin.

    ``bins`` is a dict of equal-length arrays (lam_min, lam_max, zob_min,
    zob_max, sigma_z); ``hmf`` and ``dv`` are callables with the shared
    datablock_models conventions. Returns (n_bins,) expected counts.

    The (lambda_true, z) contraction itself lives in the shared
    explicit-3d core (shared.explicit_grid_core), which the shear
    explicit-3d backend contracts against its radial profile instead of
    against 1.
    """
    lnm_x, lnm_w, weights = explicit_grid_core.explicit_mass_weights(
        bins, mor, plob_splines, hmf, dv, sci=None,
        zt_low=zt_low, zt_high=zt_high,
        lnm_low=lnm_low, lnm_high=lnm_high,
        n_lnm=n_lnm, n_z=n_z, n_q=n_q, l_lam=l_lam)
    return weights @ lnm_w


# ---------------------------------------------------------------------------
# CosmoSIS module entry points
# ---------------------------------------------------------------------------

def setup(options):
    from cosmosis.datablock import option_section
    sf = sel_kernels.load()
    cfg = {}
    for key in ("lam_min", "lam_max", "zob_min", "zob_max", "sigma_z"):
        cfg[key] = sf._read_array(options, option_section, key)
    n = cfg["lam_min"].size
    for key in ("lam_max", "zob_min", "zob_max", "sigma_z"):
        if cfg[key].size != n:
            raise ValueError(
                f"numcounts_explicit_gl: axis '{key}' size {cfg[key].size} != {n}")

    cfg["zt_low"] = sf._read_scalar_or_first(options, option_section,
                                             "zt_low", 0.05)
    cfg["zt_high"] = sf._read_scalar_or_first(options, option_section,
                                              "zt_high", 0.80)
    cfg["lnm_low"] = sf._read_scalar_or_first(options, option_section,
                                              "lnm_low", np.log(1.0e13))
    cfg["lnm_high"] = sf._read_scalar_or_first(options, option_section,
                                               "lnm_high", np.log(9.0e15))
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
    mor = sf._read_mor(block)
    plob = sf._make_plob_splines(block)

    vals = compute_counts(
        cfg, mor, plob, dm.HMF(source), dm.DVDoDz(source),
        zt_low=cfg["zt_low"], zt_high=cfg["zt_high"],
        lnm_low=cfg["lnm_low"], lnm_high=cfg["lnm_high"],
        n_lnm=cfg["n_lnm"], n_z=cfg["n_z"], n_q=cfg["n_q"],
        l_lam=cfg["l_lam"])

    block[OUTPUT_SECTION, "vals"] = vals
    dt_ms = 1000.0 * (time.perf_counter() - t0)
    print(f"[numcounts_explicit_gl] {vals.size} bins "
          f"({cfg['n_lnm']}x{cfg['n_z']}x{cfg['n_q']} GL) — {dt_ms:.0f} ms",
          flush=True)
    return 0


def cleanup(config):
    return 0
