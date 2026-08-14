"""Number counts via the fast_mass strategy — Python backend.

The plan's `fast_mass` strategy made explicit: the redshift integral is
performed exactly, on fixed GL nodes, OUTSIDE the mass operator —

    W_ij(lnM) = int dz  n(M,z) dV/dOmega/dz(z) Omega(z) S_ij(lnM,z)
    N_ij      = int dlnM  W_ij(lnM)                       (f = 1)

with S_ij the tabulated selection tensor from sel_function.py. This is
exactly the computation the production NumCountsSel.so performs
(nosel_gl_detail::SelGLCore, same GL node placement, same bilinear
S_stack interpolation), expressed through the shared Python replica
`des_y3.shared.datablock_models.MassZWeights` — which is why this
backend agrees with production to machine precision (measured 2.4e-15;
see README.md). It exists so the namespace carries a readable,
importable statement of the production algorithm that the full_ltmz
references and the radial_series moments all build on.

DataBlock contract
------------------
Reads (options): bin_index (wall; default = every S_stack bin),
    zt_low, zt_high, lnm_low, lnm_high (required GL envelope),
    n_lnm (96), n_z (64).
Reads (datablock): sel_function/{lnM,z,S_stack},
    mass_function/{m_h,z,dndlnmh}, cluster_abundance/{hmf_s,hmf_q},
    distances/{z,d_a},
    cosmological_parameters/{h0,omega_m,omega_nu,omega_lambda,omega_k}.
Writes: numcounts_fast_mass/vals  (len(bin_index),)  [hardcoded section]

Status: reference re-expression of the production algorithm.
Production remains NumCountsSel.so.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

for _p in Path(__file__).resolve().parents:
    if (_p / "des_y3" / "shared" / "datablock_models.py").is_file():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

from des_y3.shared import datablock_models as dm
from des_y3.shared import sel_kernels

OUTPUT_SECTION = "numcounts_fast_mass"


def setup(options):
    from cosmosis.datablock import option_section
    sf = sel_kernels.load()
    cfg = {}
    try:
        cfg["bin_index"] = sf._read_array(options, option_section,
                                          "bin_index").astype(int)
    except Exception:
        cfg["bin_index"] = None            # default: every S_stack bin
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
    weights = dm.MassZWeights(
        dm.DataBlockSource(block), n_lnm=cfg["n_lnm"], n_z=cfg["n_z"],
        zt_lo=cfg["zt_low"], zt_hi=cfg["zt_high"],
        lnm_lo=cfg["lnm_low"], lnm_hi=cfg["lnm_high"], include_sci=False)
    norm = weights.norm()
    bins = (cfg["bin_index"] if cfg["bin_index"] is not None
            else np.arange(weights.n_bins))
    block[OUTPUT_SECTION, "vals"] = norm[bins]
    dt_ms = 1000.0 * (time.perf_counter() - t0)
    print(f"[numcounts_fast_mass] {len(bins)} bins "
          f"({cfg['n_lnm']}x{cfg['n_z']} GL) — {dt_ms:.0f} ms", flush=True)
    return 0


def cleanup(config):
    return 0
