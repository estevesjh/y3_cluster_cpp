#!/usr/bin/env python
"""Validate the full_ltmz shear reference.

Two comparisons on the real extraction dump, all 12 pinned bins on the
production radial grid:

1. vs the fast_mass backend (same profile, same GL mass/z nodes; the
   only difference is direct kernel evaluation vs the production S_ij
   tabulation) — isolates the tabulation error, expected at the same
   few-1e-4..1e-3 level the counts references measured;
2. vs production Shear1hMisSel.so — same information plus the .so's own
   arithmetic, reported for the record.

Usage: python validate_vs_production.py [dump_dir]
"""
from __future__ import annotations

import sys
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

from shear1h_full_ltmz import compute_shear  # noqa: E402  (same dir)

TOL = 5e-3
R_PERP = np.array([0.20000, 0.28599, 0.40896, 0.58480, 0.83625,
                   1.19581, 1.70998, 2.44521, 3.49658, 5.00000])
BINS = dict(
    lam_min=np.array([20., 30., 45., 60.] * 3),
    lam_max=np.array([30., 45., 60., 200.] * 3),
    zob_min=np.array([0.20] * 4 + [0.35] * 4 + [0.50] * 4),
    zob_max=np.array([0.35] * 4 + [0.50] * 4 + [0.65] * 4),
    sigma_z=np.full(12, 0.03),
)
ZT_LO, ZT_HI = 0.05, 0.80
LNM_LO, LNM_HI = 29.9336, 36.7300


def main():
    dump = (Path(sys.argv[1]) if len(sys.argv) > 1 else
            sel_kernels.repo_root() / "docs" / "figs"
            / "real_pipeline_extract_output")
    if not dump.is_dir():
        sys.exit(f"dump directory not found: {dump}")
    source = dm.DumpSource(str(dump))
    mor = sel_kernels.mor_from_source(source)
    plob = sel_kernels.plob_splines_default()
    profile = lp.MisMixtureProfile(
        source, lob_centers=dm.DEFAULT_LOB_CENTERS,
        f_mis=dm.F_MIS_DEFAULT, tau_mis=dm.TAU_MIS_DEFAULT,
        omega_m=source.scalar("cosmological_parameters", "omega_m"))
    sci = dm.SigmaCritInv(source)

    full = compute_shear(BINS, mor, plob, dm.HMF(source),
                         dm.DVDoDz(source), sci, profile,
                         np.arange(12), R_PERP,
                         zt_low=ZT_LO, zt_high=ZT_HI,
                         lnm_low=LNM_LO, lnm_high=LNM_HI)

    from shared import datablock_models  # noqa: F401
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                           / "python" / "0d"))
    from shear1h_fast_mass import compute_shear as fast_shear
    weights = dm.MassZWeights(source, n_lnm=96, n_z=64,
                              zt_lo=ZT_LO, zt_hi=ZT_HI,
                              lnm_lo=LNM_LO, lnm_hi=LNM_HI,
                              include_sci=True)
    fast = fast_shear(weights, profile, np.arange(12), R_PERP)
    prod = source.array("shear1hmissel", "vals")

    dev_fast = float(np.max(np.abs(full / fast - 1.0)))
    dev_prod = float(np.max(np.abs(full / prod - 1.0)))
    print("full_ltmz shear (12 bins x 10 radii):")
    print(f"  vs fast_mass backend (isolates S_ij tabulation): "
          f"max |ratio - 1| = {dev_fast:.2e}")
    print(f"  vs production Shear1hMisSel.so:                  "
          f"max |ratio - 1| = {dev_prod:.2e}  (tolerance {TOL:.0e})")
    if dev_prod > TOL:
        sys.exit("FAIL")
    print("PASS")


if __name__ == "__main__":
    main()
