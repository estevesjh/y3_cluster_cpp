#!/usr/bin/env python
"""Validate the fixed-GL shear backend against Shear1hMisSel.so.

Same exact z contraction (SelGLCore twin), same GL nodes, and
interpolation-exact replicas of both profile tables — the comparison
should be at (near) machine precision; any drift here means one of the
replicas no longer matches the production conventions.

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

from shear1h_gl import compute_shear  # noqa: E402  (same dir)

TOL = 1e-10
R_PERP = np.array([0.20000, 0.28599, 0.40896, 0.58480, 0.83625,
                   1.19581, 1.70998, 2.44521, 3.49658, 5.00000])
ZT_LO, ZT_HI = 0.05, 0.80
LNM_LO, LNM_HI = 29.9336, 36.7300


def main():
    dump = (Path(sys.argv[1]) if len(sys.argv) > 1 else
            sel_kernels.repo_root() / "docs" / "figs"
            / "real_pipeline_extract_output")
    if not dump.is_dir():
        sys.exit(f"dump directory not found: {dump}")
    source = dm.DumpSource(str(dump))
    weights = dm.MassZWeights(source, n_lnm=96, n_z=64,
                              zt_lo=ZT_LO, zt_hi=ZT_HI,
                              lnm_lo=LNM_LO, lnm_hi=LNM_HI,
                              include_sci=True)
    profile = lp.MisMixtureProfile(
        source, lob_centers=dm.DEFAULT_LOB_CENTERS,
        f_mis=dm.F_MIS_DEFAULT, tau_mis=dm.TAU_MIS_DEFAULT,
        omega_m=source.scalar("cosmological_parameters", "omega_m"))
    vals = compute_shear(weights, profile, np.arange(12), R_PERP)
    prod = source.array("shear1hmissel", "vals")
    worst = float(np.max(np.abs(vals / prod - 1.0)))
    print("fixed-GL shear vs production Shear1hMisSel.so "
          f"(12 bins x {R_PERP.size} radii): "
          f"max |ratio - 1| = {worst:.2e}  (tolerance {TOL:.0e})")
    if worst > TOL:
        sys.exit("FAIL")
    print("PASS")


if __name__ == "__main__":
    main()
