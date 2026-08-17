#!/usr/bin/env python
"""Validate the fast_mass counts backend against NumCountsSel.so.

Same algorithm, same GL nodes, same bilinear S_stack interpolation —
the comparison should be at machine precision.

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
from shared import sel_kernels

TOL = 1e-12
ZT_LO, ZT_HI = 0.05, 0.80
LNM_LO, LNM_HI = 29.9336, 36.7300


def main():
    dump = (Path(sys.argv[1]) if len(sys.argv) > 1 else
            sel_kernels.repo_root() / "docs" / "figs"
            / "real_pipeline_extract_output")
    if not dump.is_dir():
        sys.exit(f"dump directory not found: {dump}")
    source = dm.DumpSource(str(dump))
    norm = dm.MassZWeights(source, n_lnm=96, n_z=64,
                           zt_lo=ZT_LO, zt_hi=ZT_HI,
                           lnm_lo=LNM_LO, lnm_hi=LNM_HI,
                           include_sci=False).norm()
    prod = source.array("numcountssel", "vals")
    worst = float(np.max(np.abs(norm / prod - 1.0)))
    print("fast_mass counts vs production NumCountsSel.so: "
          f"max |ratio - 1| = {worst:.2e}  (tolerance {TOL:.0e})")
    if worst > TOL:
        sys.exit("FAIL")
    print("PASS")


if __name__ == "__main__":
    main()
