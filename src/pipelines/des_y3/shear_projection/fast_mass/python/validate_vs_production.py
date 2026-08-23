#!/usr/bin/env python
"""Validate the fast_mass projection shear against the C++ evaluators.

Replays a test-sampler dump that contains the projection chain
(b_sel_marginalised, xi_nl, halomodel bias, distances, plus the outputs
of the exact DSigmaPrjEvaluator [dsigma_prj] and the frozen production
module [dsigma_prj_frozen_physics]) and compares this port's exact-z
computation against both, per channel, on the pinned 180-point wall.

Expected: machine precision vs the exact evaluator (same grids, same
tables, same arithmetic); the measured frozen-physics approximation
size vs the production module (documented < 0.2%).

Usage:  python validate_vs_production.py <dump_dir>
        (a dump produced by a pipeline like the one in README.md)
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

from shared import datablock_models as dm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shear_prj_fast_mass import ShearPrjFastMass  # noqa: E402

TOL_EXACT = 1e-9
RADII = [0.0426, 0.0669, 0.1045, 0.1652, 0.2607, 0.4117, 0.6505, 1.0257,
         1.6181, 2.5537, 4.0265, 6.3490, 10.0107, 15.7832, 24.8771]


def pinned_wall():
    lb, zol, zoh, rr = [], [], [], []
    for zlo, zhi in ((0.20, 0.35), (0.35, 0.50), (0.50, 0.65)):
        for b in range(4):
            lb += [b] * 15
            zol += [zlo] * 15
            zoh += [zhi] * 15
            rr += RADII
    return dict(lambda_bin=np.array(lb), zo_low=np.array(zol),
                zo_high=np.array(zoh), radii=np.array(rr))


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: validate_vs_production.py <dump_dir> "
                 "(see README.md for the pipeline that produces one)")
    dump = Path(sys.argv[1])
    source = dm.DumpSource(str(dump))

    core = ShearPrjFastMass(pinned_wall(),
                            lob_centers=dm.DEFAULT_LOB_CENTERS)
    core.set_sample(source)
    rnd, cl, sci = core.wall_outputs()

    ex_r = source.array("dsigma_prj", "rnd")
    ex_c = source.array("dsigma_prj", "cl")
    ex_v = source.array("dsigma_prj", "vals")
    fr_v = source.array("dsigma_prj_frozen_physics", "vals")

    d_r = float(np.max(np.abs(rnd / ex_r - 1)))
    d_c = float(np.max(np.abs(cl / ex_c - 1)))
    d_v = float(np.max(np.abs((rnd + cl) / ex_v - 1)))
    d_f = float(np.max(np.abs((rnd + cl) / fr_v - 1)))
    print("fast_mass projection shear (180-point wall):")
    print(f"  vs exact DSigmaPrjEvaluator.so:  rnd {d_r:.2e}  cl {d_c:.2e}"
          f"  vals {d_v:.2e}  (tolerance {TOL_EXACT:.0e})")
    print(f"  vs frozen production module:     vals {d_f:.2e}  "
          "(the frozen-physics approximation, documented < 2e-3)")
    if max(d_r, d_c, d_v) > TOL_EXACT:
        sys.exit("FAIL: disagrees with the exact evaluator")
    print("PASS")


if __name__ == "__main__":
    main()
