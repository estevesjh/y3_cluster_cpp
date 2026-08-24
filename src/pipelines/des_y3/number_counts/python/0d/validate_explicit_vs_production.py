#!/usr/bin/env python
"""Validate 3d counts against the production fixed-GL pipeline.

Replays a real test-sampler dump (docs/figs/real_pipeline_extract.ini —
real HMF, distances, HOD parameters) and compares the explicit
(lambda_true, lnM, z) triple integral against the production
NumCountsSel.so output for all 12 pinned wall-grid bins.

The two paths share the HOD/EMG/photo-z kernels (both go through
sel_function.py) and the HMF/volume/area conventions (datablock_models
replicas of the C++ structs); the residual isolates what the production
fast path adds on top of the physics: the fixed (192 x 64) S_ij
tabulation and its bilinear interpolation onto the GL nodes. Pass
criterion: |N_full / N_production - 1| <= 5e-3 per bin.

Usage:
    python validate_vs_production.py [dump_dir]

dump_dir defaults to docs/figs/real_pipeline_extract_output; run
`cosmosis docs/figs/real_pipeline_extract.ini` first if it is missing.
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
from systematics.selection_richness.python import sel_kernels

from numcounts_explicit_gl import compute_counts  # noqa: E402  (same dir)

TOL = 5e-3

# The pinned 12-bin wall (mock_mcmc_buzzard.ini sel_function section) and
# the production NumCountsSel integration envelope.
BINS = dict(
    lam_min=np.array([20., 30., 45., 60.] * 3),
    lam_max=np.array([30., 45., 60., 200.] * 3),
    zob_min=np.array([0.20] * 4 + [0.35] * 4 + [0.50] * 4),
    zob_max=np.array([0.35] * 4 + [0.50] * 4 + [0.65] * 4),
    sigma_z=np.full(12, 0.03),
)
ZT_LOW, ZT_HIGH = 0.05, 0.80
LNM_LOW, LNM_HIGH = 29.9336, 36.7300


def main():
    if len(sys.argv) > 1:
        dump = Path(sys.argv[1])
    else:
        dump = (sel_kernels.repo_root() / "docs" / "figs"
                / "real_pipeline_extract_output")
    if not dump.is_dir():
        sys.exit(f"dump directory not found: {dump}\n"
                 "run `cosmosis docs/figs/real_pipeline_extract.ini` first")

    source = dm.DumpSource(str(dump))
    mor = sel_kernels.mor_from_source(source)
    plob = sel_kernels.plob_splines_default()

    vals = compute_counts(
        BINS, mor, plob, dm.HMF(source), dm.DVDoDz(source),
        zt_low=ZT_LOW, zt_high=ZT_HIGH,
        lnm_low=LNM_LOW, lnm_high=LNM_HIGH)

    n_prod = source.array("numcountssel", "vals")
    print("explicit-3d (explicit lambda_tr, lnM, z) vs production "
          "NumCountsSel.so (fixed-GL):")
    print(f"  {'bin':>3s} {'N_explicit-3d':>14s} {'N_production':>14s} "
          f"{'ratio':>9s}")
    worst = 0.0
    for b in range(vals.size):
        ratio = vals[b] / n_prod[b]
        worst = max(worst, abs(ratio - 1.0))
        print(f"  {b:3d} {vals[b]:14.6e} {n_prod[b]:14.6e} {ratio:9.5f}")
    print(f"  max |ratio - 1| = {worst:.2e}  (tolerance {TOL:.0e})")
    if worst > TOL:
        sys.exit("FAIL: 3d counts outside tolerance")
    print("PASS")


if __name__ == "__main__":
    main()
