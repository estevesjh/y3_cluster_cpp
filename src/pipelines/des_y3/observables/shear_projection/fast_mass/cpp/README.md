# Projection shear — `fast_mass` (C++)

**Status: reference backend** (validated 2026-08-12). Production
remains `ShearPrjFrozenPhysics.so`. Built as `ShearPrjFastMass.so`.

A thin des_y3 driver over the immutable exact core
(`sp_detail::ShearPrjCore`): the exact-z fast_mass computation, with
both observables published from ONE core pass —
`dsigma_prj_fast_mass/{vals,rnd,cl}` and
`shear_prj_fast_mass/{vals,rnd,cl}` — the same hardcoded sections as
the [Python backend](../python/README.md), so the two are drop-in
interchangeable (never run both in one pipeline: DataBlock sections
don't overwrite).

Ini section `ShearPrjFastMass`, ShearPrjCore wall + knobs (the
180-point zipped wall, zt/lnm bounds, R_max_cMpch, n_lnm, n_per_seg,
n_zring, n_zouter, lob_centers).

Validation (real pipeline, fiducial point, 180-point wall): 9.9e-12 vs
the exact `dsigma_prj` evaluator (same core), 1.0e-11 vs the Python
backend. Cost: **154 ms/sample for both observables** (the two
existing single-observable evaluators cost ~240 ms each; Python
backend 270 ms; frozen production 82 ms).
