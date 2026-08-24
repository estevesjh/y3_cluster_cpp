# ⚠ Frozen-physics ΔΣ_prj backend: rnd channel incompatible with the signed mis-centering kernel

**Raised 2026-08-24** during the issue-4 merge campaign, by
`test/des_y3_pipeline.test.py::TestShearPrjGl::test_matches_exact_evaluator_and_frozen_physics`
against a fresh `cosmosis-models/real_pipeline_extract_prj2h.ini` dump.

## The measurement (fiducial sample, 12-bin × 15-radius wall)

| Comparison | unsigned era (tip 8960bef dump) | signed era (merged) |
|---|---|---|
| frozen vs exact, `vals = rnd + cl` | max rel 1.9e-3 | max rel 1.2e+2 (median 6.5e-3) |
| frozen vs exact, `cl` channel | — | max rel 1.7e-2 |
| frozen vs exact, `rnd` channel | — | max rel 2.3e+2 |
| frozen error / pre-cancellation scale max(rnd,cl) | — | max 5.9e-1, median 6.5e-3 |

## Interpretation

The signed `single` ΔΣ_mis table (`deltasigma_signed_single.txt`) restores
the physical negative lobe at `r_mis > r`, which makes the exact
evaluator's mean-field (`rnd`) term cancel as it must (at the fiducial
sample the exact `rnd` at the outermost radius drops from +5.50 to −0.556,
a truncation-level residual). The frozen-physics backend
(`ShearPrjFrozenPhysics.so`, `src/models/sigma_prj_frozen_t.hh` /
`_interp_t.hh`) freezes the projection integrand's shape and rescales it
with an anchored amplitude — algebra derived in the unsigned era, where
`rnd` was a large, everywhere-positive term. It does not reproduce a
signed integrand whose integral cancels: its `rnd` output no longer
tracks the exact one at all. The historical "< 0.2%" frozen-physics
bound was measured on the non-cancelled (unsigned) values and is void
for `rnd`/`vals` in the signed era.

## Impact

- **Production is unaffected**: the likelihood consumes `shear_prj/cl`
  only (the rnd mean-field must cancel by construction —
  RichnessSelection#1), and the production evaluators
  (`DSigmaPrjEvaluator`/`ShearPrjEvaluator`, `sp_detail::ShearPrjCore`)
  are exact. The Python reference matches the exact evaluator at 1e-8.
- The frozen backend's `cl` channel is still usable as a fast benchmark
  at the ~2e-2 level (measured envelope, fiducial sample).
- Any consumer of frozen `rnd` or frozen `vals` is broken until the
  frozen amplitude algebra is re-derived for the signed kernel.

## Pinned by

`test/des_y3_pipeline.test.py::TestShearPrjGl::test_matches_exact_evaluator_and_frozen_physics`:
the `cl`-channel guard (≤ 2e-2) must stay green; the `vals` comparison at
the historical 3e-3 is kept DELIBERATELY red until the frozen algebra is
fixed or the backend is demoted to a cl-only benchmark.
