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

## Second observation (2026-08-24): the GPU port's cl channel is broken outright

First-ever execution of `ShearPrjFrozenGpu.so` (built on the branch's
Mac, never runnable before this Perlmutter session), driven in-pipeline
next to its CPU counterpart on the identical sample:

- `rnd`: matches the CPU frozen module to 2.6e-12 (machine precision) —
  the device ΔΣ_mis cache and the mean-field sweep are correct.
- `cl`: 15 NaNs (the entire zob=0.575 wall slice) and denormal garbage
  (values down to -1e-211) elsewhere; the finite subset deviates from
  the CPU cl by up to 100%. Signature of an uninitialized / mis-indexed
  device buffer in the cl cache path (the n_theta_max-flattened scl
  output), not a physics disagreement.

Also fixed in the merge campaign: `test/shear_prj_frozen_gpu.test.cu`
used to dlopen a hardcoded STALE tree's module
(`/pscratch/.../y3_cluster_cpp/gpu-build/...`); it now defaults to the
path CMake bakes in for THIS build's ShearPrjFrozenGpu target
(`SHEAR_PRJ_FROZEN_GPU_SO` still overrides). The test is red for the
real reason above.

## Pinned by

`test/des_y3_pipeline.test.py::TestShearPrjGl::test_matches_exact_evaluator_and_frozen_physics`:
the `cl`-channel guard (≤ 2e-2) must stay green; the `vals` comparison at
the historical 3e-3 is kept DELIBERATELY red until the frozen algebra is
fixed or the backend is demoted to a cl-only benchmark.

## GPU `cl` root cause, isolated (2026-08-27, issue #24 comment)

Two separate bugs in `shear_prj_frozen_gpu_t.cuh`, not a single
device-buffer indexing issue as originally suspected:

1. **(Fixed)** `y3_cuda::NFW_DSIGMA_MIS` had no `r_s(lnM)` accessor
   (unlike the CPU twin, `nfw_dsigma_mis.hh:129`), so the frozen-GPU
   `cl`-channel amplitude anchor hand-computed `r_200/4.0` with a
   hardcoded fixed `c=4` and a bare `rho_crit,0` constant instead of
   the profile's actual `rho_m_ref`/per-mass concentration. Added
   `r_s()` to `nfw_dsigma_mis.cuh` (mirrors the CPU formula exactly)
   and switched the anchor to call it. This alone changed the failure
   mode from overflow (`cl` up to ~1e232) to near-zero/denormal (`cl`
   ~1e-310) — real, verified improvement, but not sufficient on its
   own.
2. **(Open)** `bsel_k[k][it]` (the b_sel(θ) plateau weight) is NaN at
   zob=0.425 and denormal garbage at zob=0.575. Root cause: the
   frozen-GPU module hand-rolls a zob-interpolation over
   `b_sel_marginalised/{zob,lob,b_small,b_large}` assuming they form a
   dense cartesian `n_zob × n_lob` grid, indexed `j0*n_lob+lob_bin`.
   They don't -- per `src/models/bsel_bins_t.hh`, that section is a
   flat list of *exact wall rows* (12 for the fiducial 4-λ×3-z wall),
   not sorted/unique by zob. The CPU frozen counterpart
   (`sigma_prj_frozen_interp_t.hh:182`) does an **exact**
   `(lambda_bin, zob)` lookup via `BSelBins::at`, no interpolation.
   The GPU module's zob-walk overruns past the true 12-element arrays
   for any zob beyond the first slice, reading adjacent heap memory.
   Fix: replace the ad-hoc interpolation block with a `BSelBins`
   instance + `.at(lob_bin, zob)`, matching the CPU reference. Not
   landed yet -- see the issue #24 comment thread for full detail.
