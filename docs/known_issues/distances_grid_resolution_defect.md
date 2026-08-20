# Coarse 50-point `distances` grid biases every projection-chain observable at the ~1% level

**Status:** open (found 2026-08-20 by the issue-#11 quad-truth validation)
**Component:** `cp_camb.py` distances output (`nz = 50` over z ∈ [0, 4],
Δz = 0.0816) + `p_operator_t.hh` / `sigma_prj_t.hh` chi handling.

## What is wrong

1. **Input defect.** `cp_camb` publishes `distances/{z, d_c, d_a}` on the
   same z-grid as P(k): `nz = 50` linear points over [0, 4]
   (Δz ≈ 0.0816). Every consumer linearly interpolates this table.
   Against the exact comoving-distance integral at the fiducial
   cosmology, the piecewise-linear chi has
   - an absolute sagitta of up to **−1.1 cMpc/h** (mid-segment), and
   - a **local slope error of up to −1.4%** (at zob = 0.425;
     −0.6% at 0.275 / 0.575 — a *sawtooth in z* whose phase is set by
     where the survey z-bins fall relative to the table nodes).

   The slope error directly distorts the line-of-sight Δχ geometry of
   the exclusion ring in the P[X] operators and the projection
   integrands: Δχ_∥ scales with the local dχ/dz, and ξ(Δχ) ~ Δχ^−1.8
   amplifies it ~2×.

2. **Double-resampling defect (C++).** `p_operator_t.hh` resamples chi
   from the coarse table onto its own linear `zt_ref` grid
   (N=80 over [0.05, 0.90]) and interpolates *that*, while `chi_o` is
   evaluated directly from the table. Where a table kink falls inside a
   zt_ref segment near zob (e.g. table node z = 0.5714 at
   zob = 0.575), the resampled chi is off by up to **0.21 cMpc/h ≈ 17%
   of R_excl** across the ring, hitting only the ξ-carrying operators.

## Measured impact (fiducial extract pipeline, nz=50 → nz=400)

Rerunning `docs/figs/real_pipeline_extract_prj2h.ini` with
`cp_camb.nz=400` moves the production outputs by:

| Output | max shift | structure |
|---|---|---|
| `b_sel_marg_P1` | 0.27% | z-sawtooth |
| `b_sel_marg_I1` | 0.92% | z-sawtooth, mid-bin worst |
| `b_sel_marg_J`  | 0.80% | z-sawtooth |
| `dsigma_prj` / `shear_prj_frozen_physics` vals | **1.20%** | rnd 0.65%, **cl (correlated/2h) channel 1.70%** |
| `shear1hmissel` | 0.85% | |
| `numcountssel`  | 0.69% | |

Buzzard relevance: a z-structured, cl-channel-heavy distortion of the
projection shear is exactly the class of systematic the Buzzard
large-scale misfit hunt (issue #11 / `docs/buzzard_convergence_plan.md`)
is looking for. This is not shown to be *the* driver, but it is real,
percent-level, and one ini knob away from removal.

## Separately: wall operator truncation at production knobs

With a dense (nz=400) chi table on both sides, the C++ wall still
deviates from the adaptive quad-truth
(`test/make_bsel_quad_truth_pins.py`) by

- P1: max 1.2e-3 (median 5.9e-4)
- I1: max 4.8e-3 (median 2.1e-3)
- J:  max 4.3e-3 (median 2.7e-3)

i.e. the production ini knobs (`n_zring=20, n_zouter=20, n_theta=10,
n_lt=60, n_lnm=24`) do NOT reach the sub-0.01% the companion doc quotes
for N_z=80 defaults. (The theta-outer restructure's global-grid
exclusion masking was tested and exonerated: ≤3e-4.)

## Fix

- **Input:** raise `cp_camb` `nz` (e.g. 400 → Δz=0.01) in the pipeline
  inis. Cost: astropy distance evaluation is trivial; every consumer
  reads the table through `Interp1D`/`np.interp` and needs no change.
  NOTE: the widePlanck self-closure data vectors were generated at
  nz=50 — regenerate DV and model together, or fix only the Buzzard
  override ini first.
- **C++:** stop double-resampling chi through `zt_ref` in
  `p_operator_t.hh` (evaluate the table directly at the z nodes), or
  make `zt_ref` dense enough to be irrelevant once the table is dense.
- **Wall knobs:** bump `n_zouter` to the doc's 30 (and re-measure) if
  the remaining few-1e-3 matters downstream.

## Pinned by

`test/bsel_external.test.py` — quad-truth pins at the project-default
1e-3 tolerance; the I1/J rows are DELIBERATELY red until the fixes land
(same convention as the HOD-normalization and radial-series pins).
