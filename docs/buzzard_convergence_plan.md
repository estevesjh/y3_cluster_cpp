# Buzzard convergence campaign — issue triage and validation plan

**Date:** 2026-08-20
**Goal:** make the Buzzard-mock pipeline fit converge on the input cosmology.
**Scope:** all open issues in `estevesjh/y3_cluster_cpp` (#2, #3, #4, #5, #6,
#8, #9, #10, #11) and `estevesjh/des-nersc-cluster-scripts` (#1, #2), plus the
external-validation unit-test suite requested for P(k), ξ, ΔΣ, b_sel(θ), and
shear_prj.

---

## 1. Where the fit stands (evidence, most recent first)

All from `des-nersc-cluster-scripts`, branch
`polychord-widePlanck-logspace-ab`:

1. **Scale-split under realistic covariance** (commit `aa722d2`, jobs
   56961355/365/370): fitting shear on **large scales only [1.2–5]** rails
   Ω_m to the prior floor with σ8 ≈ 1.07; **small scales only [0.2–1.2]**
   lands at Ω_m ≈ 0.25 / σ8 ≈ 0.90; neither recovers fiducial. The
   large-scale (two-halo / projection) regime is the outlier under both
   covariances → **real model misspecification in the shear_prj regime, not
   a covariance artifact.**
2. **h-unit stopgap on the mock DV** (commit `6ce25b0`): mock stored ΔΣ in
   physical M⊙/pc² vs the model's h·M⊙/pc² — one power of h. Fixing the DV
   moved log Z from −2522 to −1335 and halved the Ω_m bias, but best-fit
   χ² is still 19.6/dof with a **coherent shear radial tilt (+10σ at small
   R → −13σ at large R)** and an **NC z-trend** (data/theory 0.55 at low z
   → 0.77 at high z).

So the residual misfit has two separable signatures:

- **(A) shear radial tilt, dominated at large R** → the shear_prj chain
  (b_sel wall → Σ_prj → ΔΣ_prj → γ_t) and, at small R, the 1-halo term.
- **(B) NC z-trend** → survey-geometry bookkeeping (Ω(z), z-bin edges,
  box-seam excision), not the lensing chain.

## 2. What is *excluded* as a major cause (and where that's shown)

These verdicts must be visible on the issues themselves (§6); the underlying
evidence is the reproducible report in
`validations/second_halo_term/report/` (branch
`claude/issue-4-dsigma-hh-2h-term`, `run_all.sh` regenerates everything):

| Component | Verdict | Evidence |
|---|---|---|
| ΔΣ_hh two-halo producer (issue #4) | **Fixed and validated — excluded.** | Post-fix Σ_hh matches the independent CLensPy FFTLog+Abel chain to 0.8%, ΔΣ_hh to 1.0%; 4-way P(k)→ξ bench (cluster_toolkit / CLensPy / pyccl / CLMM) agrees ≤1.6%; `test/halo_model.test.py` re-pinned 26/26. Also: the production max model composes `max(1h, b·ΔΣ_2h)`, so even the pre-fix NaN/z-degeneracy never reached the shear_prj branch. |
| ξ_NL table math (part of issue #9) | **Correct — excluded.** | Table agrees with a converged reference to 0.15%; consumers `p_operator_t.hh`, `sigma_prj_t.hh` audited clean (report §xi_NL). |
| Linear-vs-nonlinear P(k) input (issue #9) | **Accepted for now — not the large-scale railing.** | Measured with the Takahashi-2012 halofit reference at z=0.41: ΔΣ_prj shifts only −9% to +3% (b_sel absorbs most of the ξ change; the −36% b_sel small-scale plateau largely cancels in the ratio). Scale-dependent and signal-shaped, so it stays on the books — but an ≤9% effect cannot produce the Ω_m railing. **Owner decision 2026-08-20: live with linear P(k) for now**; the NL emulator path (`cp_camb` `nonlinear_pk_path`, Takahashi pinned in `outputs/pk_camb.npz`) is future work. |
| ΔΣ_prj algorithm identity | **Ports agree — but this is not method validation.** | C++/CUDA/Python backends agree to ~1e-8 (same algorithm); frozen-physics vs exact-z evaluator < 0.2%. What is *missing* is an independent method-level reference — that gap is issue #11 and is exactly where a shear_prj method bug would hide. |

**Conclusion for the campaign:** the large-scale misfit is *not* explained by
any already-diagnosed defect. Suspicion falls on the parts of the shear_prj
chain that have **never been independently validated** (issue #11) plus
unit/convention mismatches between model and mock (the h-unit episode proves
this failure mode is live).

## 3. Prime suspects, ranked

1. **shear_prj method-level error** — grid/kernel/convention bug in
   `sp_detail::ShearPrjCore` or the b_sel_marg wall (P1/I1/J). Zero
   independent validation today (issue #11, blocked by #10). A wrong θ_max,
   Δχ convention, or h-factor here is invisible to every existing test.
2. **Unit/convention mismatch model↔mock at large R** — comoving vs physical
   R, h powers in ⟨Σ_crit⁻¹⟩, γ_t vs ΔΣ. The mock-side one-power-of-h bug
   was already real; the model side has documented traps (chi in Mpc vs
   R_λ in cMpc/h).
3. **1-halo term at z=0** (issue #3) — production `Shear1hMisSel` reads
   `haloModel/dSigma_nfw`, which is the z=0 profile at every z;
   concentration is 5–15% high for z=0.2–0.65. Wrong radial *shape* →
   contributes to the small-R end of the tilt.
4. **NC z-trend** — z-bin seam + Ω(z) (issues #8, nersc#1, nersc#2). Fully
   explains signature (B); does not touch shear.
5. **HOD normalization** (issue #2) — up to +19% density non-normalization
   at μ_sat < 2 feeds S_i(M,z) in the live mass range; affects NC and the
   selection weighting of everything. Needs quantification before verdict.

## 4. Issue-by-issue triage

| Issue | Verdict / action | Phase |
|---|---|---|
| y3 **#4** ΔΣ_hh | Fixed on `claude/issue-4-dsigma-hh-2h-term`. **Action: open PR, merge, close.** | 0 |
| y3 **#10** wall-metadata contract | **Done** (`2210b5c`, revised per owner review — no extra module): `cosmology/bsel.py` itself reads the wall geometry from the `[b_sel_marg]` ini section at setup and republishes it into `b_sel_marg_P1/_I1/_J` at execute. `real_pipeline_extract_prj2h.ini` regenerates the projection dump sections. | 1 |
| y3 **#11** wall + projection validation | **Core of the campaign.** Build the dump-fed quad-truth wall reference (re-point RichnessSelection `frozen_bsel_validation.quad_truth` / `sel_bias._P_operator` at the dump's ξ_NL/HMF/bias/selection tables) + an independent adaptive Python reference for Σ_prj/ΔΣ_prj/shear_prj. Pin both into unit tests (§5). | 2 |
| y3 **#9** linear ξ_NL | **Re-scope: accepted limitation.** Comment the decision + measured impact; keep open, deprioritized; NL emulator later via `cp_camb nonlinear_pk_path`. | 1 (doc) |
| y3 **#3** 1h term z=0 | **Fix.** Publish per-z 1-halo tables (3-D table or per-z-bin sections — schema decision at implementation); re-pin `TestFirstHaloTermRedshiftHandling`. Directly improves the small-R shear model. | 3 |
| y3 **#8** Ω(z) hardcoded + no z-excision | **Fix.** (a) ini-configurable Ω(z) (table or `survey_area`); (b) optional true-z exclusion range for the box seam. Code-side root of the NC z-trend. | 3 |
| nersc **#2** z-bin edges/seam | **Premise corrected (owner review):** the DV harvester bins z_obs at the base edges 0.20/0.35/0.50/0.65 and drops the seam as a *true-z* cut (`build_buzzard_datavector.py:47-48,106`) — the model's observed-z edges were already right; the ini-edge override was reverted (`bb4140a`). The real fix is the true-z exclusion range in the selection integrand → rides entirely on y3#8. | 3 (via #8) |
| nersc **#1** Ω(z) validation | Confirm `OMEGA_Z_DES(z)` equals what the mock's per-halo `w_area` rescaling used; decide in-integral vs bin-averaged ⟨Ω⟩. Pairs with y3#8. | 3 |
| y3 **#2** HOD normalization | **Quantify first:** HMF-weighted bias of S_i(M,z) over the live mass range; then normalize-or-accept decision. Red tests stay red until then. | 3 |
| y3 **#5** radial_series CONC=4 | **Parked.** Candidate backend, not in production; cannot affect Buzzard fit. Resolve before any promotion. | — |
| y3 **#6** NFW_DSIGMA_MIS 0.5% point | **Parked (low).** Single point, −0.5%, production-irrelevant at current precision. Optional finer (r, r_mis) scan to classify. | — |

## 5. External-validation unit-test suite

Requested standing coverage: every stage checked against an *independent
external* reference (never a same-algorithm port), tolerances explicit,
linear P(k) assumed. Convention per CLAUDE.md: adaptive/closed-form
reference, project default tolerance 1e-3 unless a defect doc says
otherwise.

| Stage | Test (new/extended) | External reference | Gate |
|---|---|---|---|
| P(k) | `test/pk_external.test.py` (new) | Pinned CAMB linear (and Takahashi NL for the future gate) from `validations/second_halo_term/outputs/pk_camb.npz` — regenerable by `01_make_pk_camb.py` | interp error ≤ 1e-3 on the shared (k,z) grid |
| ξ(R) | `test/xi_nl_external.test.py` (new) | The 4-way transform bench (cluster_toolkit, CLensPy, pyccl, CLMM) reduced to pinned reference values | ≤ 1.6% cross-code envelope; ≤ 1e-3 vs the designated adaptive reference |
| ΔΣ 1h | exists: `test/nfw_dsigma_mis.test.cu` (cluster_toolkit) + `halo_model.test.py` closed-form NFW | cluster_toolkit + closed-form | 1e-3 (known single-point −0.5% stays pinned red per #6) |
| ΔΣ 2h (ΔΣ_hh) | extend `test/halo_model.test.py::TestSecondHaloTermVsClenspy` → pinned CLensPy post-fix values | CLensPy FFTLog+Abel | ≤ 1% (measured 0.8–1.0%) |
| b_sel(θ) / wall P1,I1,J | `test/bsel_external.test.py` (new) | dump-fed scipy quad-truth (adaptive, from RichnessSelection machinery re-pointed at the fiducial dump) | 1e-3 vs quad-truth |
| shear_prj (Σ_prj, ΔΣ_prj, γ_t) | extend `test/shear_prj_cross_backend.test.py` with an *external* leg | independent adaptive Python reference (full_ltmz-class, built for #11) | 1e-3 vs adaptive; existing cross-backend identity legs kept |

Notes:
- References get checked in as small pinned arrays with provenance comments
  (grid, ini, date, generator script) — same style the cross-backend test
  already uses.
- The bsel/shear_prj rows are **blocked by #10**: without the wall metadata
  the dump can't regenerate the projection sections.
- Accuracy is quoted against the adaptive reference, never against a
  production `.so` (per `src/pipelines/des_y3` testing rules).

## 6. Documentation pass across issues

Each open issue gets a comment stating its convergence verdict and pointing
at the evidence (this plan + the report). Specifically:

- **#4**: close-out comment (post-merge) — fixed, validated, excluded as a
  Buzzard-misfit cause.
- **#9**: decision comment — linear accepted for now, measured ΔΣ_prj impact
  −9%…+3%, not the railing; NL path documented for later.
- **#11**: elevated to campaign core; plan + blocked-by-#10 noted.
- **#10**: recommended resolution (permanent shim module) recorded.
- **#3**: production-relevance note (Shear1hMisSel consumes the z=0 tables →
  small-R tilt candidate; scheduled fix).
- **#8 / nersc#1 / nersc#2**: cross-linked as the NC z-trend cluster with
  the split (ini quick fix vs code hook vs validation).
- **#2**: quantify-first plan.
- **#5 / #6**: parked-with-reason notes.

## 7. Phases and gates

- **Phase 0 — land the 2h work.** PR `claude/issue-4-dsigma-hh-2h-term` →
  master; close #4. *(Push/PR on owner's go — commits are batched locally.)*
- **Phase 1 — quick wins + documentation.** nersc#2 ini z-edges; issue
  comment pass (§6); #10 shim-module fix.
- **Phase 2 — external references + tests.** #11: dump-fed quad-truth wall
  reference; independent adaptive shear_prj reference; the §5 test suite
  checked in and green (or deliberately red with a defect doc).
- **Phase 3 — model fixes.** #3 per-z 1h tables; #8 Ω(z) + z-excision;
  nersc#1 Ω(z) validation; #2 HOD quantification. Each lands with its
  re-pinned test.
- **Phase 4 — root-cause the tilt + refit.** With the external references
  in hand: audit units/conventions end-to-end (model γ_t vs mock ΔΣ:
  comoving/physical R, h powers, ⟨Σ_crit⁻¹⟩), fix whatever the quad-truth
  comparison exposes, rebuild the mock DV, rerun the scale-split polychord.
- **Convergence gate:** full-scale fit recovers fiducial (Ω_m, σ8) within
  ~1σ, χ²/dof ≈ 1, and the small/large scale splits agree with each other
  and with the full fit.
