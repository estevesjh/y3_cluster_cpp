# Projection shear — `full_ltmz` (CUDA / PAGANI)

**Status: implemented; validation surfaced a real numerics finding —
convergence study of the wall-edge radii still open.** Built as
`DSigmaPrjFullLtmzGpu.so`; standard CUDA integration-module template,
every table on the device via `quad::Interp1D/Interp2D`; one adaptive
PAGANI triple integral per wall point over the continuous projection
integrand (exact z, ξ_NL frozen at z_ob, analytic b_sel plateaus, slab
exclusion on the clustered channel only — the same observable as the
fixed-GL fast_mass backends, integrated like the ShearPrjCuhre CPU
precedent). **The θ variable is integrated in log** — on a linear-θ
volume PAGANI silently returned 0 for ~20% of wall points *with all
statuses converged* (the ΔΣ_mis peak at θ_R = R/D_A is ~1e-5 wide for
the smallest radii).

## Validation record (2026-08-12, jobs 56795501 / 56796021, 1×A100)

- 95 s/sample at eps_rel 1e-3; 463 s at 1e-4 (180 wall points).
- vs the fixed-GL fast_mass reference at **production knobs**
  (n_per_seg=10, n_zring=n_zouter=20): median 2.7e-3, but up to 2.8%
  at the innermost and outermost radii — with PAGANI's *reported*
  errors ≤1e-3 (fooled by the exclusion discontinuity) AND the
  fixed-GL reference itself under-resolved:
- refining the reference (n_per_seg=40, n_z×3) moves it by up to
  **2.3%** — and at the outermost radii it lands on PAGANI to 2e-4
  (e.g. wall row 59: 7.483 → 7.656 vs GPU 7.658). Against the refined
  reference: median 9.5e-4, max 2.2% (innermost radii only).

Two open items, both recorded in the roadmap: (1) the innermost-radius
rows (R = 0.0426 cMpc/h, where θ_excl(z) sweeps through the ΔΣ_mis
peak) need a dedicated convergence study — neither method is
demonstrably converged there yet; (2) the finding that **production
knobs under-resolve the projection wall extremes at the ~2% level**
applies to the production ShearPrjFrozenPhysics settings too and
should be checked independently of this backend.

## Relation to the earlier GPU projection module (avgGammaProjBu)

The Buzzard-era `sigma_buzzard_y3/avgGammaProjBu.cu` integrates the
*previous* projection observable in 3–5 s because PAGANI only sees
three smooth selection dimensions (λ_ob, z_t, lnM): its two-halo term,
miscentring, bias and Σ_crit⁻¹ all enter as pre-tabulated device
interpolators (`DSIGMA_PROJ`). This module integrates the
Costanzi-2026 selection-affected observable, where θ must be an
explicit variable (b_sel(θ) and the per-z slab exclusion couple it to
z) and no offline table exists yet — hence the ~100× cost. The design
lesson is the plan's own: pre-tabulate the hard geometry (the fixed
tailored-θ contraction of the fast_mass backends; ultimately the
projection `radial_series` U-tables) and reserve adaptive integration
for smooth dimensions — this backend's role is the independent
cross-check, which is how it exposed the 2.3% wall-edge
under-resolution of the production-knob grids.
