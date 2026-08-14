# Projection shear — `fast_mass` (Python, exact z)

**Status: reference implementation** (validated 2026-08-12). Production
remains `shear_prj_frozen_physics` (`ShearPrjFrozenPhysics.so`).

The two-halo term sourced by correlated line-of-sight structure with
the selection-affected bias b_sel(θ), computed with the redshift
integral done **exactly** outside the radial operator — no
frozen-physics approximation. Per (λ_ob bin, z_ob) slice the exact
z-contraction builds `wrnd(M)` (mean-density channel) and `wcl(θ, M)`
(clustered channel, with ξ_NL, halo bias, and the per-z line-of-sight
slab exclusion resolved in z), then every wall radius is a θ×M dot
product against the single-offset NFW cache, weighted by the analytic
Costanzi-2026 b_sel(θ) sigmoid on the `b_sel_marginalised` plateaus.

This is a convention-exact port of the *exact* C++ core
(`sp_detail::ShearPrjCore`, `src/models/sigma_prj_t.hh`): identical θ
grid (per-slice breakpoints + log-GL segments), z grid (exclusion ring
+ log-|Δχ| wings with the 40-iteration χ inversion), parabolic photo-z
weight on the compiled `z_kernel_data.hh` σ_z table (parsed at import
by `des_y3.shared.z_kernel`), and no Ω(z) (it cancels in a surface
density). The production module freezes the clustered channel's mass
shape at z_ob; the exact treatment costs nothing here because the
(z, θ, M) contraction is a single einsum per slice.

## Validation (2026-08-12, real pipeline at the fiducial point, 180-point wall)

- vs the exact `DSigmaPrjEvaluator.so` (`[dsigma_prj]`, same knobs):
  **rnd 2.9e-14, cl 2.2e-11, total 1.6e-11** — machine precision;
- vs the frozen production module: total **5.5e-5** — the measured
  frozen-physics approximation at the fiducial point (documented bound
  < 0.2%);
- cost: **270 ms**/sample for 180 wall points (12 slices), pure Python
  — comparable to the exact C++ evaluator (~230–250 ms) and 3.3× the
  frozen production module (~82 ms).

The validation pipeline: the production extraction chain +
`b_sel_marg` → `bsel` → `[dsigma_prj]` (exact) →
`[shear_prj_frozen_physics]` (frozen) → this module, all on the pinned
wall with `zt 0.10–0.75`, `lnm 29.9336–35.6814`, `R_max_cMpch = 35`,
`n_lnm = 16`, `n_per_seg = 10`, `n_zring = n_zouter = 20`;
`validate_vs_production.py <dump_dir>` replays the comparison from the
saved dump.

## Contract

Options: the 180-point zipped wall (`lambda_bin/zo_low/zo_high/radii`),
the knobs above, `lob_centers` (default 25 37.5 52.5 130). DataBlock
reads and the two hardcoded output sections
(`dsigma_prj_fast_mass/{vals,rnd,cl}` in M⊙/(h pc²),
`shear_prj_fast_mass/{vals,rnd,cl}` dimensionless) are in the module
docstring. Note the production module additionally aliases its shear
triple to `shear_prj/*` for the likelihood; this reference deliberately
does not write that section.
