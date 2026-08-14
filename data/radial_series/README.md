# `data/radial_series/` — offline unit-profile tables for `radial_series`

Versioned derived data required by the approved layout proposal
(`docs/module_reorganization_plan.md`, "Offline unit-profile table"):
generated **once** by
`src/pipelines/des_y3/observables/shear_1h2h/radial_series/python/generate_radial_series_tables.py`,
committed here, loaded once at module construction, and **never
regenerated inside an MCMC sample**. Changing cosmology or HOD parameters
must not touch these files; everything sample-dependent enters through
the analytic amplitude (`A_sample`, `A0(y)`), the mixture weight
`f_mis`, and the query coordinates.

## `radial_series_nfw_mis_gamma_v1.npz` (+ `.json` metadata sidecar)

The ℓ = 0…3 unit-profile radial-series functions

```text
U_ell(ln x, ln x_mis) = (1 / ell! A0(y)) d^ell/dy^ell [ A0(y) u(R e^-y, r_mis e^-y) ]
```

for the fixed one-halo family of `src/models/nfw_dsigma_mis.hh`:
centred Wright & Brainerd NFW (c = 4, ρ_crit, 200c) and the gamma-kernel
(Gamma(2, x_mis·r_s)) miscentred profile. `A0(y) = 2 e^y δ_c ρ_crit
1e-12` (so `ΔΣ = A_sample · A0 · u` in M⊙/(h pc²)); `A0 ∝ e^y`, so the
tables need only the dimensionless `(ln x, ln x_mis)` axes.

Arrays: `lnx (619,)`, `lnxm (306,)`, `U{0..3}_mis (619, 306)` laid out
`[ix, ixm]`, `U{0..3}_cen (619,)`, `meta_json`. `U1` is retained for
validation only (its population coefficient μ₁ vanishes). Axis domains:
x ∈ [9.8e-4, 5.0e3], x_mis ∈ [9.8e-3, 20] — the x_mis domain is
deliberately smaller than the source table's (production usage is
x_mis ≲ 2); evaluators clamp at the edges, like `Interp2D::clamp`.

Recorded in the metadata (see the `.json`): profile conventions
(concentration, density convention, miscentering kernel, units, radial
domain), generator provenance, SHA-256 checksums of the
`data/nfw_off_center` source tables, the generator's numerical scheme,
and its self-check results.

## Why the values are *regenerated*, not differentiated from the table

The committed `data/nfw_off_center` gamma table carries ~1e-5…3e-3
point-to-point noise in ln u (measured 2026-08-11), which third-order
differentiation would amplify into O(1) garbage. The generator rebuilds
the same profile family from first principles at ~1e-9 smoothness and
differentiates that instead. Fidelity to the production table is then
*measured*, not assumed: over the physically relevant window
(x ∈ [0.05, 60], x_mis ∈ [0.03, 3]),

- `U0` vs source table: median |Δ ln u| = 1.7e-5, max 3.8e-4 (at the
  table's own noise level); spot checks against fully independent
  adaptive quadrature agree with both to ≤ 7e-7 where the table is clean;
- centred `U_ell` vs mpmath Taylor derivatives: ≤ 1.6e-10 (generation)
  and ≤ 1.1e-8 including runtime cubic interpolation;
- miscentred `U_1..U_3` vs an independent wide-window Taylor fit of the
  committed `U0` surface: median ≤ 1e-8, max ≤ 5e-5;
- ℓ ≤ 2 series truncation on the 12 real DES Y3 population weights:
  max 0.45% against the exact mass integral (ℓ ≤ 3: 0.75%).

Full numbers: `validate_radial_series.py` in the generator's directory.

## Versioning

New physics conventions (different concentration, kernel, or amplitude
separation) require a **new** versioned file and a new generator run —
never an in-place overwrite. A profile whose sample-dependent parameters
change its dimensionless shape cannot reuse this table at all (approved
plan, "Offline unit-profile table" reuse rule).
