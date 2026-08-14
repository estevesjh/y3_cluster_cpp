# One-halo miscentred shear — `radial_series` (Python)

**Status: reference/candidate implementation** (validated 2026-08-11
against its own exact fixed-GL mass integral and the real pipeline
ingredients; see "Validation" below). Not a production entry point; the
production shear stage remains `Shear1hMisSel.so` (exact fixed-GL mass
sum). This is the first complete implementation of the approved plan's
`radial_series` strategy — offline U_ℓ tables, exact redshift
contraction, population moments — replacing the experimental runtime
`idea2` stencil approach.

## The three steps (approved plan, "radial_series")

1. **Offline (once, committed):** `generate_radial_series_tables.py`
   tabulates the unit-profile functions U₀…U₃ on dimensionless
   `(ln x, ln x_mis)` axes under [`data/radial_series/`](../../../../../../data/radial_series/README.md)
   for the fixed profile family of `nfw_profile_family.py` (centred
   W&B NFW c=4 + gamma-kernel miscentred, exactly the
   `nfw_dsigma_mis.hh` conventions). Never regenerated per sample.
2. **Per sample:** the exact redshift contraction
   `W_ij(lnM) = ∫dz n dV/dΩdz Ω S_ij Σ_crit⁻¹` on fixed GL nodes
   (`des_y3.shared.datablock_models.MassZWeights`, the SelGLCore
   replica — matches `NumCountsSel.so` to machine precision), then the
   population moments of y = ln r_s(M): N_ij, ȳ_ij, μ₂, μ₃. There is
   no redshift-freezing approximation.
3. **Per (bin, R):** interpolate U₀, U₂ (and U₃ if `ell_max=3`) at
   `(ln R − ȳ, ln r_mis − ȳ)`, restore the analytic amplitude, and
   assemble `O_ij(R) ≈ N_ij A0(ȳ) [u_mix,0 + μ₂ u_mix,2 + μ₃ u_mix,3]`
   with `u_mix,ℓ = (1−f_mis) U_ℓ^cen + f_mis Ω_m U_ℓ^mis`.

`ell_max` defaults to 2: both the radial-factorization study
(`docs/shear1h_radial_factorization.tex`) and this validation find the
μ₃ term does not improve the nearly symmetric DES Y3 mass weights;
ℓ = 3 stays supported as the plan's maximum retained order.

## Files

| File | Role |
|---|---|
| `nfw_profile_family.py` | fixed profile conventions: r_s(M), A0(y), W&B shapes (float64 + mpmath), source-table reader |
| `generate_radial_series_tables.py` | offline U_ℓ generator (first-principles, ~1e-9 smooth; see its docstring for why the noisy source table is not differentiated); also writes the text export the C++ backend reads |
| `shear1h_radial_series.py` | CosmoSIS module (setup/execute/cleanup) + `RadialSeriesTable` loader + `evaluate_series` |
| `validate_radial_series.py` | the four-check validation battery |
| `compare_backends.py` | Python-cubic vs C++-bilinear equivalence (measured 1.6e-4) |

A C++ backend with the same contract lives in [`../cpp/`](../cpp/README.md)
(`Shear1hRadialSeries.so`).

## DataBlock contract

Grid semantics match `Shear1hMisSel`: options `bin_index` × `r_perp`
cartesian product, bin slow / R fast; `lob_centers` (default
25 37.5 52.5 130) resolves the richness bin as `bin_index mod 4`;
required envelope `zt_low, zt_high, lnm_low, lnm_high`; knobs
`n_lnm` (96), `n_z` (64), `ell_max` (2), `table` (path to the npz).

Reads: `sel_function/{lnM,z,S_stack}`, `mass_function/{m_h,z,dndlnmh}`,
`cluster_abundance/{hmf_s,hmf_q}`, `distances/{z,d_a}`,
`average_sigma_crit_inv/{zlense,sci_average}`,
`miscentering/{f_mis,tau_mis}` (optional, defaults 0.22/0.17),
`cosmological_parameters/{h0,omega_m,omega_nu,omega_lambda,omega_k}`.

Writes (hardcoded section): `shear1h_radial_series/vals` (n_bins·n_R,),
plus per-bin diagnostics `norm, y_eff, mu2, mu3`.

## Validation (2026-08-11, real extraction dump)

`validate_radial_series.py`, replaying
`docs/figs/real_pipeline_extract.ini`:

1. centred U_ℓ (table + runtime interpolation) vs mpmath Taylor:
   ≤ 1.1e-8;
2. miscentred U₁…U₃ vs an independent wide-window Taylor fit of the
   committed U₀ surface: max ≤ 5e-5 (median ≤ 1e-8);
3. weight builder (f = 1) vs `NumCountsSel.so`: 2.4e-15; series
   truncation vs the exact fixed-GL mass integral of the same profile
   family, all 12 pinned bins on the production radial grid:
   **ℓ ≤ 2 max 0.450%** (tol 0.75%), ℓ ≤ 3 max 0.752% (tol 1.0%) —
   matching the study's real-pipeline Idea-2 numbers. Under the
   namespace accuracy policy (vs the doubled-node `full_ltmz` fiducial
   with this strategy's own fixed-convention profile), the **total**
   error — S_ij tabulation + truncation + interpolation — is
   **3.7e-3 max** over the same grid;
4. reported, not asserted: max shape deviation vs `Shear1hMisSel.so`
   is 5.2–10.1% per bin. This is the **disclosed centred-profile
   convention gap**, not a series error: this family uses the fixed
   c = 4 W&B centred term (the miscentring table's own centred limit),
   while production interpolates the per-sample `haloModel/dSigma_nfw`
   Python table (different concentration/normalisation conventions; see
   `validations/README.md` and the study §"two independently-sourced
   tables"). The per-sample haloModel table changes its dimensionless
   shape with the sample, so under the approved plan's reuse rule it
   cannot silently share this offline table; aligning the centred
   conventions is a separate, scoped decision before any production
   promotion.

Cost: 6–7 ms per sample for 12 bins × 10 radii in pure Python
(measured in the extraction pipeline; `Shear1hMisSel.so`'s exact GL
mass sums took 9 ms on the same sample).
