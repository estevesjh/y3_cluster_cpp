# Number counts — `full_ltmz` reference (C++ / Cuhre)

**Status: reference backend** (validated 2026-08-12 against the Python
`full_ltmz` reference and the production pipeline; numbers below). Not a
production entry point; production counts remain `sel_function` +
`NumCountsSel.so` (`fast_mass`). Built as `NumCountsFullLtmz.so`.

The same explicit triple integral as the
[Python backend](../python/README.md), evaluated as one adaptive-Cuhre
integral per bin instead of fixed Gauss–Legendre quadrature:

```text
N_ij = ∫dlt ∫dzt ∫dlnM  n(M,zt) dV/dΩdz(zt) Ω(zt)
                         K_j(zt) K_i(lt, zt) P_HOD(lt | M, zt)
```

Every physics term is an existing immutable model reused as-is:
`HMF_t`, `DV_DO_DZ_t`, `OMEGA_Z_DES`, `MOR_HOD_t`, `PlobLtrEMG_t` +
`RichnessKernel_t` (EMG K_i), `richness_zkernel` (K_j). The driver
instantiates the immutable `CosmoSISScalarIntegrationModule` template.
`PlobLtrEMG_t` needs the `plob_ltr_params` section — include the
`y3_buzzard/prj_params.py` shim in the pipeline (the Python backend's
built-in default is the same frozen table).

## Configuration

Wall-of-numbers, one entry per bin, zipped with `bin_index`:
`lam_min/lam_max/zob_min/zob_max/sigma_z` (bin definitions),
`lt_low/lt_high/zt_low/zt_high/lnm_low/lnm_high` (volumes), plus
`algorithm = cuhre`, `eps_rel`, `eps_abs`, `max_eval`,
`use_cartesian_product = F`.

Two numerically load-bearing choices, learned during validation:

- **`eps_rel = 1e-4`, not 1e-3.** The HOD is a near-delta ridge at
  `lt ≈ 1` for masses just above M_min (central-only halos whose
  projection tail feeds the lowest-richness bins). At `eps_rel = 1e-3`
  Cuhre *reports* convergence but under-integrates that ridge by ~1% in
  the λ ∈ [20, 30] bins; at `1e-4` all bins agree with the fixed-GL
  references to a few 1e-4. Cost: ~3.4 s per sample for 12 bins —
  acceptable for a reference (the plan explicitly exempts `full_ltmz`
  from production timing).
- **Per-bin `lt_high ≈ 4·lam_max`** (e.g. 120/180/240/800 for the Y3
  bins) rather than one generous global bound: shrinking the volume
  concentrates Cuhre's subdivision where the integrand lives.
  `lt_low` must be strictly positive (the EMG parameter family is
  singular at lt = 0); 0.1 is below any support that can reach
  λ_ob ≥ 20.

## Validation (2026-08-12, real extraction pipeline, fiducial point)

With `eps_rel = 1e-4` on the pinned 12-bin wall:

- vs the Python `full_ltmz` reference (fixed-GL, per-(M,z) λ brackets):
  **max |ratio − 1| = 4.9e-4** — two completely different quadrature
  strategies over the same physics;
- vs production `NumCountsSel.so` (`fast_mass`): max |ratio − 1| =
  1.1e-3, consistent with the production S_ij-tabulation error measured
  for the Python reference (7.6e-4) plus the Cuhre tolerance;
- all 12 Cuhre statuses converged (`status = 0`).

Output: `numcountsfullltmz/{vals, errors, probs, status, nregions}`.
