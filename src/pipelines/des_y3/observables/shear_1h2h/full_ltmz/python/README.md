# One-halo miscentred shear — `full_ltmz` reference (Python)

**Status: reference implementation** (validated 2026-08-12 against the
fast_mass backend and production). Production remains
`Shear1hMisSel.so`.

The full (λ_true, lnM, z) reference for the shear observable: every
selection kernel (shifted-Poisson HOD, EMG richness kernel, Gaussian
photo-z) evaluated at the quadrature nodes with no S_ij tabulation,
contracted against the production miscentred mixture Φ_i(R, lnM):

```text
O_ij(R) = ∫dz ∫dlnM ∫dλ_tr  n dV/dΩdz Ω Σ_crit⁻¹
                             K_j K_i P_HOD · Φ_i(R, lnM)
```

The (λ_true, z) contraction is the shared full_ltmz core
(`des_y3.shared.full_ltmz_core`, the same code the counts reference
uses, here with Σ_crit⁻¹ folded into the z factors); Φ comes from the
shared `lensing_profiles` production replicas. Because Φ is z-free
(fixed concentration and reference density), it commutes past the
(λ_true, z) integrals exactly — the study's "free win" — so the full
reference needs no extra quadrature dimension beyond the counts
reference.

## Validation (2026-08-12, real extraction dump, 12 bins × 10 radii)

- vs the `fast_mass` backend (same profile, same GL nodes; the only
  difference is direct kernel evaluation vs the production S_ij
  tabulation): **max |ratio − 1| = 8.4e-4** — the tabulation error,
  the same class the counts references measured (7.6e-4);
- vs production `Shear1hMisSel.so`: max |ratio − 1| = 8.4e-4 (the
  fast_mass backend sits at 3.1e-15 from the .so, so both comparisons
  see the same thing);
- in-pipeline smoke run: 149 ms/sample.

DataBlock contract: see the module docstring (`shear1h_full_ltmz.py`);
output `shear1h_full_ltmz/vals` (hardcoded section, bin slow / R fast).

Scope note: this is the reference for the *current maintained
observable* (miscentred 1h shear). The plan's "1h+2h" full reference
additionally includes the two-halo term, which in this pipeline is
carried by the projection stage (`shear_prj`); that belongs to the
shear_projection observable's implementations, not here.
