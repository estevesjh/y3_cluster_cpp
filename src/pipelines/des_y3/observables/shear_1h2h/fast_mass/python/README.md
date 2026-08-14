# One-halo miscentred shear — `fast_mass` (Python)

**Status: reference re-expression of the production algorithm**
(validated 2026-08-12). Production remains `Shear1hMisSel.so`
(`method = exact`); this module is the same computation stated in
importable Python.

The `fast_mass` strategy extended from counts to shear (plan owner's
direction, 2026-08-12): the z integral done exactly on fixed GL nodes
outside the radial operator —
`W_ij(lnM) = ∫dz n·dV/dΩdz·Ω·Σ_crit⁻¹·S_ij`, then
`O_ij(R) = ∫dlnM W_ij(lnM) Φ_i(R, lnM)` with the production miscentred
mixture `Φ = (1−f_mis)·ΔΣ_nfw + f_mis·ΔΣ_mis`.

Composed from the shared layer only: `MassZWeights` (SelGLCore twin)
and `lensing_profiles.MisMixtureProfile` (interpolation-exact replicas
of the `haloModel/dSigma_nfw` bilinear table and the gamma-kernel
`NFW_DSIGMA_MIS` reader).

Validation (real extraction dump + in-pipeline smoke run, 12 bins × 10
radii), under the namespace accuracy policy (accuracy vs the
`full_ltmz` fiducial; production agreement is an identity check):

- **accuracy: 8.4e-4 from the fiducial** — the production S_ij
  tabulation error, inherited by construction;
- algorithm identity: 3.1e-15 vs `Shear1hMisSel.so`; any future drift
  here means a replica no longer matches the production conventions.
  Cost: 74 ms/sample (pure Python; the .so does the same sums in 9 ms).

DataBlock contract: see the module docstring (`shear1h_fast_mass.py`);
output `shear1h_fast_mass/vals` (hardcoded section, bin slow / R fast).
