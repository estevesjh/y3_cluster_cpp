# One-halo miscentred shear — `fast_mass` (C++)

**Status: reference re-expression** (validated 2026-08-12). Production
remains `Shear1hMisSel.so`. Built as `Shear1hFastMass.so`.

A thin des_y3 driver composing the immutable `SelGLCore` (exact z
contraction, Sigma_crit_inv folded in) with the production miscentred
mixture (haloModel `dSigma_nfw` + gamma-table `NFW_DSIGMA_MIS`) — the
`method = exact` algorithm of `Shear1hMisSelGL`, under the namespace
label `Shear1hFastMass` and the namespace output section
`shear1h_fast_mass/vals` (shared with the Python backend; never run
both in one pipeline — DataBlock sections do not overwrite).

Validation (real pipeline, fiducial point, 12 bins × 10 radii):
**bitwise identical to production `Shear1hMisSel.so`** in the same
run; 3.1e-15 vs the Python backend. Accuracy vs the adaptive
`full_ltmz` reference: 8.4e-4 (the S_ij tabulation, inherited by
construction). Cost: 9 ms/sample — identical to production.
