# Number counts — `fast_mass` (Python)

**Status: reference re-expression of the production algorithm**
(validated 2026-08-12). Production remains `NumCountsSel.so`; this
module is the same computation stated in importable Python.

The `fast_mass` strategy: the redshift integral is performed exactly,
on fixed GL nodes, *outside* the mass operator —
`W_ij(lnM) = ∫dz n·dV/dΩdz·Ω·S_ij`, then `N_ij = ∫dlnM W_ij` (f = 1).
Implemented entirely by the shared SelGLCore replica
(`des_y3.shared.datablock_models.MassZWeights`) — the same object the
`radial_series` moments and every namespace validator build on.

Validation (real extraction dump + in-pipeline smoke run, 12 pinned
bins), under the namespace accuracy policy (accuracy vs the `full_ltmz`
fiducial; production agreement is an identity check):

- **accuracy: 7.6e-4 from the fiducial** — the production S_ij
  tabulation error, which this algorithm inherits by construction;
- algorithm identity: 2.4e-15 vs `NumCountsSel.so` (node placement,
  S_stack bilinear interpolation, and term composition replicated
  exactly). Cost: 5 ms/sample.

DataBlock contract: see the module docstring
(`numcounts_fast_mass.py`); output `numcounts_fast_mass/vals`
(hardcoded section).
