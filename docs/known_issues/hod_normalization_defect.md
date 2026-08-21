# ⚠ Shifted-Poisson HOD density is not normalized at low occupation

**Raised 2026-08-13** while writing the first dedicated unit tests for
`src/pipelines/shared/sel_function.py` (previously untested; see
`test/sel_function.test.py`). The continuous shifted-Poisson density
`P(lambda_tr | M, z)` — `sel_function._p_hod_scalar`, matching
`src/models/mor_hod_t.hh`'s `MOR_HOD_t` line for line — does not
integrate to 1 over `lambda_tr` once the mean satellite occupation
`mu_sat` drops below roughly 2. **The richness selection function
`S_i(M, z)` inherits whatever this density provides at every mass**, so
this is live in the actual DES Y3 integration range, not an edge case.

Producer: `sel_function.py::_p_hod_scalar` /
`src/models/mor_hod_t.hh::MOR_HOD_t` (both instantiate the same closed
form, `docs/source/science/index.md`'s "Shifted-Poisson HOD model"
section, doc Eq. 28).

## The measurement

At the fiducial-ish HOD point `log10_Mmin=13.8, log10_M1=14.5,
alpha=1.1, epsilon=-0.2, sigma_lambda=0.35` and `z=0.30`, integrating the
density (`scipy.integrate.quad`, an independent re-derivation via
`scipy.special.gammaln` — confirmed to match `_p_hod_scalar` itself to
1e-3 relative, so this is not a re-derivation bug) over `lambda_tr` from 0
to a large upper bound:

| Mass | `mu_sat` | `integral` | deviation |
|---|---|---|---|
| 1.5e14 | 0.32 | 1.193 | **+19.3%** |
| 2.0e14 | 0.52 | 1.093 | **+9.3%** |
| 3.0e14 | 0.95 | 1.027 | **+2.7%** |
| 5.0e14 | 1.86 | 0.998 | −0.2% |
| 1.0e15 | 4.31 | 0.992 | −0.8% |
| 3.0e15 | 15.15 | 0.995 | −0.5% |

Deviation shrinks monotonically with `mu_sat` and is negligible (<1%)
once `mu_sat` exceeds roughly 2, but is large for `mu_sat` of order 1 or
below — exactly the regime a DES Y3 `lnm_low ~ ln(1e13)` mass integral
passes through on its way up from the low-mass end of the halo mass
function.

Three failing tests pin this (kept at the project's default 1e-3
tolerance deliberately — see `test/sel_function.test.py`):

- `TestShiftedPoissonHod.test_normalizes_to_unity_over_ltr`
- `TestShiftedPoissonHod.test_gl_bracket_quadrature_also_normalizes_to_unity`
  (the production Gauss-Legendre bracket/nodes reproduce the same
  non-normalization, so it is not a quadrature artifact)
- `TestRichnessSelectionFunction.test_s_i_matches_independent_double_integral`
  (the defect propagates into the actual `S_i(M, z)` assembly at a
  low-`mu_sat` mass point, not just the raw HOD integral)

`TestShiftedPoissonHod.test_low_occupation_normalization_defect_is_characterized`
passes and exists specifically to keep the measured range on record, so a
future change to the HOD model shows up as a deliberate diff there rather
than a silent regression either direction.

## Why this wasn't caught earlier

`sel_function.py` (and its C++ sibling `mor_hod_t.hh`) had no dedicated
unit tests before this file — every consumer (`numcounts_fast_mass`,
`shear1h_fast_mass`, ...) only ever exercised it indirectly, through its
own composed output, which never isolated the HOD density's own
normalization.

## Open questions / suggested next steps

1. **Quantify the practical impact.** The observed-richness kernel
   `S_i(lambda_tr, z)` (the bin-integrated selection window) suppresses
   contributions far from the richness bin's support — for a bin like
   `[20, 30]`, a halo with `mu_sat ~ 0.3` (so `lambda_tr` sits near 1, the
   central-only value) contributes essentially nothing to that bin
   *regardless* of whether `P_HOD` is exactly normalized. Whether the
   normalization defect meaningfully biases `S_i(M, z)` — and therefore
   `numcountssel`/`shear1hmissel` — likely depends on how much of the
   `dn/dlnM`-weighted mass integral sits in the `mu_sat < 2` range for the
   real DES Y3 halo mass function; that weighting has not been checked
   here.
2. **Decide whether the model needs a normalization correction** (e.g. a
   mass-dependent renormalization factor) or whether the deviation is an
   accepted property of the shifted-Poisson approximation at low
   occupation, in which case the three tests above should get an explicit,
   documented tolerance (referencing this file and the GitHub issue) instead
   of staying red.
3. Re-run `test/sel_function.test.py` after either change and update the
   measured-deviation table above.

Related: `docs/source/science/index.md` "Selection functions" chapter
(the closed-form derivation this file's tests check against);
`docs/known_issues/dsigma_hh_debug_flag.md` (the sibling known-defect writeup this
one's structure mirrors).
