# ⚠ The 1-halo lensing term is always evaluated at z=0, not the cluster's z

**Raised 2026-08-13** while writing the first dedicated, genuinely
independent unit tests for `y3_buzzard/haloModel.py` /
`halo_model_cosmosis.py` (see `test/halo_model.test.py`). An earlier pass
mistakenly flagged `child18_mass_concentration`'s `Mstar` formula as a
bug relative to `colossus`; that was wrong and has been retracted (see
below) — `Mstar = 10**(12.5-1.5*z)` is Child et al. (2018)'s own
published fit (arXiv:1804.10199 give `log(M*/h^-1 Msun)` = 12.5, 11, 9.5
at z=0, 1, 2 — exactly this formula), not a generic nonlinear-mass
approximation, so disagreeing with `colossus`'s
`peaks.nonLinearMass(z)` substitute is expected, not evidence of an
error. The real finding is upstream of that entirely.

Producer: `y3_buzzard/halo_model_cosmosis.py::execute`, the line

```python
if compute_lensing_1h:
    lensModel.first_halo_term(M, z=0, conc_model_name="Child18")
```

## The measurement

`execute()` builds a real per-z grid (`z = z_k`, the `matter_power_lin`
z-axis) and uses it for the bias table, `xi_NL(r, z)`, `second_halo_term`,
and `scaleShiftCosmo` — every other z-dependent quantity in the module.
The 1-halo term is the one exception: it is always called with the
literal `z=0`, regardless of what `z` actually contains. Consequently
`haloModel/Sigma_nfw`, `haloModel/dSigma_nfw`, and
`haloModel/concentration` are published as 2D tables over `(r_sigma,
lnM)` only — there is no `z` axis for the 1-halo term in the datablock
at all, and every C++ consumer that interpolates them
(`lensing_weights.hh`, `kappa_max.hh`/`.cuh`, `gamma_max.hh`/`.cuh`,
`sigma_mis_joint.cuh`, `n_operator_sel_gl_t.hh`, ...) queries them with
no z dependence.

`first_halo_term`/`child18_mass_concentration` are themselves correctly
z-dependent when called directly — this is purely a wiring gap in
`execute()`, not a bug in the model:

| z | concentration (M=1e14 Msun/h) | vs z=0 |
|---|---|---|
| 0.00 | 4.689 | — |
| 0.20 | 4.437 | -5.4% |
| 0.40 | 4.210 | -10.2% |
| 0.65 | 3.974 | -15.2% |

At a typical DES cluster redshift (z~0.3-0.6) concentration is
5-15% lower than the z=0 value production always uses. Since
`rs = r200/c`, this is not just a normalization shift — it changes the
radial shape of the projected NFW profile itself (in this repo's own
comparison, central Sigma differed by tens of percent between the z=0
and z=0.4 profiles at fixed M).

Two tests pin this in `test/halo_model.test.py`:

- `TestFirstHaloTermRedshiftHandling.test_first_halo_term_concentration_is_genuinely_z_dependent`
  (passes — confirms the underlying model responds correctly to z)
- `TestFirstHaloTermRedshiftHandling.test_execute_hardcodes_z_equals_0_for_the_1h_term`
  (passes — pins the hardcoded `z=0` call site; meant to fail loudly,
  and need a deliberate update, the day this is fixed)

## Why this wasn't caught earlier

No prior test exercised `halo_model_cosmosis.py::execute()`'s actual
wiring at all — only `haloModel.py`'s classes were reachable without a
full CosmoSIS `DataBlock`. The z=0 hardcoding is invisible from `haloModel.py`
alone since `first_halo_term` itself works correctly for any z passed to it.

## Open questions / suggested next steps

1. **Decide whether this was deliberate.** The 1-halo term's datablock
   schema has no z axis at all (2D `(r_sigma, lnM)` tables only), so
   `z=0` may have been a placeholder chosen when the schema was designed
   under the (possibly reasonable at the time) assumption that 1-halo
   shear varies weakly with z across the DES cluster sample — or it may
   be an oversight. Check whether the DES Y3 cluster sample's actual z
   range (and the precision the 1-halo shear term is used at downstream)
   makes the 5-15% concentration shift material.
2. **If real, the fix needs a schema change**, not just passing the
   right `z` into `first_halo_term`: `Sigma_nfw`/`dSigma_nfw` would need
   a z axis (3D table or per-z-bin sections), and every C++ consumer's
   `Interp2D(..., "Sigma_nfw")` call would need to become an `Interp3D`
   or per-z lookup. That is a bigger change than this doc's scope; this
   file only establishes that the current z=0 evaluation is real and
   quantifies its size.
3. Re-run `test/halo_model.test.py` after either change and update the
   measured-deviation table above.

Related: `docs/dsigma_hh_debug_flag.md` and
`docs/hod_normalization_defect.md`, the sibling known-defect writeups
this one's structure mirrors; `test/halo_model.test.py::TestBiasModel`
and `::TestFirstHaloTerm`, which now pass against genuinely independent
references (`cluster_toolkit.bias.bias_at_nu`,
`cluster_toolkit.deltasigma`) once compared apples-to-apples in a single
Msun/h, Mpc/h, Msun h/pc² convention.
