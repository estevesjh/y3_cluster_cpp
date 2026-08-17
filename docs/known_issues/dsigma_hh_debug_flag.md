# ⚠ ΔΣ_hh (two-halo lensing term) needs debugging

**Raised 2026-08-12** while implementing the traditional 1h+2h max
model (`src/pipelines/des_y3/observables/shear_1h2h/fast_mass/python/shear1h2h_max.py`).
The `max(ΔΣ_1h, b·ΔΣ_2h)` observable is the first consumer to exercise
`haloModel/dSigma_hh` on the modern pipeline, and it exposed three
independent problems. **The traditional-shear arm cannot be trusted for
science until these are resolved**; the implementation is validated
against its own reference but inherits whatever `dSigma_hh` provides.

Producer: `y3_buzzard/haloModel.py::ct_2hTerm` (published by
`y3_buzzard/halo_model_cosmosis.py` under `compute_lensing_2h = T`).

## 1. 60% of the table is NaN — by explicit construction, not by failure

`ct_2hTerm.pk_to_dsigma` ends with

```python
self.dSigma = np.where(self.dSigma < 0., np.nan, self.dSigma)
```

i.e. every **negative** ΔΣ_2h is overwritten with NaN. Negative values
are the expected outcome of the exclusion subtraction at small radii
(`_to_dsigma` minus `deltaSigmaNFW_Analytical`), so the whole
small-radius half of the table is blanked: in the fiducial dump
**3850/6400 entries (60%) are NaN, everything below R ≈ 2.48 cMpc/h**.
Note `Sigma_hh` has **no** NaNs — only the ΔΣ path clips.

Consequences: any consumer must sanitize (the max model zero-fills,
which is exactly right for `max(1h, 0) = 1h`, but a *sum*-based 1h+2h
model would silently produce NaN everywhere through the mass integral).
The masking also hides whether the negatives are physical (exclusion
dominating) or a bug in the subtraction.

## 2. The z axis is degenerate — `Sigma_hh` is identical at every z

`pk_to_sigma` loops over `zvec` but calls `self._pk_to_sigma(Rp, k, pk)`
with **no z argument**, so the same (k, P_k) slice is used for every
redshift. Verified on the fiducial dump: all 50 z rows of `Sigma_hh`
are bit-identical across z ∈ [0, 4]. Whatever z-dependence the two-halo
term should have (growth of P_k_nl at minimum) is absent, and the
`dSigma_hh(R, z)` interpolation that the max model performs is
therefore interpolating a constant in z.

**Quantified against a genuinely independent, common-P(k) reference.**
The user's own CLensPy package (`clenspy.halo.twohalo.TwoHaloTerm`,
FFTLog + Abel projection — a wholly separate numerical pipeline from
cluster_toolkit's Hankel transform) was fed the *identical* real
fiducial P(k, z) used everywhere else in this doc, converted to
physical units (`k_phys = k_h * h`, `P_phys = P_h / h**3`, verified to
reproduce `cluster_toolkit.xi.xi_mm_at_r` to ~1% at several z first).
Its ξ(r, z) at a fixed comoving separation (r = 1 Mpc/h) varies by more
than **15×** between z=0 (ξ=5.58) and z=4 (ξ=0.36) — i.e. the true
two-halo term is very far from z-independent. Comparing production's
(z-invariant) `Sigma_hh` against CLensPy's Σ_2h at a representative DES
cluster redshift (z≈0.41):

| R [cMpc/h] | production `Sigma_hh` | true Σ_2h(z≈0.41) | deviation |
|---|---|---|---|
| 3.00 | 3.80 | 1.40 | **+172%** |
| 4.38 | 2.53 | 1.12 | **+126%** |
| 6.41 | 1.70 | 0.85 | **+99%** |
| 9.36 | 1.12 | 0.61 | **+84%** |
| 13.69 | 0.70 | 0.40 | **+76%** |
| 20.00 | 0.41 | 0.24 | **+71%** |

(units: Msun h/pc², comoving.) Production's fixed Sigma_hh overshoots
the true z≈0.4 two-halo term by 71–172% across this radial range —
substantial, not a rounding-level artifact. Pinned by
`test/halo_model.test.py::TestSecondHaloTermVsClenspy` (hard-coded
reference values, generated once from CLensPy — see that class's
docstring for the regeneration recipe; the test file itself carries no
runtime dependency on CLensPy).

**Caveat: this measurement uses linear P(k), because that is all this
fixture has.** `docs/figs/real_pipeline_extract.ini`'s `[cp_camb]`
section sets no `nonlinear_pk_path`, so `matter_power_nl` is never
written for this dump, and `halo_model_cosmosis.py::execute()`'s own
fallback (`if block.has_section("matter_power_nl"): ... else: P_k_nl,
k_nl, z_nl = P_k, k_h, z_k`) means the "nl" P(k) `second_halo_term`
receives here is just the linear one under another name. Both
"production" and the CLensPy reference above use that same linear
P(k) — internally consistent, and correct for this fixture — but a
production run with `nonlinear_pk_path` actually configured would feed
`second_halo_term` a genuinely nonlinear P(k), and the deviation
percentages would differ from the table above (the z-degeneracy bug
itself is a z-loop bug, not a P(k)-choice artifact, so it is present
either way).

## 3. Dummy halo parameters inside the exclusion terms

`ct_2hTerm.__init__` defaults to `Md=1e13, cd=4` and
`halo_model_cosmosis.py` instantiates it as `Md=1e14, cd=5, bias=1.0`;
`_to_dsigma` then passes `self.Md/10.`. These dummy mass/concentration
values enter both the added `sigmaNFW_Analytical` and the subtracted
`deltaSigmaNFW_Analytical`, so the exclusion correction is tied to a
fixed halo rather than the mass being integrated over — inconsistent
with a mass-resolved observable, and a plausible source of the
negatives in (1).

## Suggested debugging order

1. Remove the `np.nan` clip and inspect the raw negatives: are they the
   physical exclusion regime, or is the added/subtracted NFW pair
   mismatched (note the commented-out `0.993 *` factor in the same
   function, evidence of past tuning)?
2. Fix the z loop to pass the per-z power spectrum; re-check that
   `Sigma_hh`/`dSigma_hh` then vary with z as expected.
3. Decide the exclusion convention: the dummy `Md`/`cd` should either
   become the actual halo mass (making ΔΣ_2h mass-dependent) or the
   exclusion should be applied at the observable level, not inside the
   table.
4. Re-validate the max model afterwards
   (`validate_shear1h2h_max.py <dump>`) — its adaptive reference and
   the 2h→0 limit check are already in place and will quantify the
   change.

Related: `docs/source/observables/second_halo_term.md` documents the
max model and the dummy-halo recipe.
