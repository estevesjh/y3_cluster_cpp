# ⚠ radial_series disagrees with full_ltmz by 56-86% in raw ΔΣ

**Raised 2026-08-14** while adding a cross-backend consistency test for
one-halo miscentred shear (`test/shear1h_cross_backend.test.py`). The
project's existing `radial_series` validator
(`validate_radial_series.py`) only ever reported a *shape-only*
comparison against production (normalized so both profiles equal 1 at
the first radius, "reported only, not asserted" for a documented
convention gap) — nobody had directly checked the raw ΔΣ values against
the full_ltmz reference until this pass.

Producer: `src/pipelines/des_y3/shear_1h2h/0d/python/nfw_profile_family.py`,
line 36: `CONC = 4.0` — a hardcoded module-level constant.

## The measurement

Comparing the C++ `radial_series` backend (`Shear1hRadialSeries.so`)
against the C++ `full_ltmz` reference (`Shear1hFullLtmz.so`) on the
pinned 12-bin × 10-radius wall, raw ΔΣ (not shape-normalized):

| Richness bin | max \|ratio − 1\| | at innermost R |
|---|---|---|
| 0 (λ∈[20,30]) | 56-58% | 56-58% |
| 1 (λ∈[30,45]) | 65-67% | 65-67% |
| 2 (λ∈[45,60]) | 73-75% | 73-75% |
| 3 (λ∈[60,200]) | **82-86%** | 82-86% |

(z-bin index — 0/1/2 within each richness group — makes only a small,
secondary difference; richness bin, i.e. mass, is what drives it.) This
is **not just a shape/curvature effect** — the deviation is already
56-86% at the very first (innermost) radius, i.e. a genuine amplitude
offset, not merely a difference in how the profile falls off with R.
Once that amplitude offset is normalized out (dividing each profile by
its own value at R[0], matching what `validate_radial_series.py`'s
"check 4" already does), a **further ~10%** shape/curvature residual
remains — this second number is what the project's own validator has
long reported, but the ~56-86% raw-amplitude offset underneath it had
not been quantified before.

## Why

`nfw_profile_family.py` hardcodes `CONC = 4.0` as a single module-level
constant, used for **both** halves of the profile family:

- `y_of_lnM(lnM)` — computes `r_s(M)` at the fixed `CONC`, not the
  halo's real concentration.
- `A0_of_y(y)` / `DELTA_C` — the profile's amplitude normalization,
  also built from the fixed `CONC`.
- `u_cen(x)` — the shape function itself, whose own docstring says
  "Centred unit profile: A0(y) u_cen = DSigma_NFW **at fixed c=4**."

`full_ltmz`, `fast_mass`, and production instead read the real,
per-sample concentration from `haloModel/dSigma_nfw` — the actual
Child18 mass-concentration relation (this session separately measured
it at roughly c≈3.6-5.7 across the relevant mass range, decreasing with
mass). Since `r_s = r_200/c`, using a fixed c=4 instead of the true,
mass-dependent concentration shifts both the overall ΔΣ amplitude and
where a given physical radius `R` lands on the shape function
`u(R/r_s)` — and the offset grows with mass exactly because real
concentration moves further from 4 as mass increases, matching the
observed richness-bin trend above.

One deliberately-red test pins this (kept at the project's default
1e-3 tolerance — see `test/shear1h_cross_backend.test.py`):

- `TestShear1hCrossBackend.test_cpp_radial_series_matches_cpp_full_ltmz`

This is separate from, and does not replace,
`test_python_radial_series_matches_cpp_radial_series` (C++ vs Python
radial_series identity, ~1.6e-4) — that test isolates a genuinely
different question (do the two language implementations of the *same*
fixed-c=4 family agree), which they do.

## Why this wasn't caught earlier

`radial_series` is documented as a "candidate moment-expansion
implementation," not a production or reference backend
(`docs/source/pipeline_organization.md`), and its own validator's
production comparison was deliberately "reported only, not asserted" —
appropriate for tracking a known modeling tradeoff, but it meant no
CTest target ever failed loudly when the raw amplitude (as opposed to
shape) diverged this much.

## Open questions / suggested next steps

1. **Decide whether `CONC = 4.0` should track the real per-sample
   concentration.** Doing so would need per-(M,z) U_ℓ tables (or a
   concentration-dependent correction term), which may defeat the
   strategy's whole point (offline tables computed once, reused for
   every sample) — or a documented, accepted amplitude-correction
   factor could be applied instead if the shape residual (~10%) is
   judged acceptable on its own.
2. **Quantify whether this matters for `radial_series`'s intended use.**
   If it's meant only as a fast shape/curvature approximation with
   normalization applied downstream, the raw-amplitude mismatch may be
   irrelevant in practice — but that intent isn't currently documented
   anywhere, and this test now assumes raw ΔΣ should match unless told
   otherwise.
3. Re-run `test/shear1h_cross_backend.test.py` after either change and
   update the measured-deviation table above.

Related: `docs/known_issues/dsigma_hh_debug_flag.md`,
`docs/known_issues/hod_normalization_defect.md`,
`docs/known_issues/first_halo_term_z0_defect.md`, the sibling known-defect writeups
this one's structure mirrors.
