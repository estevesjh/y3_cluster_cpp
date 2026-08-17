# ⚠ NFW_DSIGMA_MIS: one tabulated point misses the cluster_toolkit reference by ~0.5%

**Raised 2026-08-14** while reviewing `test/nfw_dsigma_mis.test.cu`
("Test NFW Misc Implementation"), an existing test comparing the
miscentered-NFW ΔΣ device/host model (`y3_cluster::NFW_DSIGMA_MIS`,
`src/models/nfw_dsigma_mis.hh`/`.cuh`) against genuinely independent
`cluster_toolkit`-generated reference values (see the file's own
generation recipe in its trailing comment block:
`ct.miscentering.DeltaSigma_mis_at_R`). The test's tolerance was set to
`epsrel = 5.0e-3` at authoring time — looser than this project's
standard 1e-3 — because one of the 30 tabulated points did not pass at
1e-3. Restoring the strict 1e-3 tolerance and re-running surfaces that
point directly rather than masking it.

Producer: `y3_cuda::NFW_DSIGMA_MIS::operator()`
(`src/models/nfw_dsigma_mis.cuh`, host twin
`src/models/nfw_dsigma_mis.hh`), which reads the pretabulated
`data/nfw_off_center/table_1000_1e-03_5e+03_gamma_*.txt` interpolation
tables.

## The measurement

30 points checked: 10 log-spaced radii `r` in [0.1, 10] Mpc/h × 3
miscentering offsets `r_mis` in {0.1, 0.5, 2.0} Mpc/h × 1 mass
(`lnM=32.2361913`, M≈1.0e14 Msun/h, c=4). **29/30 pass at strict 1e-3.**
The one failing point:

| r [Mpc/h] | r_mis [Mpc/h] | lnM | model | cluster_toolkit | deviation |
|---|---|---|---|---|---|
| 10.0 | 2.0 | 32.2361913 | 0.7821106 | 0.7860161 | **-0.497%** |

This is the largest radius and largest miscentering offset in the
tested grid, but **not a table-domain-edge artifact**: at this point
`x = r/r_s ≈ 35.5` and `x_mis = r_mis/r_s ≈ 7.1` (r_s computed at c=4
for this mass), both comfortably inside the table's tabulated domain
`x ∈ [1e-3, 5e3]` (log10 x ≈ 1.55 and 0.85, nowhere near either log10
edge at -3 or ~3.7). The other 9 points at `r_mis=2.0` (smaller `r`)
all pass at 1e-3, so the deviation is specific to this particular
(r, r_mis) combination, not a systematic bias across all large-r_mis
points.

Pinned by `test/nfw_dsigma_mis.test.cu`'s "Test NFW Misc Implementation"
(kept at the project's default 1e-3 tolerance rather than the looser
5e-3 it was authored with).

## Why this wasn't caught earlier

The test was authored with `epsrel = 5.0e-3` from the start (not
loosened in a later commit — git history shows no prior stricter
value), because this one point didn't pass at 1e-3 when the test was
written; the looser tolerance was never revisited.

## Open questions / suggested next steps

1. **Root cause not yet identified.** Candidates to check: (a)
   interpolation-grid resolution of the committed
   `nfw_off_center` table specifically in the `r/r_mis` region this
   point sits in (1000 grid points log-spaced over 6 decades may be
   coarser at this particular product of x/x_mis than elsewhere); (b)
   whether the `cluster_toolkit` reference itself used a different
   internal quadrature/grid resolution for this specific radius that
   introduces its own small error, making the "reference" not
   perfectly exact either; (c) a genuine bug in the gamma-kernel
   miscentering average specific to the r >> r_mis, r_mis >> r_s
   regime.
2. **Check whether this is isolated or the tip of a broader pattern.**
   Only 30 points were ever tested; a finer scan around
   (r=10, r_mis=2.0) — varying r_mis continuously, or adding
   intermediate r values — would show whether this is a genuinely
   isolated grid artifact or part of a smoother, broader deviation the
   current 30-point grid just barely resolves.
3. Re-run `test/nfw_dsigma_mis.test.cu` after either change and update
   the measured table above.

Related: `docs/known_issues/radial_series_vs_full_ltmz_defect.md` (the sibling
miscentered-NFW finding from the same review pass, a much larger
effect with a known cause); `docs/known_issues/dsigma_hh_debug_flag.md`,
`docs/known_issues/hod_normalization_defect.md`, `docs/known_issues/first_halo_term_z0_defect.md`
— the other known-defect writeups this one's structure mirrors.
