# DES Y3 observable implementations

This directory contains the maintained DES Y3 implementations for cluster
number counts, one-halo lensing, traditional one-plus-two-halo lensing, and
selection-affected projection lensing.

The directory is organized as:

```text
observable / numerical strategy / language or backend
```

Strategy folders count **adaptive integration dimensions** — the quantity
that drives per-sample computing time. `0d` means no adaptive integration at
all (fixed Gauss--Legendre sums and offline tables — fast, MCMC-viable);
`Nd` means N adaptive (Cuhre/Vegas/PAGANI) dimensions. The
maximum-dimension folder of each observable is always the adaptive
reference, and every lower folder is a documented dimension reduction from
it. Code identifiers follow the same quadrature scheme: `SijGl` marks the
S_ij-tabulated fixed-GL fast paths (`NumCountsSijGl`, `Shear1hGl`,
`ShearPrjGl`), `explicit_gl` the explicit fixed-GL Python references
(`numcounts_explicit_gl.py`, `shear1h_explicit_gl.py`), and `3d` the
adaptive references (`NumCounts3d`, `Shear1h3d`, `Shear1h2hMax3d`,
`*3dGpu`). Module labels are the ini `[section]` names and the labels
double as DataBlock output sections (`numcounts_sij_gl/vals`,
`shear1h_gl/vals`), so the local inis were updated in the same commit;
the retired `fast_mass`/`full_ltmz` strings survive only in "formerly"
notes like the ones below.

The observable READMEs are the detailed documentation — each leads with
the physics (the integral being computed and what each dims tag
approximates), then the per-dims backend sections including the
Gauss--Legendre node placement, with the run mechanics last. This file
is the map of the directory and the short decision guide for choosing a
numerical method.

## Folder structure

The layout is `<observable>/<language>/<dims>/`, with one README per
observable carrying the physics, the per-dims sections, and the run
mechanics:

```text
src/pipelines/des_y3/
├── README.md
├── number_counts/
│   ├── README.md
│   ├── cpp/{0d, 3d}/     0d: S_ij-tab fast path (formerly fast_mass);
│   │                     3d: adaptive explicit Cuhre reference
│   ├── cuda/3d/          adaptive explicit PAGANI reference
│   └── python/0d/        fast path + explicit fixed-GL reference
│                         (formerly full_ltmz Python) + validators
├── shear_1h2h/
│   ├── README.md
│   ├── cpp/{0d, 3d}/     0d: 1h z-contracted + max z-resolved +
│   │                     radial series; 3d: adaptive explicit (1h, max)
│   ├── cuda/{0d, 3d}/    0d: max model GPU; 3d: adaptive explicit GPU
│   └── python/0d/        all fixed-GL replicas, explicit fixed-GL
│                         references, radial series + validators
└── shear_projection/
    ├── README.md
    ├── cpp/{0d, 2d}/     0d: region-split GL exact-z; 2d: ShearPrjCuhre
    │                     (fixed log-GL angle, adaptive (z, lnM))
    ├── cuda/{0d, 3d}/    0d: frozen-physics GPU; 3d: fully-coupled
    │                     adaptive PAGANI diagnostic
    └── python/0d/        exact-z reference + validator
```

`<dims>` counts the adaptive integration dimensions of the backends in
that directory. A missing `<language>/<dims>/` directory means that the
backend is not implemented.

The observable implementations use the sibling layers
[`../systematics/`](../systematics/README.md),
[`../cosmology/`](../cosmology/README.md), and `../shared/`. The systematics
layer owns richness selection, selection bias, projection parameters, and
the projection-shear C++ cores. The cosmology layer contains halo-model
physics; `shared/` contains lower-level datablock models, mass/redshift
weights, and common integration utilities.

## Observable entry points

Start with the observable README, then follow its links to the numerical
strategies and language backends.

| Observable | Physical quantity | Documentation |
| --- | --- | --- |
| Cluster number counts | Counts in observed richness and redshift bins | [`number_counts/`](number_counts/README.md) |
| One-halo shear | Miscentered target-cluster NFW lensing | [`shear_1h2h/`](shear_1h2h/README.md) |
| Traditional one-plus-two-halo model | Maximum of one-halo and conventional two-halo terms | [`shear_1h2h/`](shear_1h2h/README.md) |
| Projection shear | Lensing from correlated line-of-sight structure with selection-affected bias | [`shear_projection/`](shear_projection/README.md) |

## Numerical strategies

| Strategy tag | Numerical idea | Role |
| --- | --- | --- |
| `3d` | All three integration variables handled by adaptive Cuhre/Vegas/PAGANI quadrature. | Independent reference and convergence tool; the maximum-dimension adaptive reference of each observable. |
| `2d` (projection only) | Fixed log-GL angular grid; adaptive quadrature over the inner (z, lnM). | Adaptive comparison backend (`ShearPrjCuhre`). |
| `0d` | No adaptive integration: fixed Gauss--Legendre sums (with the redshift/selection contractions of the former `fast_mass` paths, or the full explicit composition on fixed nodes) and offline tables + moments (the former `radial_series`). | Maintained production methods; fast and MCMC-viable. |

The `3d` adaptive implementations define the accuracy baseline
("Precision vs 3d" in every table); fixed Gauss--Legendre backends are
evaluated against that baseline. Agreement with an existing production
module is a separate backend-identity check, not an accuracy measurement.

The radial-series `0d` algorithm is not an exact replacement for the
varying-concentration one-halo model: its current profile uses a fixed
concentration and is not scientifically interchangeable with the other
implementations.

## Precision and cost overview

The measurements below are pinned fiducial DES Y3 benchmarks, not universal
performance guarantees. Costs are per sample. The benchmark uses 12 count
bins, 12 one-halo bins with 10 radii each, and 180 projection wall points.
CPU timings depend on the Perlmutter node and build; GPU timings use an A100.

The precision column is quoted against the `3d` adaptive reference of each
observable ("Precision vs 3d"); where a number was measured against a
different baseline (production identity, backend twins, the exact
evaluator), the baseline is stated in the cell:

| Dims | Observable | Method and backend | Cost | Precision vs 3d |
| --- | --- | --- | ---: | --- |
| `3d` | Counts | [`3d`](number_counts/README.md#the-3d-backends), adaptive Python | 25 s | Reference (3d); reported integration error at or below 1e-6 |
| `3d` | Counts | [`3d`](number_counts/README.md#the-3d-backends), Cuhre C++ | 3.1 s | 1.1e-4 direct vs the 3d Python reference (re-measured 2026-08-26, same fiducial point as `real_pipeline_extract_output`; supersedes the older 4.9e-4 proxy-via-0d-Python figure) |
| `3d` | Counts | [`3d`](number_counts/README.md#the-3d-backends), PAGANI CUDA/A100 | 2.0 s | 1.2e-4 direct vs the 3d Python reference (re-measured 2026-08-26); CUDA vs C++ twin 2.1e-5 (separate baseline, shared A100) |
| `0d` | Counts | [`0d`](number_counts/README.md#explicit-fixed-gl-python-reference), explicit Python (3-dim GL) | 83 ms | 3.5e-5 |
| `0d` | Counts | [`0d`](number_counts/README.md#redshift-contracted-fast-path), fast path Python (2-dim GL, S_ij tab) | 5 ms | 7.6e-4; also 2.4e-15 vs production (separate baseline); 1.1e-3 direct vs cuda-3d (re-measured 2026-08-26, identical to the C++ row -- numerically identical output) |
| `0d` | Counts | [`0d`](number_counts/README.md#redshift-contracted-fast-path), fast path C++ (2-dim GL, S_ij tab) | 6 ms | 7.6e-4; also identity with production (separate baseline); 1.1e-3 direct vs cuda-3d (re-measured 2026-08-26) |
| `3d` | One-halo shear | [`3d`](shear_1h2h/README.md#the-3d-backends), adaptive Python | 35 s | Reference (3d); reported integration error at or below 1e-6 |
| `3d` | One-halo shear | [`3d`](shear_1h2h/README.md#the-3d-backends), Cuhre C++ | 51 s | 2.6e-4 direct vs the 3d Python reference (re-measured 2026-08-26 on the reduced 6-pt corner wall, same fiducial point) |
| `3d` | One-halo shear | [`3d`](shear_1h2h/README.md#the-3d-backends), PAGANI CUDA/A100 | 32 s | 3.0e-4 direct vs the 3d Python reference (re-measured 2026-08-26); CUDA vs C++ twin 4.3e-5 (separate baseline, shared A100) |
| `0d` | One-halo shear | [`0d`](shear_1h2h/README.md#the-0d-backends), explicit Python (3-dim GL) | 149 ms | 4.9e-5 |
| `0d` | One-halo shear | [`0d`](shear_1h2h/README.md#the-0d-backends), 1h C++ (1-dim GL, z contracted) | 9 ms | 8.4e-4; also identity with production (separate baseline); 2.3e-4 direct vs cuda-3d on the reduced 6-pt wall (re-measured 2026-08-26, python twin gave numerically identical values) |
| `0d` | One-halo shear | [`0d`](shear_1h2h/README.md#the-0d-backends), radial series (tables + moments) | 6--7 ms | 56--86% (known fixed-c=4 defect); 3.7e-3 internal fixed-profile consistency (separate baseline) |
| `0d` | Max model | [`0d`](shear_1h2h/README.md#the-0d-backends), max C++/CUDA (2-dim GL, z-resolved) | 11 / 8 ms | 8.3e-4; CUDA vs C++ twin 6.4e-15 (separate baseline) |
| `3d` | Projection shear | [`3d`](shear_projection/README.md#the-3d-backend), PAGANI on A100, eps_rel=1e-3 | 95 s | Reference-class diagnostic (3d); convergence open — median 9.5e-4, maximum 2.2% vs region-split GL (separate baseline) |
| `2d` | Projection shear | [`2d`](shear_projection/README.md#the-2d-backend), `ShearPrjCuhre` C++ | ~72 s/pt (measured 2026-08-26, 3-pt sample; full 180-pt wall ≈ 3.6 h, not run interactively) | not yet measured |
| `0d` | Projection shear | [`0d`](shear_projection/README.md#the-0d-backends), exact-z Python (3-dim region-split GL) | 270 ms | median 9.5e-4, max 2.2% vs the 3d diagnostic (its convergence is open); 1.6e-11 vs exact evaluator, 5.5e-5 vs frozen production (separate baselines) |
| `0d` | Projection shear | [`0d`](shear_projection/README.md#the-0d-backends), exact-z C++ (3-dim region-split GL) | 154 ms | median 9.5e-4, max 2.2% vs the 3d diagnostic (its convergence is open); 1e-11 vs exact evaluator (separate baseline) |
| `0d` | Projection shear | [`0d`](shear_projection/README.md#the-0d-backends), frozen GPU path | 16 ms (measured 2026-08-26, full 180-pt wall) | **FIXED, verified 2026-08-28 on Perlmutter GPU**: `vals`/`rnd`/`cl` all agree with the CPU frozen module to ~1e-10 (issue #24 closed) — [known defect writeup](../../../docs/known_issues/frozen_physics_signed_rnd_defect.md#resolved-2026-08-28-verified-on-perlmutter-gpu) |

The observable READMEs contain the complete backend tables, grid
settings (including the "GL nodes and weights" quadrature sections),
tolerances, and comparison definitions:

- [Number counts](number_counts/README.md)
- [One-halo and traditional one-plus-two-halo shear](shear_1h2h/README.md)
- [Projection shear](shear_projection/README.md)

## Cost distribution and robustness across the prior (apriori sampling)

The table above is single-point measurements at the fiducial
cosmology/HOD. To check that cost is stable and the pipeline doesn't
silently misbehave away from that one point,
`cosmosis-models/des_y3_cpp0d_fast_apriori.ini`,
`des_y3_python0d_fast_apriori.ini`, `des_y3_gpu0d_fast_apriori.ini`
(1000 draws each) and `des_y3_cpp3d_slow_reference_apriori.ini` (10
draws) run the same module chains under CosmoSIS's `apriori` sampler,
drawing independently from the full `mock_mcmc_widePlanck_values_mis.ini`
prior (`h0`, `omega_m`, `omega_b`, `n_s`, `sigma8`, and the five
`cluster_mor` HOD parameters) instead of evaluating once at the
fiducial point, with `timing = T` reporting module-by-module wall-clock
for every draw. `cp_camb`'s CAMB-emulator grid stays at `nz = 50` in
every variant -- raising it toward the production default of 400 makes
each sample much slower for no benefit to these fixed-node/adaptive-node
chains. Two independent runs of this sweep (unseeded `np.random.uniform`
draws, so each run samples different points) gave consistent findings,
below. Raw per-sample timing arrays are saved to
`cosmosis-models/des_y3_apriori_timing_data.json`.

**~30% of draws are free, near-zero-cost rejections, by design, not a
defect.** `cp_camb.py`'s pre-emulator bound-box check deliberately
rejects points outside the CAMB emulator's *trained* range even though
the values-file's declared prior is wider (e.g. the `n_s` prior is
$U(0.8, 1.15)$ but the trained box is only $[0.823, 1.103]$) --
extrapolating past the emulator's training region would produce
untrustworthy $P(k)$ that could otherwise crash
`cluster_toolkit.peak_height.nu_at_M` with a GSL abort further
downstream. `cosmosis` reports these as `Pipeline failed on these
parameters`, which reads alarming out of context but is the module
working as intended (fail fast and cheap, not silently on garbage
`P(k)`). Exclude these draws from any cost or precision statistics --
they never run past `cp_camb`.

**Finding 1 -- `sel_function`'s ~1.2-1.3 s fiducial cost is one-time
JIT compilation, not steady state.** Across ~700 completed fast draws
(cpp/python/gpu 0d chains all show the same pattern), `sel_function`
takes ~1.3 s on the *first* sample of the process and a steady
46-80 ms (median ~65 ms) on every subsequent one -- the known numba
`cache=False` issue (`sel_kernels` module-name cross-poisoning fix,
tracked separately). A real MCMC chain (thousands of samples per
process) amortizes this to zero; a single-fiducial-point benchmark
cannot see that and reports the inflated one-time number instead.

**Finding 2 -- the adaptive-3d modules' cost is strongly
cosmology-dependent (40-60x spread); `shear_prj_cuhre` is comparatively
stable.** `NumCounts3d`/`Shear1h3d`/`Shear1h2hMax3d` each run
0.7-0.9 s at most prior draws but spike to 26-47 s at a handful of
others -- some HOD/cosmology corners push the near-delta richness
ridge or the mass/redshift integrand into a shape Cuhre needs far more
subdivisions to resolve at `eps_rel=1e-4`. `shear_prj_cuhre`'s
per-point cost, by contrast, stays within about 10-15% across the same
draws, because its reduced 3-point wall is dominated by one expensive
angular integral rather than the adaptive mass/redshift integral. The
single fiducial-point costs quoted in the table above land inside this
spread but are not representative of either tail -- **anyone budgeting
a batch job around these backends should size the wall time on the
worst draw in the relevant prior region, not the fiducial-point
number.**

| Module | Fast 0d (cpp/python/gpu), ~700/3000 completed | Slow 3d/2d (cpp), ~5-8/10-19 completed |
| --- | --- | --- |
| `halo_model` | median ~450-470 ms (320-720 ms) | median ~400-490 ms (390-650 ms) |
| `sel_function` | median ~60-70 ms (46 ms-1.3 s incl. one-time JIT) | median ~55-65 ms (49 ms-1.2 s incl. one-time JIT) |
| `NumCountsSel`/`numcounts_sij_gl` vs `NumCounts3d` | median 5-8 ms (4-13 ms) | median 0.8-1.8 s (0.7-47.2 s) |
| `Shear1hMisSel`/`shear1h_gl` vs `Shear1h3d` | median 11-71 ms (9-91 ms) | median 0.9-1.5 s (0.7-26.1 s) |
| `Shear1h2hMax`/`shear1h2h_max`/`Shear1h2hMaxGpu` vs `Shear1h2hMax3d` | median 12-77 ms (9-139 ms) | median 0.9-1.7 s (0.7-29.3 s) |
| `ShearPrjGl`/`shear_prj_gl`/`ShearPrjFrozenGpu` vs `shear_prj_cuhre` | median 10 ms-590 ms (7 ms-865 ms) | median ~69-221 s/pt (68-222 s/pt) |
| Total pipeline | median 0.77-1.44 s (0.60-2.78 s) | median 224-251 s (204-325 s) |

**What this does and doesn't validate.** This confirms all three
languages' fast 0d chains and the slow adaptive-3d chain run cleanly
(no crashes, no NaN propagation) across hundreds of genuinely different
cosmology/HOD draws, and replaces the single-point cost numbers with
real distributions -- see `des_y3_apriori_timing_data.json` for the
raw per-sample arrays behind every number above. It does **not** give
a 0d-vs-3d precision comparison *across the prior*, since the fast and
slow apriori runs draw unrelated random points. At the single fiducial
point, that direct comparison *has* been done (see "Precision and cost
overview" above: python-3d vs cpp-3d vs cuda-3d, and 0d-fast-path vs
cuda-3d, for both number counts and one-halo shear). Extending it
*across* the prior would need a third ini running both the fast 0d and
slow 3d modules in the *same* pipeline on the *same* draws (the
pattern `real_pipeline_extract_max2h.ini` already uses for CPU-vs-GPU)
-- not yet built.

## Recommended methods

The maintained smoke/reference pipeline uses:

| Dims | Observable | Strategy | Backend |
| --- | --- | --- | --- |
| `0d` | Number counts | fast path (2-dim GL, S_ij tab) | `NumCountsSijGl.so` |
| `0d` | One-halo miscentered shear | 1h z-contracted (1-dim GL) | `Shear1hGl.so` |
| `0d` | Projection shear | region-split GL exact-z | `ShearPrjGl.so` |

These are the production or reference choices for the DES Y3 implementations.
The corresponding `3d` adaptive methods are validation tools. The traditional
max model and the frozen projection GPU path (both `0d`) are optional
variants.

## Important limitations

- The radial-series `0d` algorithm currently uses a fixed concentration and
  is not validated as an accurate replacement for the varying-concentration
  one-halo model.
- The projection `3d` GPU calculation has unresolved wall-edge
  convergence; do not call its default result fully converged.
- The traditional max model depends on the known `haloModel/dSigma_hh` data
  defect and is provisional as a scientific result. See
  [`docs/known_issues/dsigma_hh_debug_flag.md`](../../../docs/known_issues/dsigma_hh_debug_flag.md).
- The benchmark values are fiducial measurements. They are useful for method
  selection and regression detection, not as hardware-independent guarantees.

For the detailed validation policy and current maintenance backlog, use the
validation documents under `docs/` rather than treating this README as the
source of module-level DataBlock contracts.
