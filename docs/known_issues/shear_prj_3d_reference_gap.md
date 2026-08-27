# ⚠ Projection shear has no CPU or Python 3-D adaptive reference — only the CUDA/PAGANI diagnostic

**Raised 2026-08-26** during the des_y3 precision/cost table rebuild
(`src/pipelines/des_y3/README.md`, issue #23 task). Not fixed here —
tracked as a gap to close another day.

## The asymmetry

Number counts and one-halo shear each have a genuine three-way
cross-check at their maximum-dimension (`3d`) tier:

| Observable | Python 3d | C++ 3d (Cuhre) | CUDA 3d (PAGANI) |
| --- | --- | --- | --- |
| Number counts | `shared/explicit_grid_core.py` | `NumCounts3d` | `NumCounts3dGpu` |
| One-halo shear | `shared/explicit_grid_core.py` | `Shear1h3d` | `Shear1h3dGpu` |
| **Projection shear** | **none** | **none** | `DSigmaPrj3dGpu` only |

For the first two observables, three independently-written
implementations (different language, different quadrature library)
converging to the same answer at `eps_rel~1e-4` IS the accuracy
certificate — that is what "Precision vs 3d" measures throughout
`src/pipelines/des_y3/README.md`. Projection shear's `3d` row has no
such certificate: `DSigmaPrj3dGpu` (`src/pipelines/des_y3/shear_projection/cuda/3d/`)
is the *only* fully-coupled (θ, z, lnM) adaptive implementation of the
Σ_prj/DSigma_prj integral in the whole repo. There is no
`shear_projection/cpp/3d/` and no `shear_projection/python/.../3d`
directory — per this repo's own convention ("a missing
`<language>/<dims>/` directory means the backend is not implemented",
`src/pipelines/des_y3/README.md`), this backend is simply missing, not
just unmeasured.

## Why it matters

This is exactly why `shear_projection/README.md`'s `3d` row is
documented as "Reference-class diagnostic (3d); convergence open" and
"median 9.5e-4, maximum 2.2% vs region-split GL" rather than a clean
"Reference (3d)" the way the other two observables' `3d` rows read. The
2.2% number is a cross-*approximation* check against the `0d`
region-split fixed-GL evaluator (itself not adaptive), not a
convergence check against an independent adaptive implementation. If
`DSigmaPrj3dGpu` has a bug shared with no other code path (e.g. a
subtly wrong quadrature weight, an off-by-one in the θ/z/lnM volume
construction), nothing in the current test suite would catch it,
because there is nothing else in the repo computing the same
fully-coupled integral to compare against.

## Scope of the fix (deferred)

Two independent pieces, either of which alone would close most of the
gap (having both is what number counts and one-halo shear enjoy):

1. **`shear_projection/cpp/3d/`**: an adaptive Cuhre C++ backend over
   the θ_prj/z_ob/M_prj integrand (`src/models/sigma_prj_t.hh`'s
   `sp_detail::ShearPrjCore` documents the integral; `NumCounts3d`/
   `Shear1h3d` are the structural template — CosmoSIS integration
   module reading `algorithm`/`eps_rel`/`eps_abs`/`max_eval`/
   `use_cartesian_product`, same `lambda_bin`/`zo_low`/`zo_high`/
   `radii` grid convention as `ShearPrjCuhre`/`ShearPrjGl`).
2. **`shear_projection/python/.../3d`**: an explicit adaptive Python
   reference in the `shared/explicit_grid_core.py` style already used
   for number counts and one-halo shear (adaptively subdividing the
   outer mass/redshift integral around the fixed-GL inner
   contraction, reporting quadrature error at or below 1e-6).

Either would let `DSigmaPrj3dGpu`'s "convergence open" caveat be
retired the same way the number-counts and one-halo-shear `3d` rows
already are validated.

## Status

Deliberately deferred. No code changes in this pass; this note is the
tracking record. See `src/pipelines/des_y3/README.md`'s "Precision and
cost overview" and `shear_projection/README.md`'s "The 3d backend"
section for the current (diagnostic-only) numbers.
