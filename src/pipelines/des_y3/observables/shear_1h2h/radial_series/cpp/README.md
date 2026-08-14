# One-halo miscentred shear — `radial_series` (C++)

**Status: reference/candidate backend** — same scientific contract,
committed derived data, and maintenance status as the
[Python backend](../python/README.md); production remains
`Shear1hMisSel.so`. Built as the CosmoSIS module `Shear1hRadialSeries.so`.

## Design

- `shear1h_radial_series_t.hh` — `RadialSeriesTable` (loads the
  `data/radial_series` **text export** once via `read_vector`, GSL
  bilinear/linear interpolation with clamped queries) and
  `Shear1hRadialSeries`, which composes the immutable
  `nosel_gl_detail::SelGLCore` for the exact fixed-GL redshift
  contraction and adds the y = ln r_s population moments (N, ȳ, μ₂, μ₃)
  from SelGLCore's public nodes/weights (SelGLCore itself is untouched —
  it only carries lnM moments through μ₂).
- `Shear1hRadialSeries.cc` — thin driver:
  `DEFINE_COSMOSIS_SCALAR_EVALUATOR_MODULE(Shear1hRadialSeries)`.
  No existing template or model header is modified (approved-plan rule).
- Registered in `src/modules/CMakeLists.txt` (the module registry) via an
  explicit-binary-dir `add_subdirectory`.

Ini options mirror `Shear1hMisSel` grid semantics (`bin_index` ×
`r_perp` cartesian product, bin slow / R fast; `lob_centers`;
required `zt_low/zt_high/lnm_low/lnm_high`; `n_lnm`/`n_z`), plus
`ell_max` (2 default / 3) and `table_stem`. Output:
`shear1h_radial_series/vals` — the same section as the Python backend
(alternative backends of one stage; never run both in one pipeline).

## Backend equivalence

Both backends read the same committed values
(`data/radial_series/radial_series_nfw_mis_gamma_v1*`); the only
difference is interpolation (Python: cubic on the npz; C++: GSL
bilinear/linear on the text export, the pipeline's production
convention). Measured with `../python/compare_backends.py` on the real
12-bin × 10-radius grid: **max relative difference 1.6e-4**, far below
the ~0.45% ℓ≤2 truncation tolerance that governs the strategy.

## Tests

`test/shear1h_radial_series.test.cc` (`ctest -R shear1h_radial_series`):

- table interpolation and full series assembly against golden values
  computed with an independent Python bilinear reference (1e-9);
- the composite `Ω_m · A0(y) · U0` against the production
  `NFW_DSIGMA_MIS` reader at physical points (5e-4 — the measured U₀
  fidelity), so the fixed conventions cannot drift apart silently;
- the centred limit `U0_mis(x, x_mis→min) → U0_cen(x)`.

## Build / validate checklist (GPU node, official recipe)

From `BUILDING.md` (the tree only officially builds on Perlmutter GPU
compute nodes with the pinned toolchain):

```bash
salloc --nodes 1 --qos interactive --time 02:00:00 --constraint gpu \
       --gpus 4 --account=des_g
source ~/cosmosis_init.sh
module swap cudatoolkit/12.9 cudatoolkit/12.2
module swap gcc-native/13.2 gcc-native/12.3
export PATH=$(echo $PATH | tr ':' '\n' | grep -v homebrew | tr '\n' ':' | sed 's/:$//')
export Y3_CLUSTER_CPP_DIR=/pscratch/sd/j/jesteves/github/y3_cluster_cpp
cd $Y3_CLUSTER_CPP_DIR/release-build
cmake .   # picks up the new registry entry
ninja Shear1hRadialSeries shear1h_radial_series_test
ctest -R shear1h_radial_series
```

Then the end-to-end cross-check on a login node (modules load fine
there): run the smoke pipeline with `Shear1hRadialSeries.so` in place of
the Python backend and compare `shear1h_radial_series/vals` against
`../python/compare_backends.py`'s bilinear column — agreement should be
at text-roundtrip precision (~1e-12).

A CPU-only login-node build (`cmake -DCMAKE_BUILD_TYPE=Release` without
`-DUSE_CUDA`) also compiles this module and its test; it was used for
the first validation pass of this backend (see the commit message).
The pinned-toolchain re-run happened 2026-08-12 in the CUDA-configured
build (`/pscratch/sd/j/jesteves/y3cpp_gpu_build`, cudatoolkit 12.2 +
gcc 12.3): module and test build cleanly and the ctest passes.

## CUDA

Deliberately not implemented for this stage: after the offline tables
and per-sample moments, evaluation is a handful of table lookups per
(bin, R) — microseconds on CPU, nothing for a GPU to accelerate. The
natural first CUDA target in the des_y3 namespace is a heavy integrand
(e.g. a `full_ltmz` PAGANI reference), not this evaluator. The derived
data is backend-neutral by construction ("CPU and CUDA implementations
must read the same derived data"), so a future CUDA port reads the same
tables.
