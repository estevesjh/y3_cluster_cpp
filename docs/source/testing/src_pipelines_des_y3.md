# `src/pipelines/des_y3/`

The tests below are split by the observable folder. Strategy folders
count adaptive integration dimensions only: `0d` collects every
fixed-GL/table backend (formerly `fast_mass`, `radial_series`, and the
`full_ltmz` fixed-GL Python references), while `2d`/`3d` are the
adaptive backends; test-target and file names keep their historical
strings. The implementation links point to the source being
exercised; validation scripts are listed separately when they are not
CTest targets.

## `number_counts/`

| Test target/source | Implementation or script under test | What it tests | Status |
|---|---|---|---|
| [`numcounts_full_ltmz_test`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/test/numcounts_full_ltmz.test.cc) | [`cpp/3d/NumCountsFullLtmz.cc`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/src/pipelines/des_y3/number_counts/cpp/3d/NumCountsFullLtmz.cc) | Explicit selection integration and independent kernel/HOD composition | Passing |
| [`python/0d/validate_explicit_vs_production.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/src/pipelines/des_y3/number_counts/python/0d/validate_explicit_vs_production.py) | [`python/0d/numcounts_full_ltmz.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/src/pipelines/des_y3/number_counts/python/0d/numcounts_full_ltmz.py) | Python reference versus production counts on a saved dump | Standalone validation |
| [`python/0d/validate_fast_vs_production.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/src/pipelines/des_y3/number_counts/python/0d/validate_fast_vs_production.py) | [`python/0d/numcounts_fast_mass.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/src/pipelines/des_y3/number_counts/python/0d/numcounts_fast_mass.py) | Python fast-mass replica versus `NumCountsSel.so` | Standalone validation |
| — | [`cpp/0d/NumCountsFastMass.cc`](https://github.com/estevesjh/y3_cluster_cpp/blob/pipelines/des_y3/src/pipelines/des_y3/number_counts/cpp/0d/NumCountsFastMass.cc) | Thin des_y3-labeled wrapper around `SelGLCore`/`NumCountsSelGL`, expected bitwise-identical to production `NumCountsSel.so` | **No dedicated unit test yet** — new module, syntax-verified against real include paths only; not exercised by `numcounts_cross_backend_test` or any Catch2 target |

CUDA source exists under `cuda/3d/`; its CTest targets are
registered centrally in `test/CMakeLists.txt`.

## `shear_1h2h/`

| Test target/source | Implementation or script under test | What it tests | Status |
|---|---|---|---|
| [`shear1h_full_ltmz_test`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/test/shear1h_full_ltmz.test.cc) | [`cpp/3d/Shear1hFullLtmz.cc`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/src/pipelines/des_y3/shear_1h2h/cpp/3d/Shear1hFullLtmz.cc) | Explicit one-halo integration, mixture affinity, radial behavior | Passing |
| [`shear1h_fast_mass_test`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/test/shear1h_fast_mass.test.cc) | [`cpp/0d/Shear1hFastMass.cc`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/src/pipelines/des_y3/shear_1h2h/cpp/0d/Shear1hFastMass.cc) | Exact redshift contraction, NFW mixture, and production identity | Passing |
| [`shear1h2h_max_test`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/test/shear1h2h_max.test.cc) | [`cpp/0d/Shear1h2hMax.cc`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/src/pipelines/des_y3/shear_1h2h/cpp/0d/Shear1h2hMax.cc) | Traditional max composition and NaN handling in `dSigma_hh` | Characterization; two-halo issue remains open |
| [`shear1h_radial_series_test`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/test/shear1h_radial_series.test.cc) | [`cpp/0d/shear1h_radial_series_t.hh`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/src/pipelines/des_y3/shear_1h2h/cpp/0d/shear1h_radial_series_t.hh) | Offline $U_\ell$ tables, NFW scale/amplitude separation, mixture endpoints, direct moment decomposition | Passing for internal decomposition/lookup identities; cross-backend raw-$\Delta\Sigma$ comparison is known failing (56–86%) because the Python radial-series family fixes $c=4$; see [`radial_series_vs_full_ltmz_defect.md`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/docs/radial_series_vs_full_ltmz_defect.md) |
| [`python/0d/validate_radial_series.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/src/pipelines/des_y3/shear_1h2h/python/0d/validate_radial_series.py) | [`python/0d/shear1h_radial_series.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/src/pipelines/des_y3/shear_1h2h/python/0d/shear1h_radial_series.py) and [`generate_radial_series_tables.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/src/pipelines/des_y3/shear_1h2h/python/0d/generate_radial_series_tables.py) | Independent derivative route and 12-bin truncation study | Standalone validation |
| [`python/0d/validate_shear1h2h_max.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/src/pipelines/des_y3/shear_1h2h/python/0d/validate_shear1h2h_max.py) | [`python/0d/shear1h2h_max.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/src/pipelines/des_y3/shear_1h2h/python/0d/shear1h2h_max.py) | Max-model comparison and two-halo defect characterization | Standalone validation |

## `shear_projection/`

| Test target/source | Implementation or script under test | What it tests | Status |
|---|---|---|---|
| [`shear_prj_fast_mass_test`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/test/shear_prj_fast_mass.test.cc) | [`cpp/0d/ShearPrjFastMass.cc`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/src/pipelines/des_y3/shear_projection/cpp/0d/ShearPrjFastMass.cc) | Theta grid, exclusion angle, NFW convention, wall construction | Passing |
| [`python/0d/validate_vs_production.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/src/pipelines/des_y3/shear_projection/python/0d/validate_vs_production.py) | [`python/0d/shear_prj_fast_mass.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/src/pipelines/des_y3/shear_projection/python/0d/shear_prj_fast_mass.py) | Exact projection evaluator and frozen-physics comparison | Standalone validation |
| [`shear_prj_frozen_gpu_test`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/test/shear_prj_frozen_gpu.test.cu) | [`cuda/0d/ShearPrjFrozenGpu.cu`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/src/pipelines/des_y3/shear_projection/cuda/0d/ShearPrjFrozenGpu.cu) | CUDA frozen-projection backend | GPU-only |
| [`dsigma_prj_full_ltmz_gpu_test`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/test/dsigma_prj_full_ltmz_gpu.test.cu) | [`cuda/3d/DSigmaPrjFullLtmzGpu.cu`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/src/pipelines/des_y3/shear_projection/cuda/3d/DSigmaPrjFullLtmzGpu.cu) | PAGANI full projection integration | GPU-only; wall-edge convergence remains under study |

## `shared/`

| Test source/target | Shared code under test | What it tests | Status |
|---|---|---|---|
| [`des_y3_pipeline.test.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/test/des_y3_pipeline.test.py) / `des_y3_pipeline_python_test` | [`shared/datablock_models.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/src/pipelines/des_y3/shared/datablock_models.py), [`full_ltmz_core.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/src/pipelines/des_y3/shared/full_ltmz_core.py), [`lensing_profiles.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/src/pipelines/des_y3/shared/lensing_profiles.py) | GL quadrature, moments, NFW decomposition, and shared model adapters | Passing |
| [`shear1h_cross_backend.test.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/test/shear1h_cross_backend.test.py) / `shear1h_cross_backend_test` | Cross-backend one-halo shear strategies | C++ `3d` fixture versus the CUDA `3d` backend and the Python/C++ `0d` backends (explicit GL, $z$-contracted mass sum, radial series, max model) | **Known failing:** raw radial-series amplitude comparison; C++/Python radial-series implementation identity remains passing |
| [`sel_function.test.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/test/sel_function.test.py) / `sel_function_test` | [`shared/sel_function.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/src/pipelines/des_y3/shared/sel_function.py), [`sel_kernels.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/src/pipelines/des_y3/shared/sel_kernels.py) | Independent EMG/photo-$z$ and HOD checks plus the numba-fused `_cdf_lob_stacked` production hot path; `_f_emg` remains a parity-only helper | **Known failing:** 3 normalization tests; fused-kernel checks pass; see [`docs/hod_normalization_defect.md`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/docs/hod_normalization_defect.md) |
| [`halo_model.test.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/test/halo_model.test.py) | Shared pipeline consumer: [`y3_buzzard/haloModel.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/y3_buzzard/haloModel.py) | Bias/concentration identities and observed `dSigma_hh` NaN behavior | Characterization; fixture-dependent checks may skip |
