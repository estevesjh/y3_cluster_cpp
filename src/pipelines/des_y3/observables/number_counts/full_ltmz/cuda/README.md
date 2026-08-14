# Number counts — `full_ltmz` reference (CUDA / PAGANI)

**Status: reference backend** — the GPU twin of the
[C++/Cuhre backend](../cpp/README.md), same triple integral, same
configuration contract (section `NumCountsFullLtmzGpu`,
`algorithm = pagani`). Built as `NumCountsFullLtmzGpu.so` when the tree
is configured with `-DUSE_CUDA=On`. Validation numbers live in this
file's "Validation" section and are produced by the GPU job script
described below.

This is the des_y3 namespace's first CUDA implementation, and it is
deliberately the *heavy-integrand* stage: a 3-D adaptive integral per
bin is what PAGANI is for (unlike the radial_series evaluator, whose
per-sample work is a few table lookups and stays CPU-only by design).

## Device physics

- `HMF_t`, `DV_DO_DZ_t`, `OMEGA_Z_DES`: the existing `y3_cuda` device
  models, reused as-is (same conventions as their host twins, including
  the HMF mass-axis shift and `hmf_s`/`hmf_q` nuisance).
- HOD, EMG richness kernel, photo-z kernel: no device versions existed,
  so `full_ltmz_device_kernels.cuh` carries verbatim
  `__host__ __device__` ports of `mor_hod_t.hh`, `plob_ltr_emg_t.hh`
  (clamped linear interpolation over the 15 z-nodes, exactly
  `Interp1D::clamp`) and `richness_kernel_t.hh` — scoped under
  `y3_cuda_des_y3`, no existing model or template touched.
  The numerical guard against port drift is the backend comparison
  below: CUDA vs C++ vs Python on identical pipelines.

The integrand object holds only fixed-size arrays and the `y3_cuda`
models, so it is trivially copyable to the device (the
`CosmoSISSICUDAModule` requirement).

## Build / validate

The tree's CUDA side builds only on Perlmutter GPU nodes with the
pinned toolchain. The self-contained job used for validation (build →
ctest → end-to-end pipeline → four-way comparison) is
`/pscratch/sd/j/jesteves/y3cpp_gpu_validate/build_and_validate.sh`
(debug QOS, 1 GPU, < 30 min); its ini runs the CUDA and C++ backends,
the Python reference, and production `NumCountsSel.so` in one pipeline.

## Validation (2026-08-12, SLURM job 56780752, 1×A100, fiducial point)

Four-way comparison on the pinned 12-bin wall (identical volumes and
`eps_rel = 1e-4` for both adaptive backends; pinned toolchain,
`Y3GCC_TARGET_ARCH=80-real`):

- vs the C++/Cuhre backend: **max |ratio − 1| = 6.0e-5** — two
  independent adaptive integrators over the same integrand;
- vs the Python fixed-GL reference: max |ratio − 1| = 5.1e-4;
- vs production `NumCountsSel.so`: max |ratio − 1| = 1.1e-3 — the same
  agreement class as the C++ backend (dominated by the production
  S_ij-tabulation step);
- cost: 1.97 s per sample for all 12 bins on one A100 (C++ Cuhre:
  3.4 s on CPU).

Output: `numcountsfullltmzgpu/{vals, errors, probs, status, nregions}`.
