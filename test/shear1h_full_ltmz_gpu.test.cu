#include "catch2/catch.hpp"

#include "models/emg_des_t.cuh"
#include "models/mor_shifted_poisson_t.cuh"
#include "pipelines/des_y3/number_counts/cuda/3d/num_counts_full_ltmz_gpu_t.cuh"

#include "models/mor_hod_t.hh"
#include "models/plob_ltr_emg_t.hh"
#include "models/richness_kernel_t.hh"

#include "cosmosis/datablock/datablock.hh"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <vector>

using y3_cluster::PlobLtrEMG_t;
using y3_cluster::RichnessKernel_t;
using y3_cluster::richness_zkernel;

namespace {
  constexpr double REL_TOL = 1.0e-3;

  // Same synthetic plob_ltr_params fixture as
  // test/num_counts_full_ltmz.test.cc (z-independent EMG parameters), so
  // both the CPU and CUDA readers of the section see identical data.
  cosmosis::DataBlock
  make_plob_config()
  {
    cosmosis::DataBlock cfg;
    std::vector<double> const z{0.10, 0.45, 0.80};
    auto const flat = [](double v) { return std::vector<double>{v, v, v}; };
    cfg.put_val("plob_ltr_params", "z", z);
    cfg.put_val("plob_ltr_params", "a_mu", flat(0.2));
    cfg.put_val("plob_ltr_params", "b_mu", flat(1.05));
    cfg.put_val("plob_ltr_params", "a_sig", flat(0.55));
    cfg.put_val("plob_ltr_params", "b_sig", flat(0.9));
    cfg.put_val("plob_ltr_params", "a_tau", flat(0.30));
    cfg.put_val("plob_ltr_params", "b_tau", flat(0.02));
    cfg.put_val("plob_ltr_params", "a_fprj", flat(1.2));
    cfg.put_val("plob_ltr_params", "b_fprj", flat(0.65));
    return cfg;
  }

  // EMG_DES_t holds quad::Interp1D tables, which live in device memory —
  // host-side evaluation segfaults (same reason as
  // dsigma_prj_full_ltmz_gpu.test.cu), so both entry points are exercised
  // through 1-thread kernels with the model passed by value, exactly as
  // the modules pass their integrands.
  __global__ void
  params_kernel(y3_cuda::EMG_DES_t emg, double ltr, double z, double* out)
  {
    double mu, sigma, tau, fprj;
    emg.get_params(ltr, z, mu, sigma, tau, fprj);
    out[0] = mu;
    out[1] = sigma;
    out[2] = tau;
    out[3] = fprj;
  }

  __global__ void
  si_kernel(y3_cuda::EMG_DES_t emg, double lam_min, double lam_max,
            double ltr, double z, double* out)
  {
    *out = emg.cdf(lam_max, ltr, z) - emg.cdf(lam_min, ltr, z);
  }

  std::array<double, 4>
  device_params(y3_cuda::EMG_DES_t const& emg, double ltr, double z)
  {
    double* d_out = nullptr;
    REQUIRE(cudaMalloc(&d_out, 4 * sizeof(double)) == cudaSuccess);
    params_kernel<<<1, 1>>>(emg, ltr, z, d_out);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    std::array<double, 4> out{};
    REQUIRE(cudaMemcpy(out.data(), d_out, 4 * sizeof(double),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    cudaFree(d_out);
    return out;
  }

  double
  device_si(y3_cuda::EMG_DES_t const& emg, double lam_min, double lam_max,
            double ltr, double z)
  {
    double* d_out = nullptr;
    REQUIRE(cudaMalloc(&d_out, sizeof(double)) == cudaSuccess);
    si_kernel<<<1, 1>>>(emg, lam_min, lam_max, ltr, z, d_out);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    double out = 0.0;
    REQUIRE(cudaMemcpy(&out, d_out, sizeof(double),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    cudaFree(d_out);
    return out;
  }
}

// The full_ltmz GPU backends (Shear1hFullLtmzGpu / NumCountsFullLtmzGpu)
// compose the gpu_prj_costanzi2026 device models (Arwa Qadi, upstream
// PR #3) instead of the retired verbatim ports. This is the cross-check
// that keeps them honest against the CPU models the C++ backends use:
// EMG_DES_t's parameter provider and CDF-difference S_i versus
// PlobLtrEMG_t + RichnessKernel_t, and zkernel_sj versus
// richness_zkernel, all reading the SAME plob_ltr_params section.
TEST_CASE("des_y3 full_ltmz GPU device models match their CPU "
          "counterparts (Shear1hFullLtmzGpu / NumCountsFullLtmzGpu)")
{
  auto cfg = make_plob_config();
  PlobLtrEMG_t const plob_cpu(cfg);
  y3_cuda::EMG_DES_t emg_gpu(cfg);

  // EMG parameter provider: mu/sigma/tau/fprj at (ltr, z).
  for (auto const& p : std::vector<std::array<double, 2>>{
        {25.0, 0.35}, {60.0, 0.5}, {110.0, 0.65}}) {
    double const ltr = p[0], z = p[1];
    auto const got = device_params(emg_gpu, ltr, z);
    CHECK(got[0] == Approx(plob_cpu.mu(ltr, z)).epsilon(REL_TOL));
    CHECK(got[1] == Approx(plob_cpu.sigma(ltr, z)).epsilon(REL_TOL));
    CHECK(got[2] == Approx(plob_cpu.tau(ltr, z)).epsilon(REL_TOL));
    CHECK(got[3] == Approx(plob_cpu.fprj(ltr, z)).epsilon(REL_TOL));
  }

  // S_i via the analytic CDF difference versus RichnessKernel_t's
  // composition of the same closed forms:
  //   S_i = F_total(lam_max | ltr, z) - F_total(lam_min | ltr, z).
  RichnessKernel_t const s_i(20.0, 30.0);
  for (auto const& p : std::vector<std::array<double, 2>>{
        {25.0, 0.35}, {60.0, 0.5}, {110.0, 0.65}}) {
    double const ltr = p[0], z = p[1];
    double const expected = s_i(ltr, z, plob_cpu);
    double const got = device_si(emg_gpu, 20.0, 30.0, ltr, z);
    CHECK(got == Approx(expected).epsilon(REL_TOL));
  }

  // richness_zkernel vs zkernel_sj (Gaussian observed-redshift S_j) —
  // pure closed form, host-callable.
  for (auto const& p : std::vector<std::array<double, 4>>{
        {0.30, 0.25, 0.45, 0.02}, {0.42, 0.25, 0.45, 0.02},
        {0.55, 0.25, 0.45, 0.02}}) {
    double const zt = p[0], zmin = p[1], zmax = p[2], sig = p[3];
    CHECK(y3_cuda_des_y3::zkernel_sj(zt, zmin, zmax, sig) ==
          Approx(richness_zkernel(zt, zmin, zmax, sig)).epsilon(REL_TOL));
  }
}
