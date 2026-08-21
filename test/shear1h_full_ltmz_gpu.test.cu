#include "catch2/catch.hpp"

#include "pipelines/des_y3/observables/number_counts/full_ltmz/cuda/full_ltmz_device_kernels.cuh"

#include "models/mor_hod_t.hh"
#include "models/plob_ltr_emg_t.hh"
#include "models/richness_kernel_t.hh"

#include "cosmosis/datablock/datablock.hh"

#include <array>
#include <cmath>
#include <vector>

using y3_cluster::MOR_HOD_t;
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
}

// full_ltmz_device_kernels.cuh's own header comment: those host models
// "have no .cuh counterparts, and the approved plan forbids editing them,
// so the CUDA backend carries these verbatim __host__ __device__ ports
// instead" -- MorHodDevice/PlobEmgDevice/richness_kernel/zkernel versus
// MOR_HOD_t/PlobLtrEMG_t/RichnessKernel_t/richness_zkernel. This test is
// exactly the cross-check that comment calls for; it is shared by both
// Shear1hFullLtmzGpu.cu and NumCountsFullLtmzGpu.cu, which both #include
// this header.
TEST_CASE("des_y3 full_ltmz device kernel ports match their CPU "
          "counterparts (Shear1hFullLtmzGpu / NumCountsFullLtmzGpu)")
{
  // MOR_HOD_t vs MorHodDevice: both the normal shifted-Poisson branch and
  // the mu_sat -> 0 narrow-Gaussian fallback branch.
  MOR_HOD_t const mor_cpu(/*log10_Mmin=*/13.8, /*log10_M1=*/14.5,
                         /*alpha=*/1.1, /*epsilon=*/-0.2,
                         /*sigma_lambda=*/0.35);
  y3_cuda_des_y3::MorHodDevice mor_gpu;
  mor_gpu.log10_Mmin = 13.8;
  mor_gpu.log10_M1 = 14.5;
  mor_gpu.alpha = 1.1;
  mor_gpu.epsilon = -0.2;
  mor_gpu.sigma_lambda = 0.35;

  double const lnM = std::log(std::pow(10.0, 14.2));
  double const lnM_near_min = std::log(std::pow(10.0, 13.79)); // mu_sat ~ 0
  for (auto const& p : std::vector<std::array<double, 3>>{
        {5.0, lnM, 0.3}, {12.0, lnM, 0.5}, {1.0, lnM_near_min, 0.4}}) {
    double const lt = p[0], m = p[1], zt = p[2];
    CHECK(mor_gpu(lt, m, zt) == Approx(mor_cpu(lt, m, zt)).epsilon(REL_TOL));
  }

  // PlobLtrEMG_t vs PlobEmgDevice, reading the SAME plob_ltr_params section.
  auto cfg = make_plob_config();
  PlobLtrEMG_t const plob_cpu(cfg);
  auto const plob_gpu = y3_cuda_des_y3::PlobEmgDevice::from_datablock(cfg);

  for (auto const& p : std::vector<std::array<double, 2>>{
        {25.0, 0.35}, {60.0, 0.5}, {110.0, 0.65}}) {
    double const ltr = p[0], z = p[1];
    CHECK(plob_gpu.mu(ltr, z) ==
          Approx(plob_cpu.mu(ltr, z)).epsilon(REL_TOL));
    CHECK(plob_gpu.sigma(ltr, z) ==
          Approx(plob_cpu.sigma(ltr, z)).epsilon(REL_TOL));
    CHECK(plob_gpu.tau(ltr, z) ==
          Approx(plob_cpu.tau(ltr, z)).epsilon(REL_TOL));
    CHECK(plob_gpu.fprj(ltr, z) ==
          Approx(plob_cpu.fprj(ltr, z)).epsilon(REL_TOL));
  }

  // RichnessKernel_t vs richness_kernel, fed the device EMG parameters.
  RichnessKernel_t const k_i(20.0, 30.0);
  for (auto const& p : std::vector<std::array<double, 2>>{
        {25.0, 0.35}, {60.0, 0.5}, {110.0, 0.65}}) {
    double const ltr = p[0], z = p[1];
    double const expected = k_i(ltr, z, plob_cpu);
    double const got = y3_cuda_des_y3::richness_kernel(
      20.0, 30.0, plob_gpu.mu(ltr, z), plob_gpu.sigma(ltr, z),
      plob_gpu.tau(ltr, z), plob_gpu.fprj(ltr, z));
    CHECK(got == Approx(expected).epsilon(REL_TOL));
  }

  // richness_zkernel vs zkernel (Gaussian photo-z K_j).
  for (auto const& p : std::vector<std::array<double, 4>>{
        {0.30, 0.25, 0.45, 0.02}, {0.42, 0.25, 0.45, 0.02},
        {0.55, 0.25, 0.45, 0.02}}) {
    double const zt = p[0], zmin = p[1], zmax = p[2], sig = p[3];
    CHECK(y3_cuda_des_y3::zkernel(zt, zmin, zmax, sig) ==
          Approx(richness_zkernel(zt, zmin, zmax, sig)).epsilon(REL_TOL));
  }
}
