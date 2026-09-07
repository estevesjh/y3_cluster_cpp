#include "catch2/catch.hpp"

#include "models/emg_des_t.cuh"
#include "models/mor_shifted_poisson_t.cuh"
#include "pipelines/des_y3/number_counts/cuda/3d/num_counts_3d_gpu_t.cuh"
// The class under test in the second half of this file: its constructor,
// set_grid_point()'s bin-index check, and module_label() are pure
// host-side DataBlock reads (no CUDA model is touched until
// set_sample()), so Shear1h3dGpu is instantiated directly below.
#include "pipelines/des_y3/shear_1h2h/cuda/3d/shear1h_3d_gpu_t.cuh"

#include "models/mor_hod_t.hh"
#include "models/plob_ltr_emg_t.hh"
#include "models/richness_kernel_t.hh"

#include "cosmosis/datablock/datablock.hh"
#include "cosmosis/datablock/ndarray.hh"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
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
  // dsigma_prj_3d_gpu.test.cu), so both entry points are exercised
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

// The full_ltmz GPU backends (Shear1h3dGpu / NumCounts3dGpu)
// compose the gpu_prj_costanzi2026 device models (Arwa Qadi, upstream
// PR #3) instead of the retired verbatim ports. This is the cross-check
// that keeps them honest against the CPU models the C++ backends use:
// EMG_DES_t's parameter provider and CDF-difference S_i versus
// PlobLtrEMG_t + RichnessKernel_t, and zkernel_sj versus
// richness_zkernel, all reading the SAME plob_ltr_params section.
TEST_CASE("des_y3 full_ltmz GPU device models match their CPU "
          "counterparts (Shear1h3dGpu / NumCounts3dGpu)")
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

namespace {
  cosmosis::DataBlock
  make_shear1h3d_bins_cfg(std::size_t n)
  {
    cosmosis::DataBlock cfg;
    char const* mod = "Shear1h3dGpu";
    cfg.put_val(mod, "lam_min", std::vector<double>(n, 20.0));
    cfg.put_val(mod, "lam_max", std::vector<double>(n, 30.0));
    cfg.put_val(mod, "zob_min", std::vector<double>(n, 0.2));
    cfg.put_val(mod, "zob_max", std::vector<double>(n, 0.4));
    cfg.put_val(mod, "sigma_z", std::vector<double>(n, 0.02));
    return cfg;
  }
}

TEST_CASE("Shear1h3dGpu constructor validates its bin definition arrays")
{
  SECTION("mismatched bin array lengths throw")
  {
    cosmosis::DataBlock cfg;
    char const* mod = "Shear1h3dGpu";
    cfg.put_val(mod, "lam_min", std::vector<double>{20.0, 20.0});
    cfg.put_val(mod, "lam_max", std::vector<double>{30.0});
    cfg.put_val(mod, "zob_min", std::vector<double>{0.2, 0.2});
    cfg.put_val(mod, "zob_max", std::vector<double>{0.4, 0.4});
    cfg.put_val(mod, "sigma_z", std::vector<double>{0.02, 0.02});
    CHECK_THROWS_AS(Shear1h3dGpu{cfg}, std::runtime_error);
  }
  SECTION("more than MAX_BINS (32) bins throws")
  {
    auto cfg = make_shear1h3d_bins_cfg(33);
    CHECK_THROWS_AS(Shear1h3dGpu{cfg}, std::runtime_error);
  }
  SECTION("a well-formed bin set constructs cleanly")
  {
    auto cfg = make_shear1h3d_bins_cfg(2);
    CHECK_NOTHROW(Shear1h3dGpu{cfg});
    CHECK(std::string(Shear1h3dGpu::module_label()) == "Shear1h3dGpu");
  }
}

TEST_CASE("Shear1h3dGpu constructor: lob_centers default, custom override, "
         "and validation")
{
  SECTION("default lob_centers used when the key is absent")
  {
    auto cfg = make_shear1h3d_bins_cfg(2);
    CHECK_NOTHROW(Shear1h3dGpu{cfg});
  }
  SECTION("custom lob_centers honored")
  {
    auto cfg = make_shear1h3d_bins_cfg(2);
    cfg.put_val("Shear1h3dGpu", "lob_centers", std::vector<double>{30.0, 90.0});
    CHECK_NOTHROW(Shear1h3dGpu{cfg});
  }
  SECTION("empty lob_centers throws")
  {
    auto cfg = make_shear1h3d_bins_cfg(2);
    cfg.put_val("Shear1h3dGpu", "lob_centers", std::vector<double>{});
    CHECK_THROWS_AS(Shear1h3dGpu{cfg}, std::runtime_error);
  }
  SECTION("more than MAX_BINS (32) lob_centers throws")
  {
    auto cfg = make_shear1h3d_bins_cfg(2);
    cfg.put_val("Shear1h3dGpu", "lob_centers", std::vector<double>(33, 50.0));
    CHECK_THROWS_AS(Shear1h3dGpu{cfg}, std::runtime_error);
  }
}

TEST_CASE("Shear1h3dGpu constructor honors use_halo_model_conc's has_val "
         "default and explicit override")
{
  SECTION("absent defaults to false (has_val false branch)")
  {
    auto cfg = make_shear1h3d_bins_cfg(1);
    CHECK_NOTHROW(Shear1h3dGpu{cfg});
  }
  SECTION("explicit true is accepted (has_val true branch)")
  {
    auto cfg = make_shear1h3d_bins_cfg(1);
    cfg.put_val("Shear1h3dGpu", "use_halo_model_conc", true);
    CHECK_NOTHROW(Shear1h3dGpu{cfg});
  }
}

TEST_CASE("Shear1h3dGpu::set_grid_point validates bin_index against the "
         "configured bin count")
{
  auto cfg = make_shear1h3d_bins_cfg(2);
  Shear1h3dGpu integrand(cfg);

  CHECK_NOTHROW(integrand.set_grid_point({0.0, 0.5}));
  CHECK_NOTHROW(integrand.set_grid_point({1.0, 1.2}));
  CHECK_THROWS_AS(integrand.set_grid_point({-1.0, 0.5}), std::out_of_range);
  CHECK_THROWS_AS(integrand.set_grid_point({2.0, 0.5}), std::out_of_range);
}

TEST_CASE("Shear1h3dGpu grid/volume builders parse a wall-of-numbers "
         "configuration")
{
  cosmosis::DataBlock cfg;
  char const* mod = "Shear1h3dGpu";
  cfg.put_val(mod, "bin_index", std::vector<double>{0.0, 1.0});
  cfg.put_val(mod, "r_perp", std::vector<double>{0.5, 1.0});
  cfg.put_val(mod, "lt_low", std::vector<double>{0.0, 0.0});
  cfg.put_val(mod, "lt_high", std::vector<double>{200.0, 200.0});
  cfg.put_val(mod, "zt_low", std::vector<double>{0.1, 0.1});
  cfg.put_val(mod, "zt_high", std::vector<double>{0.9, 0.9});
  cfg.put_val(mod, "lnm_low", std::vector<double>{30.0, 30.0});
  cfg.put_val(mod, "lnm_high", std::vector<double>{36.0, 36.0});

  auto const grid = Shear1h3dGpu::make_grid_points(cfg);
  CHECK(grid.size() == 2u);
  auto const vols = Shear1h3dGpu::make_integration_volumes(cfg);
  CHECK(vols.size() == 2u);
}

namespace {
  // A synthetic-but-complete sample: everything Shear1h3dGpu::set_sample
  // reads (HMF_t, DV_DO_DZ_t, OMEGA_Z_DES, haloModel/dSigma_nfw,
  // average_sigma_crit_inv, the miscentred-NFW table, MOR_HOD_t,
  // EMG_DES_t, and the required miscentering/f_mis,tau_mis scalars) --
  // same role as dsigma_prj_3d_gpu.test.cu's make_sample(), not a
  // fiducial physics pin. With `with_conc_and_phys`, also publishes
  // haloModel/concentration (for the use_halo_model_conc_ opt-in path's
  // set_concentration_table call) and haloModel/one_halo_physical_density
  // (for the phys_density_ has_val branch) -- both otherwise-untested
  // set_sample() branches.
  cosmosis::DataBlock
  make_shear1h3d_sample(bool with_conc_and_phys = false)
  {
    cosmosis::DataBlock db;
    db.put_val("cosmological_parameters", "h0", 0.7);
    db.put_val("cosmological_parameters", "omega_m", 0.3);
    db.put_val("cosmological_parameters", "omega_M", 0.3);
    db.put_val("cosmological_parameters", "omega_nu", 0.0);
    db.put_val("cosmological_parameters", "omega_lambda", 0.7);
    db.put_val("cosmological_parameters", "omega_k", 0.0);

    db.put_val("distances", "z", std::vector<double>{0.0, 0.3, 0.9});
    db.put_val("distances", "d_a",
              std::vector<double>{0.0, 1150.0 / 1.3, 2100.0 / 1.9});
    db.put_val("distances", "d_c", std::vector<double>{0.0, 1150.0, 2100.0});

    db.put_val("mass_function", "m_h", std::vector<double>{1.0e13, 1.0e16});
    db.put_val("mass_function", "z", std::vector<double>{0.0, 0.9});
    db.put_val(
      "mass_function", "dndlnmh",
      cosmosis::ndarray<double>(std::vector<double>{2.0, 2.0, 2.0, 2.0},
                                {2, 2}));
    db.put_val("cluster_abundance", "hmf_s", 1.0);
    db.put_val("cluster_abundance", "hmf_q", 0.0);

    db.put_val("cluster_mor", "log10_Mmin", 13.8);
    db.put_val("cluster_mor", "log10_M1", 14.5);
    db.put_val("cluster_mor", "alpha", 1.1);
    db.put_val("cluster_mor", "sigma_lambda", 0.35);
    db.put_val("cluster_mor", "epsilon", -0.2);

    std::vector<double> const zp{0.10, 0.45, 0.80};
    auto const flat = [](double v) { return std::vector<double>{v, v, v}; };
    db.put_val("plob_ltr_params", "z", zp);
    db.put_val("plob_ltr_params", "a_mu", flat(0.2));
    db.put_val("plob_ltr_params", "b_mu", flat(1.05));
    db.put_val("plob_ltr_params", "a_sig", flat(0.55));
    db.put_val("plob_ltr_params", "b_sig", flat(0.9));
    db.put_val("plob_ltr_params", "a_tau", flat(0.30));
    db.put_val("plob_ltr_params", "b_tau", flat(0.02));
    db.put_val("plob_ltr_params", "a_fprj", flat(1.2));
    db.put_val("plob_ltr_params", "b_fprj", flat(0.65));

    db.put_val("haloModel", "r_sigma", std::vector<double>{0.1, 5.0});
    db.put_val("haloModel", "lnM", std::vector<double>{29.0, 36.0});
    db.put_val(
      "haloModel", "dSigma_nfw",
      cosmosis::ndarray<double>(std::vector<double>{5.0, 5.0, 5.0, 5.0},
                                {2, 2}));
    db.put_val("haloModel", "rho_m_ref", 0.3 * 2.77533742639e+11);

    db.put_val("average_sigma_crit_inv", "zlense",
              std::vector<double>{0.0, 0.9});
    db.put_val("average_sigma_crit_inv", "sci_average",
              std::vector<double>{2.0e-4, 2.0e-4});

    // Required: no fallback to fiducial defaults.
    db.put_val("miscentering", "f_mis", 0.2);
    db.put_val("miscentering", "tau_mis", 0.3);

    if (with_conc_and_phys) {
      db.put_val("haloModel", "concentration", std::vector<double>{4.0, 6.0});
      db.put_val("haloModel", "one_halo_physical_density", 1);
    }
    return db;
  }

  // dsigma_nfw_/sci_ are quad::Interp2D/Interp1D-backed (device memory at
  // construction); evaluate through a trivial kernel, same reason and
  // pattern as dsigma_prj_3d_gpu.test.cu and the NumCounts3dGpu test above.
  __global__ void
  shear1h3d_eval_kernel(Shear1h3dGpu obj, double lt, double zt, double lnM,
                        double* out)
  {
    *out = obj(lt, zt, lnM);
  }

  double
  shear1h3d_eval_on_device(Shear1h3dGpu const& obj, double lt, double zt,
                           double lnM)
  {
    double* d_out = nullptr;
    REQUIRE(cudaMalloc(&d_out, sizeof(double)) == cudaSuccess);
    shear1h3d_eval_kernel<<<1, 1>>>(obj, lt, zt, lnM, d_out);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    double out = 0.0;
    REQUIRE(cudaMemcpy(&out, d_out, sizeof(double), cudaMemcpyDeviceToHost) ==
           cudaSuccess);
    cudaFree(d_out);
    return out;
  }
}

TEST_CASE("Shear1h3dGpu::set_sample builds its device models and "
         "operator() returns a finite value")
{
  auto cfg = make_shear1h3d_bins_cfg(1);
  Shear1h3dGpu integrand(cfg);

  auto sample = make_shear1h3d_sample();
  CHECK_NOTHROW(integrand.set_sample(sample));
  integrand.set_grid_point({0.0, 1.0});  // bin 0, R = 1.0 cMpc/h

  double const lt = 20.0, zt = 0.3, lnM = std::log(1.0e14);
  double const val = shear1h3d_eval_on_device(integrand, lt, zt, lnM);
  CHECK(std::isfinite(val));
}

TEST_CASE("Shear1h3dGpu::set_sample honors use_halo_model_conc "
         "(set_concentration_table) and one_halo_physical_density")
{
  // Complement of the TEST_CASE above: use_halo_model_conc=true so the
  // constructor's flag is true, then a sample publishing
  // haloModel/concentration and haloModel/one_halo_physical_density=1 so
  // set_sample() takes the has_val-true branch for both -- pins
  // dsigma_mis_->set_concentration_table(s) and phys_density_ = true,
  // neither of which the default-config TEST_CASE above reaches.
  auto cfg = make_shear1h3d_bins_cfg(1);
  cfg.put_val("Shear1h3dGpu", "use_halo_model_conc", true);
  Shear1h3dGpu integrand(cfg);

  auto sample = make_shear1h3d_sample(/*with_conc_and_phys=*/true);
  CHECK_NOTHROW(integrand.set_sample(sample));
  integrand.set_grid_point({0.0, 1.0});

  double const lt = 20.0, zt = 0.3, lnM = std::log(1.0e14);
  double const val = shear1h3d_eval_on_device(integrand, lt, zt, lnM);
  CHECK(std::isfinite(val));
}
