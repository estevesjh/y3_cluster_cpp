#include "catch2/catch.hpp"

// This is the code we're actually testing: the device models the
// explicit-3d GPU backends compose — y3_cuda::MOR_HOD_t (the device
// mirror of the host HOD MOR), y3_cuda::EMG_DES_t's closed-form
// primitives, the b_sel-operator MOR y3_cuda::MOR_SHIFTED_POISSON_t —
// plus the zkernel_sj observed-redshift kernel and the NumCounts3dGpu
// class itself, carried by num_counts_3d_gpu_t.cuh. NumCounts3dGpu's
// constructor, set_grid_point() and the make_grid_points()/
// make_integration_volumes() static glue are pure host-side DataBlock
// reads (no CUDA model is touched until set_sample()), so they are
// instantiated and exercised directly below. set_sample() and operator()
// need the HMF_t/DV_DO_DZ_t/OMEGA_Z_DES/MOR_HOD_t/EMG_DES_t device models
// live -- a synthetic-but-complete sample is built further below (same
// role as dsigma_prj_3d_gpu.test.cu's make_sample()) so that path gets
// exercised too, not just the individual device models in isolation.
#include "models/emg_des_t.cuh"
#include "models/mor_hod_t.cuh"
#include "models/mor_shifted_poisson_t.cuh"
#include "pipelines/des_y3/number_counts/cuda/3d/num_counts_3d_gpu_t.cuh"

// Host twins: rk_detail's phi/erfcx closed forms (richness_kernel_t.hh)
// for the EMG primitives and richness_zkernel for S_j; MOR_HOD_t for
// the shifted-Poisson identity below. Comparing against them directly
// is a stronger, more independent check than hand-picking reference
// numbers, and needs no DataBlock or dump.
#include "models/mor_hod_t.hh"
#include "models/richness_kernel_t.hh"

#include "cosmosis/datablock/datablock.hh"
#include "cosmosis/datablock/ndarray.hh"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

using y3_cluster::MOR_HOD_t;
namespace rk = y3_cluster::rk_detail;

namespace {
  constexpr double PORT_TOL = 1.0e-13;
}

TEST_CASE("emg_des_t device phi_cdf/erfcx_impl match the host rk_detail twins")
{
  for (double x : {-3.5, -1.0, -0.1, 0.0, 0.37, 1.2, 4.0}) {
    CHECK(y3_cuda::phi_cdf(x) == Approx(rk::phi(x)).epsilon(PORT_TOL));
  }
  // Cover both erfcx_impl branches: direct (|x| < 4) and the asymptotic
  // series (|x| >= 4).
  for (double t : {0.0, 0.5, 3.0, 3.9, 4.0, 10.0, 30.0, 100.0}) {
    CHECK(y3_cuda::erfcx_impl(t) == Approx(rk::erfcx(t)).epsilon(PORT_TOL));
  }
}

TEST_CASE("zkernel_sj matches the host richness_zkernel closed form")
{
  for (auto const& p : {std::array<double, 4>{0.30, 0.25, 0.45, 0.02},
                        std::array<double, 4>{0.42, 0.25, 0.45, 0.02},
                        std::array<double, 4>{0.55, 0.25, 0.45, 0.02},
                        std::array<double, 4>{0.10, 0.25, 0.45, 0.08}}) {
    double const zt = p[0], zmin = p[1], zmax = p[2], sig = p[3];
    CHECK(y3_cuda_des_y3::zkernel_sj(zt, zmin, zmax, sig) ==
          Approx(y3_cluster::richness_zkernel(zt, zmin, zmax, sig))
            .epsilon(PORT_TOL));
  }
}

TEST_CASE("MOR_SHIFTED_POISSON_t is MOR_HOD_t shifted by the central count")
{
  // The Costanzi-2026 P-operator form (x = ltr + delta) relates to
  // MOR_HOD_t's central-shifted form (x = ltr - lambda_cen + delta,
  // lambda_cen = 1 above Mmin) exactly by
  //
  //   MOR_SP(ltr, lnM, z) = MOR_HOD(ltr + 1, lnM, z)   for M >= Mmin
  //
  // away from the mu_sat -> 0 fallback branch. This pins the
  // b_sel-operator device model's formula; the explicit-3d GPU
  // backends themselves use y3_cuda::MOR_HOD_t (parity test below),
  // so no cross-backend offset survives in the pipeline.
  double const log10_Mmin = 13.8, log10_M1 = 14.5, alpha = 1.1,
               sigma_lambda = 0.35, epsilon = -0.2, z_pivot = 0.45;
  MOR_HOD_t const mor_hod(log10_Mmin, log10_M1, alpha, epsilon,
                          sigma_lambda, z_pivot);
  y3_cuda::MOR_SHIFTED_POISSON_t const mor_sp(
    log10_Mmin, log10_M1, alpha, sigma_lambda, epsilon, z_pivot);

  double const lnM = std::log(std::pow(10.0, 14.2));
  double const lnM_high = std::log(std::pow(10.0, 14.8));
  for (auto const& p : {std::array<double, 3>{5.0, lnM, 0.3},
                        std::array<double, 3>{12.0, lnM, 0.5},
                        std::array<double, 3>{40.0, lnM_high, 0.4},
                        std::array<double, 3>{80.0, lnM_high, 0.6}}) {
    double const lt = p[0], m = p[1], zt = p[2];
    CHECK(mor_sp(lt, m, zt) ==
          Approx(mor_hod(lt + 1.0, m, zt)).epsilon(PORT_TOL));
  }

  // Below Mmin both conventions vanish.
  double const lnM_below = std::log(std::pow(10.0, 13.5));
  CHECK(mor_sp(5.0, lnM_below, 0.4) == 0.0);
}

TEST_CASE("y3_cuda::MOR_HOD_t matches the host MOR_HOD_t exactly")
{
  // The device HOD MOR is the model the explicit-3d GPU backends
  // (NumCounts3dGpu, Shear1h3dGpu) actually compose; it must be
  // bit-level the same algebra as the host class so CPU<->GPU
  // cross-backend pins compare identical physics.
  double const log10_Mmin = 11.4, log10_M1 = 12.7, alpha = 0.86,
               sigma_lambda = 0.18, epsilon = 0.0, z_pivot = 0.45;
  MOR_HOD_t const host(log10_Mmin, log10_M1, alpha, epsilon,
                       sigma_lambda, z_pivot);
  y3_cuda::MOR_HOD_t const dev(log10_Mmin, log10_M1, alpha, epsilon,
                               sigma_lambda, z_pivot);

  // Span the DES Y3 integration range: masses below/at/above Mmin,
  // richness from the lambda_central cutoff into the bins, both
  // redshift edges. Includes the mu_sat -> 0 Gaussian fallback branch
  // (M barely above Mmin) and the lt < lambda_central hard zero.
  for (double log10_M : {11.0, 11.4000001, 11.8, 13.0, 14.5, 15.5}) {
    double const lnM = std::log(std::pow(10.0, log10_M));
    for (double lt : {0.0, 0.5, 0.9999, 1.0, 2.0, 20.0, 60.0, 199.0}) {
      for (double zt : {0.05, 0.45, 0.80}) {
        double const h = host(lt, lnM, zt);
        double const d = dev(lt, lnM, zt);
        if (h == 0.0) {
          CHECK(d == 0.0);
        } else {
          CHECK(d == Approx(h).epsilon(PORT_TOL));
        }
      }
    }
  }
}

// The TEST_CASEs above isolate the device models NumCounts3dGpu composes.
// NumCounts3dGpu itself is never instantiated -- its constructor, its bin
// out-of-range check, and its module_label()/make_grid_points()/
// make_integration_volumes() glue are pure host-side reads of a
// cosmosis::DataBlock (no CUDA models are touched until set_sample()), so
// they can and should be exercised directly here, without a real sample or
// a kernel launch.
namespace {
  cosmosis::DataBlock
  make_numcounts3d_cfg(std::size_t n_lam_min, std::size_t n_lam_max,
                       std::size_t n_zob_min, std::size_t n_zob_max,
                       std::size_t n_sigma_z)
  {
    cosmosis::DataBlock cfg;
    char const* mod = "NumCounts3dGpu";
    cfg.put_val(mod, "lam_min", std::vector<double>(n_lam_min, 20.0));
    cfg.put_val(mod, "lam_max", std::vector<double>(n_lam_max, 30.0));
    cfg.put_val(mod, "zob_min", std::vector<double>(n_zob_min, 0.2));
    cfg.put_val(mod, "zob_max", std::vector<double>(n_zob_max, 0.4));
    cfg.put_val(mod, "sigma_z", std::vector<double>(n_sigma_z, 0.02));
    return cfg;
  }
}

TEST_CASE("NumCounts3dGpu constructor validates bin definition arrays")
{
  SECTION("mismatched bin array lengths throw")
  {
    auto cfg = make_numcounts3d_cfg(2, 3, 2, 2, 2);
    CHECK_THROWS_AS(NumCounts3dGpu{cfg}, std::runtime_error);
  }
  SECTION("more than MAX_BINS (32) bins throws")
  {
    auto cfg = make_numcounts3d_cfg(33, 33, 33, 33, 33);
    CHECK_THROWS_AS(NumCounts3dGpu{cfg}, std::runtime_error);
  }
  SECTION("a well-formed bin set constructs cleanly")
  {
    auto cfg = make_numcounts3d_cfg(3, 3, 3, 3, 3);
    CHECK_NOTHROW(NumCounts3dGpu{cfg});
    CHECK(std::string(NumCounts3dGpu::module_label()) == "NumCounts3dGpu");
  }
}

TEST_CASE("NumCounts3dGpu::set_grid_point validates bin_index against the "
         "configured bin count")
{
  auto cfg = make_numcounts3d_cfg(2, 2, 2, 2, 2);
  NumCounts3dGpu integrand(cfg);

  CHECK_NOTHROW(integrand.set_grid_point({0.0}));
  CHECK_NOTHROW(integrand.set_grid_point({1.0}));
  CHECK_THROWS_AS(integrand.set_grid_point({-1.0}), std::out_of_range);
  CHECK_THROWS_AS(integrand.set_grid_point({2.0}), std::out_of_range);
}

TEST_CASE("NumCounts3dGpu grid/volume builders parse a wall-of-numbers "
         "configuration")
{
  cosmosis::DataBlock cfg;
  char const* mod = "NumCounts3dGpu";
  cfg.put_val(mod, "bin_index", std::vector<double>{0.0, 1.0});
  cfg.put_val(mod, "lt_low", std::vector<double>{0.0, 0.0});
  cfg.put_val(mod, "lt_high", std::vector<double>{200.0, 200.0});
  cfg.put_val(mod, "zt_low", std::vector<double>{0.1, 0.1});
  cfg.put_val(mod, "zt_high", std::vector<double>{0.9, 0.9});
  cfg.put_val(mod, "lnm_low", std::vector<double>{30.0, 30.0});
  cfg.put_val(mod, "lnm_high", std::vector<double>{36.0, 36.0});

  auto const grid = NumCounts3dGpu::make_grid_points(cfg);
  CHECK(grid.size() == 2u);
  auto const vols = NumCounts3dGpu::make_integration_volumes(cfg);
  CHECK(vols.size() == 2u);
}

namespace {
  // A synthetic-but-complete sample: everything NumCounts3dGpu::set_sample
  // reads (HMF_t, DV_DO_DZ_t, OMEGA_Z_DES, MOR_HOD_t, EMG_DES_t), same
  // role as dsigma_prj_3d_gpu.test.cu's make_sample() -- not a fiducial
  // physics pin, just internally-consistent inputs so set_sample() and
  // operator() actually run instead of only being exercised in isolation.
  cosmosis::DataBlock
  make_numcounts3d_sample()
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
    return db;
  }

  // NumCounts3dGpu holds quad::Interp2D-backed device models (HMF_t) that
  // copy their tables to device memory at construction and segfault if
  // evaluated directly from host code (same reason as
  // dsigma_prj_3d_gpu.test.cu) -- evaluate through a trivial kernel
  // instead, exactly as PAGANI itself invokes the integrand.
  __global__ void
  numcounts3d_eval_kernel(NumCounts3dGpu obj, double lt, double zt,
                          double lnM, double* out)
  {
    *out = obj(lt, zt, lnM);
  }

  double
  numcounts3d_eval_on_device(NumCounts3dGpu const& obj, double lt, double zt,
                             double lnM)
  {
    double* d_out = nullptr;
    REQUIRE(cudaMalloc(&d_out, sizeof(double)) == cudaSuccess);
    numcounts3d_eval_kernel<<<1, 1>>>(obj, lt, zt, lnM, d_out);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    double out = 0.0;
    REQUIRE(cudaMemcpy(&out, d_out, sizeof(double), cudaMemcpyDeviceToHost) ==
           cudaSuccess);
    cudaFree(d_out);
    return out;
  }
}

TEST_CASE("NumCounts3dGpu::set_sample builds its device models and "
         "operator() returns a finite, non-negative integrand value")
{
  auto cfg = make_numcounts3d_cfg(1, 1, 1, 1, 1);
  NumCounts3dGpu integrand(cfg);

  auto sample = make_numcounts3d_sample();
  CHECK_NOTHROW(integrand.set_sample(sample));
  integrand.set_grid_point({0.0});

  double const lt = 20.0, zt = 0.3, lnM = std::log(1.0e14);
  double const val = numcounts3d_eval_on_device(integrand, lt, zt, lnM);
  CHECK(std::isfinite(val));
  CHECK(val >= 0.0);
}
