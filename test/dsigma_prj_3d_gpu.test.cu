#include "catch2/catch.hpp"
#include "cosmosis/datablock/datablock.hh"
#include "cosmosis/datablock/ndarray.hh"

// This is the code we're actually testing.
#include "pipelines/des_y3/shear_projection/cuda/3d/dsigma_prj_3d_gpu_t.cuh"

#include "models/nfw_dsigma_mis.cuh"

#include <array>
#include <cmath>
#include <vector>

namespace {

  // quad::Interp1D/Interp2D copy their tables to device memory at
  // construction (set_sample()); their clamp() dereferences those device
  // pointers, which segfaults if called directly from host code. Evaluate
  // the integrand through a trivial kernel instead, mirroring how PAGANI
  // itself invokes the integrand (by-value copy to the device).
  __global__ void
  eval_kernel(DSigmaPrj3dGpu obj, double lntheta, double zt,
             double lnM, double* out)
  {
    *out = obj(lntheta, zt, lnM);
  }

  double
  eval_on_device(DSigmaPrj3dGpu const& obj, double lntheta, double zt,
                double lnM)
  {
    double* d_out = nullptr;
    cudaMalloc(&d_out, sizeof(double));
    eval_kernel<<<1, 1>>>(obj, lntheta, zt, lnM, d_out);
    cudaError_t const err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
      throw std::runtime_error(std::string("eval_kernel failed: ") +
                              cudaGetErrorString(err));
    double result = 0.0;
    cudaMemcpy(&result, d_out, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    return result;
  }

  // A tiny, internally-consistent synthetic sample: every 2D table (hmf,
  // bias, xi_nl) is constant-valued so bilinear interpolation cannot
  // introduce any orientation/layout ambiguity, and every 1D table is
  // queried exactly at a grid node so its interpolated value is exact.
  // This isolates what this test actually checks -- that operator()
  // assembles the documented dSigma_prj integrand formula correctly --
  // from the (separately, exhaustively) validated correctness of the
  // interpolators and NFW_DSIGMA_MIS themselves.
  cosmosis::DataBlock
  make_sample()
  {
    cosmosis::DataBlock db;
    db.put_val("cosmological_parameters", "h0", 0.7);
    db.put_val("cosmological_parameters", "omega_m", 0.3);
    db.put_val("cosmological_parameters", "omega_M", 0.3);
    db.put_val("cosmological_parameters", "omega_nu", 0.0);
    db.put_val("cosmological_parameters", "omega_lambda", 0.7);
    db.put_val("cosmological_parameters", "omega_k", 0.0);

    // distances: z=0.3 is an exact grid node (index 1).
    db.put_val("distances", "z", std::vector<double>{0.0, 0.3, 0.6});
    db.put_val("distances", "d_c", std::vector<double>{0.0, 1150.0, 2100.0});
    db.put_val("distances", "d_a",
              std::vector<double>{0.0, 1150.0 / 1.3, 2100.0 / 1.6});

    // mass_function / cluster_abundance: constant dn/dlnM grid -> HMF_t's
    // interpolated nmz factor is 2.0 everywhere in range.
    db.put_val("mass_function", "m_h", std::vector<double>{1.0e13, 1.0e16});
    db.put_val("mass_function", "z", std::vector<double>{0.0, 0.6});
    db.put_val(
      "mass_function", "dndlnmh",
      cosmosis::ndarray<double>(std::vector<double>{2.0, 2.0, 2.0, 2.0},
                                {2, 2}));
    db.put_val("cluster_abundance", "hmf_s", 1.0);
    db.put_val("cluster_abundance", "hmf_q", 0.0);

    // haloModel bias: constant 2.0.
    db.put_val("haloModel", "lnM", std::vector<double>{20.0, 40.0});
    db.put_val("haloModel", "rho_m_ref", 0.3 * 2.77533742639e+11);
    db.put_val("haloModel", "z", std::vector<double>{0.0, 0.6});
    db.put_val("haloModel", "bias",
              cosmosis::ndarray<double>(std::vector<double>{2.0, 2.0, 2.0, 2.0},
                                        {2, 2}));

    // xi_nl: constant 0.5.
    db.put_val("xi_nl", "r", std::vector<double>{0.1, 5000.0});
    db.put_val("xi_nl", "z", std::vector<double>{0.0, 0.6});
    db.put_val("xi_nl", "xi_nl",
              cosmosis::ndarray<double>(std::vector<double>{0.5, 0.5, 0.5, 0.5},
                                        {2, 2}));

    // b_sel_marginalised: one exact wall row per lambda bin.  The test wall
    // below uses zo=[0,0.6], so zob=0.3 and must match this row exactly.
    db.put_val("b_sel_marginalised", "lambda_bin",
              std::vector<int>{0, 1, 2, 3});
    db.put_val("b_sel_marginalised", "zo_low",
              std::vector<double>{0.0, 0.0, 0.0, 0.0});
    db.put_val("b_sel_marginalised", "zo_high",
              std::vector<double>{0.6, 0.6, 0.6, 0.6});
    db.put_val("b_sel_marginalised", "zob",
              std::vector<double>{0.3, 0.3, 0.3, 0.3});
    db.put_val("b_sel_marginalised", "lob",
              std::vector<double>{25.0, 37.5, 52.5, 130.0});
    db.put_val(
      "b_sel_marginalised", "b_small",
      std::vector<double>{1.0, 1.0, 1.0, 1.0});
    db.put_val(
      "b_sel_marginalised", "b_large",
      std::vector<double>{3.0, 3.0, 3.0, 3.0});
    return db;
  }

}

TEST_CASE("DSigmaPrj3dGpu integrand is exactly zero outside the photo-z window")
{
  cosmosis::DataBlock db = make_sample();
  DSigmaPrj3dGpu integrand(db);
  integrand.set_sample(db);

  // zob = 0.3 from zo=[0,0.6]; lob_bin = 0; R = 0.3 cMpc/h.
  std::array<double, 4> const pt{0.0, 0.0, 0.6, 0.3};
  integrand.set_grid_point(pt);

  // zt = 5.0 is many photo-z widths away from zob = 0.3 for any
  // realistic sigma_z(z) -- |u| = |zt - zob| / sigma_z >= 1, so the
  // integrand must short-circuit to 0.0 before touching bias/xi_nl/hmf.
  double const lnM = std::log(1.0e14);
  CHECK(eval_on_device(integrand, std::log(0.05), 5.0, lnM) == 0.0);
}

TEST_CASE("DSigmaPrj3dGpu integrand matches its documented formula assembly")
{
  cosmosis::DataBlock db = make_sample();
  DSigmaPrj3dGpu integrand(db);
  integrand.set_sample(db);

  double const zob = 0.3;
  double const R = 0.3;
  std::array<double, 4> const pt{0.0, 0.0, 0.6, R};  // lob_bin = 0
  integrand.set_grid_point(pt);

  double const theta = 0.05;     // well outside the ~1.2e-3 rad exclusion
                                  // ring at this (zob, lob_bin) sample.
  double const zt = zob;         // u = 0 exactly -> w_pz = 1 exactly,
                                  // independent of the compiled sigma_z
                                  // table's actual value.
  double const lnM = std::log(1.0e14);

  double const h0 = 0.7, omega_m = 0.3, omega_l = 0.7, omega_k = 0.0;
  double const da_zt = 1150.0 / 1.3;          // exact grid node at z=0.3
  double const ez = std::sqrt(omega_m * (1.0 + zt) * (1.0 + zt) * (1.0 + zt)
                              + omega_k * (1.0 + zt) * (1.0 + zt) + omega_l);
  double const dv_do_dz =
    2997.92 * (1.0 + zt) * (1.0 + zt) * da_zt * h0 * da_zt * h0 / ez;
  double const w_pz = 1.0;
  double const hmf = 2.0 * (std::log10(1.0e14) - 13.8124426028);

  double const chi_o = (1150.0) * h0;          // exact grid node at zob=0.3
  double const d_a_o = chi_o / (1.0 + zob);
  double const r_excl = std::pow(25.0 / 100.0, 0.2) * (1.0 + zob);
  double const theta_lam = std::pow(25.0 / 100.0, 0.2) * (1.0 + zob) / chi_o;
  double const k_sig = 2.5 / theta_lam;
  double const theta0 = 0.5 * theta_lam;
  double const cos_ex = (2.0 * chi_o * chi_o - r_excl * r_excl)
                       / (2.0 * chi_o * chi_o);
  double const theta_excl = std::acos(std::max(-1.0, std::min(1.0, cos_ex)));
  REQUIRE(theta > theta_excl);   // confirms this point exercises the
                                 // clustered (cl != 0) branch, not the
                                 // exclusion short-circuit.
  double const bsel =
    1.0 + (3.0 - 1.0) / (1.0 + std::exp(-k_sig * (theta - theta0)));
  double const bias = 2.0, xi_nl = 0.5;
  double const cl = bias * bsel * xi_nl;

  double const dchi = std::sqrt(2.0 * chi_o * chi_o * (1.0 - std::cos(theta)));
  (void)dchi;   // recomputed inside the integrand from (chi_z, chi_o, theta);
               // exposed here only to document the geometry being probed.

  // UNIFIED rho_m convention: the module reads haloModel/rho_m_ref and
  // the profile carries it for boundary AND amplitude (no external
  // omega_m factor in the integrand any more).
  y3_cuda::NFW_DSIGMA_MIS dsmis_model(4.0, 2.77533742639e+11, "single");
  dsmis_model.set_rho_ref(0.3 * 2.77533742639e+11);
  double const dsmis = dsmis_model(R, theta * d_a_o, lnM);

  double const expected = theta * 2.0 * M_PI * std::sin(theta) * dv_do_dz
                         * w_pz * hmf * (1.0 + cl) * dsmis;

  double const got = eval_on_device(integrand, std::log(theta), zt, lnM);
  CHECK(got == Approx(expected).epsilon(1.0e-6));
}
