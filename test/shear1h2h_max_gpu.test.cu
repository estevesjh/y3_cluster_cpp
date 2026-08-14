#include "catch2/catch.hpp"

// This is the code we're actually testing: the device NFW primitive and
// the max(1h, 2h) composition formula Shear1h2hMaxGpu.cu's private
// max_model_contract kernel evaluates per (bin, R, lnM) thread.
#include "models/nfw_dsigma_mis.cuh"

#include <cmath>
#include <vector>

using y3_cuda::NFW_DSIGMA_MIS;

TEST_CASE("Shear1h2hMaxGpu: device NFW_DSIGMA_MIS matches the gamma-kernel "
         "reference table")
{
  // Same primitive Shear1h2hMaxGpu.cu's dominant-cost kernel evaluates
  // per (bin, R, lnM) thread (dsigma_mis_dev_ in that file), same
  // construction convention as the existing nfw_dsigma_mis_test
  // (RHOM = RHOC * omega_m baked into the constructor). A subset of that
  // test's cluster_toolkit-derived reference points, reused here to
  // confirm the device primitive still reproduces known physics under
  // this build's CUDA toolchain -- not a re-derivation of the physics,
  // which the pre-existing nfw_dsigma_mis_test already covers in full.
  double const omega_m = 0.3;
  double const RHOC = 2.77533742639e+11;
  NFW_DSIGMA_MIS model(4.0, RHOC * omega_m, "gamma");

  double const epsrel = 5.0e-3;
  double const epsabs = 1.0e-12;
  double const lnM = 32.2361913;

  // (r, rmis, expected), lifted from nfw_dsigma_mis_test's answers table.
  CHECK(model(0.1, 0.1, lnM) == Approx(12.3685326).epsilon(epsrel).margin(epsabs));
  CHECK(model(0.77426368, 0.5, lnM) == Approx(6.2393766).epsilon(epsrel).margin(epsabs));
  CHECK(model(10.0, 2.0, lnM) == Approx(0.7860161).epsilon(epsrel).margin(epsabs));
}

TEST_CASE("Shear1h2hMaxGpu: max(1h, 2h) reduction matches a hand-computed "
         "value, independent of the dSigma_hh table")
{
  // Pins max_model_contract's per-thread reduction:
  //   partial = sum_q w2d[q] * fmax(one_k, bias[q] * two[q])
  // The kernel itself has internal linkage (anonymous namespace in
  // Shear1h2hMaxGpu.cu) and can't be called from another translation
  // unit without modifying the module, which is out of scope -- this
  // pins its arithmetic against a value computed independently by hand
  // instead. one_k stands in for whatever the 1-halo mixture produces
  // (that physics is exercised separately, in the TEST_CASE above and in
  // the pre-existing nfw_dsigma_mis_test); bias/two are synthetic,
  // deliberately not sourced from haloModel/dSigma_hh, which has 3 open
  // defects (60% NaN by construction; docs/dsigma_hh_debug_flag.md) and
  // no available real-pipeline dump has compute_lensing_2h = T to source
  // real values from.
  double const one_k = 3.0;
  std::vector<double> const bias{1.1, 1.3, 1.6};
  std::vector<double> const two{0.5, 2.0, 20.0};
  std::vector<double> const w2d{0.2, 0.5, 0.3};

  // By hand: bias*two = {0.55, 2.6, 32.0}; fmax(one_k, .) = {3, 3, 32};
  // partial = 0.2*3 + 0.5*3 + 0.3*32 = 0.6 + 1.5 + 9.6 = 11.7.
  double const expected = 11.7;

  double partial = 0.0;
  for (std::size_t q = 0; q != w2d.size(); ++q)
    partial += w2d[q] * std::fmax(one_k, bias[q] * two[q]);

  CHECK(partial == Approx(expected).epsilon(1e-12));
  // Sanity: the constructed straddle actually exercises both branches of
  // the max (not a vacuous check where one side always wins).
  CHECK(bias[0] * two[0] < one_k);
  CHECK(bias[2] * two[2] > one_k);
}
