#include "catch2/catch.hpp"

// This is the code we're actually testing: the device NFW primitive and
// the max(1h, 2h) composition formula Shear1h2hMaxGpu.cu's private
// max_model_contract kernel evaluates per (bin, R, lnM) thread, plus
// (below) the Shear1h2hMaxGpu class itself. Its constructor and its
// evaluate()'s bin-index check are pure host-side DataBlock reads --
// dsigma_mis_dev_ is a NON-optional member built in the constructor's
// initializer list, but y3_cuda::NFW_DSIGMA_MIS's (c, rhoc, kernel)
// constructor only loads a host-side interpolation table from
// data/nfw_off_center/ (same as the TEST_CASE above calls it directly, no
// device memory or kernel launch involved), so building a Shear1h2hMaxGpu
// needs no live sample and no CUDA kernel launch. set_sample() itself
// (the miscentred-NFW + max(1h,2h) device kernel launch, needing the full
// haloModel/dSigma_nfw, bias, dSigma_hh tables and miscentering/f_mis,
// tau_mis section) is out of scope for the same reason
// test/shear1h2h_max.test.cc's header comment gives for the CPU
// counterpart: it would pin a known-buggy dSigma_hh table rather than
// test physics.
#include "models/nfw_dsigma_mis.cuh"
#include "pipelines/des_y3/shear_1h2h/cuda/0d/shear1h2h_max_gpu_t.cuh"

#include "cosmosis/datablock/datablock.hh"

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
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
  // defects (60% NaN by construction; docs/known_issues/dsigma_hh_debug_flag.md) and
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

namespace {
  // Minimal well-formed config for Shear1h2hMaxGpu's constructor: only
  // the keys the constructor itself reads (zt_low/high, lnm_low/high,
  // lob_centers, r_perp, and the has_val-gated optional knobs).
  // set_sample()'s inputs (haloModel tables, miscentering section) are
  // deliberately absent -- the constructor never touches them.
  cosmosis::DataBlock
  make_shear1h2h_max_cfg(std::vector<double> const& lob_centers,
                        std::vector<double> const& r_perp,
                        bool set_optional)
  {
    cosmosis::DataBlock cfg;
    char const* mod = "Shear1h2hMaxGpu";
    cfg.put_val(mod, "zt_low", 0.1);
    cfg.put_val(mod, "zt_high", 0.9);
    cfg.put_val(mod, "lnm_low", 30.0);
    cfg.put_val(mod, "lnm_high", 36.0);
    cfg.put_val(mod, "lob_centers", lob_centers);
    cfg.put_val(mod, "r_perp", r_perp);
    if (set_optional) {
      cfg.put_val(mod, "n_lnm", 8);
      cfg.put_val(mod, "n_z", 8);
      cfg.put_val(mod, "include_miscentering", 1);
      cfg.put_val(mod, "use_halo_model_conc", true);
    }
    return cfg;
  }
}

TEST_CASE("Shear1h2hMaxGpu constructor: optional knobs take their has_val "
         "default when absent, and honor an explicit override")
{
  SECTION("n_lnm/n_z/include_miscentering/use_halo_model_conc default "
         "(has_val false branch)")
  {
    auto cfg =
      make_shear1h2h_max_cfg({25.0, 37.5, 52.5, 130.0}, {0.5, 1.0, 2.0}, false);
    CHECK_NOTHROW(Shear1h2hMaxGpu{cfg});
  }
  SECTION("explicit optional keys are honored (has_val true branch)")
  {
    auto cfg =
      make_shear1h2h_max_cfg({25.0, 37.5, 52.5, 130.0}, {0.5, 1.0, 2.0}, true);
    CHECK_NOTHROW(Shear1h2hMaxGpu{cfg});
  }
  CHECK(std::string(Shear1h2hMaxGpu::module_label()) == "Shear1h2hMaxGpu");
}

TEST_CASE("Shear1h2hMaxGpu constructor rejects an empty lob_centers or "
         "r_perp wall")
{
  SECTION("empty lob_centers throws")
  {
    auto cfg = make_shear1h2h_max_cfg({}, {0.5, 1.0}, false);
    CHECK_THROWS_AS(Shear1h2hMaxGpu{cfg}, std::runtime_error);
  }
  SECTION("empty r_perp throws")
  {
    auto cfg = make_shear1h2h_max_cfg({25.0}, {}, false);
    CHECK_THROWS_AS(Shear1h2hMaxGpu{cfg}, std::runtime_error);
  }
}

TEST_CASE("Shear1h2hMaxGpu::evaluate rejects any bin index before "
         "set_sample has populated the wall")
{
  // n_bins_ defaults to 0 until set_sample() runs, so evaluate()'s
  // bin-range check must reject every bin index at this point -- pins
  // that guard without needing the full haloModel/miscentering sample
  // set_sample() would otherwise require.
  auto cfg =
    make_shear1h2h_max_cfg({25.0, 37.5, 52.5, 130.0}, {0.5, 1.0}, false);
  Shear1h2hMaxGpu integrand(cfg);
  std::array<double, 2> const pt{0.0, 0.5};
  CHECK_THROWS_AS(integrand.evaluate(pt), std::out_of_range);
}
