// Unit tests for the tagged Bsel kernels and BselModels dispatcher.
#include "catch2/catch.hpp"

#include "cosmosis/datablock/datablock.hh"
#include "pipelines/systematics/selection_boost/Bsel.hh"

#include <gsl/gsl_integration.h>

#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

using y3_cluster::BselCostanzi26;
using y3_cluster::BselDeltaSigmaOperator;
using y3_cluster::BselModels;
using y3_cluster::BselSunayama23;
using y3_cluster::Shear1h2hMaxSelCostanzi26;
using y3_cluster::Shear1h2hMaxSelSunayama23;

TEST_CASE("BselCostanzi26 evaluates the profile and derivative")
{
  BselCostanzi26 model(0.10, 0.92, -0.53, 4.1);
  double const R = 1.0;
  double const lob = 40.0;
  double const z = 0.3;
  double const eps = 1.0e-6;
  double const numerical = (model(R + eps, lob, z) - model(R - eps, lob, z)) /
                           (2.0 * eps);
  CHECK(model.derivative(R, lob, z) == Approx(numerical).epsilon(1e-8));
  CHECK(model(0.0, lob, z) == 1.0);
}

TEST_CASE("BselCostanzi26 agrees with high-precision independent pins")
{
  // Python test computes these from the Appendix C form with 80-digit
  // Decimal arithmetic; the same pins check C++/Python numerical parity.
  BselCostanzi26 const model(0.10, 0.92, -0.53, 4.1);
  struct Pin { double R, value, derivative; };
  Pin const pins[] = {
    {0.0001, 1.000019426433576195, 0.178723188900992436},
    {0.5,    1.048428670772021167, 0.083426960803439279},
    {1.0,    1.076705775466793476, 0.023897803378583388},
    {3.0,    1.057943146945476006, -0.009814620432412978},
    {10.0,   1.030774532373601660, -0.001630560021010801}
  };
  for (auto const& pin : pins) {
    INFO("R=" << pin.R);
    CHECK(model(pin.R, 40.0, 0.3) == Approx(pin.value).epsilon(3e-15));
    CHECK(model.derivative(pin.R, 40.0, 0.3) ==
          Approx(pin.derivative).epsilon(2e-11));
  }
}

TEST_CASE("Bsel kernels match benchmark values at ten logarithmic radii")
{
  // Fixed reference values for R=logspace(-1, 1, 10), generated from the
  // independent Python implementations.  This checks the full radial range
  // used by the profile calculation, including both Sunayama branches.
  std::array<double, 10> const radii = {
    0.1, 0.16681005372000587, 0.27825594022071243,
    0.46415888336127786, 0.774263682681127,
    1.2915496650148841, 2.1544346900318834,
    3.5938136638046259, 5.9948425031894086, 10.0
  };
  std::array<double, 10> const costanzi_values = {
    1.0111785186792333, 1.01789639910452, 1.0286218236572273,
    1.0453968273535741, 1.0678416280469756, 1.0791840601769058,
    1.0680253815140701, 1.0528016345802402, 1.0403504761945139,
    1.0307745323736017
  };
  std::array<double, 10> const costanzi_derivatives = {
    0.10283306354875656, 0.09863043480487986, 0.09406590730077931,
    0.08570538684808919, 0.05493575001600590, -0.00347956712456455,
    -0.01416523906054845, -0.00763264961463009, -0.00355862842257145,
    -0.00163056002101080
  };
  std::array<double, 10> const sunayama_values = {
    1.0128571428571429, 1.021447006906858, 1.035775763742663,
    1.0596775707178785, 1.0995481877732878, 1.1660563855019137,
    1.2769987458612422, 1.447354902794312, 1.3961863451722221,
    1.3450177875501321
  };
  std::array<double, 10> const sunayama_derivatives = {
    0.12857142857142859, 0.12857142857142859, 0.12857142857142859,
    0.12857142857142859, 0.12857142857142859, 0.12857142857142859,
    0.12857142857142859, -0.02782559402207126, -0.01668100537200059,
    -0.01
  };

  BselCostanzi26 const costanzi(0.10, 0.92, -0.53, 4.1);
  BselSunayama23 const sunayama({0.45}, {3.5}, -0.1, {0.0});
  for (std::size_t i = 0; i != radii.size(); ++i) {
    INFO("R=" << radii[i]);
    CHECK(costanzi(radii[i], 40.0, 0.3) ==
          Approx(costanzi_values[i]).epsilon(2e-14));
    CHECK(costanzi.derivative(radii[i], 40.0, 0.3) ==
          Approx(costanzi_derivatives[i]).epsilon(2e-13));
    CHECK(sunayama(radii[i], 0) ==
          Approx(sunayama_values[i]).epsilon(2e-14));
    CHECK(sunayama.derivative(radii[i], 0) ==
          Approx(sunayama_derivatives[i]).epsilon(2e-14));
  }
}

TEST_CASE("BselCostanzi26 rejects invalid parameters and coordinates")
{
  double const nan = std::numeric_limits<double>::quiet_NaN();
  CHECK_THROWS_AS(BselCostanzi26(nan, 0.92, -0.53, 4.1),
                  std::invalid_argument);
  CHECK_THROWS_AS(BselCostanzi26(0.1, 0.92, -0.53, 0.0),
                  std::invalid_argument);
  BselCostanzi26 const model(0.1, 0.92, -0.53, 4.1);
  CHECK_THROWS_AS(model(-1.0, 40.0, 0.3), std::invalid_argument);
  CHECK_THROWS_AS(model(nan, 40.0, 0.3), std::invalid_argument);
  CHECK_THROWS_AS(model(1.0, 0.0, 0.3), std::invalid_argument);
  CHECK_THROWS_AS(model(0.0, 0.0, 0.3), std::invalid_argument);
  CHECK_THROWS_AS(model(1.0, 40.0, -1.0), std::invalid_argument);
  CHECK_THROWS_AS(model.derivative(0.0, 40.0, 0.3),
                  std::invalid_argument);
}

TEST_CASE("BselSunayama23 evaluates both branches")
{
  BselSunayama23 model({1.2, 1.4}, {1.0, 2.0}, 0.1, {2.0, 5.0});
  CHECK(model(0.5, 2) == Approx(1.6));
  CHECK(model(1.0, 2) == Approx(2.2));
  CHECK(model(4.0, 5) == Approx(2.4 + 0.1 * std::log(2.0)));
  CHECK_THROWS(model(1.0, 9));
}

TEST_CASE("BselSunayama23 is 1+Pi across the branch and boundary")
{
  BselSunayama23 const model({1.2, 1.4}, {1.0, 2.0}, 0.1, {2.0, 5.0});
  CHECK(model(0.0, 2) == 1.0);
  CHECK(model(0.5, 2) == Approx(1.6).epsilon(2e-15));
  CHECK(model(1.0, 2) == Approx(2.2).epsilon(2e-15));
  CHECK(model(1.0 + 1e-12, 2) ==
        Approx(2.2 + 0.1 * std::log1p(1e-12)).epsilon(2e-15));
  CHECK(model(4.0, 5) ==
        Approx(2.4 + 0.1 * std::log(2.0)).epsilon(2e-15));
  CHECK(model.derivative(1.0, 2) == Approx(1.2).epsilon(2e-15));
  CHECK(model.derivative(1.0 + 1e-12, 2) ==
        Approx(0.1 / (1.0 + 1e-12)).epsilon(2e-15));
  CHECK(model.derivative(2.0, 5) == Approx(0.7).epsilon(2e-15));
}

TEST_CASE("BselSunayama23 rejects invalid inputs and fractional labels")
{
  double const nan = std::numeric_limits<double>::quiet_NaN();
  CHECK_THROWS_AS(BselSunayama23({}, {}, 0.1), std::invalid_argument);
  CHECK_THROWS_AS(BselSunayama23({1.2}, {0.0}, 0.1),
                  std::invalid_argument);
  CHECK_THROWS_AS(BselSunayama23({1.2}, {1.0}, nan),
                  std::invalid_argument);
  BselSunayama23 const model({1.2}, {1.0}, 0.1, {5.0});
  CHECK_THROWS_AS(model(-1.0, 5), std::invalid_argument);
  CHECK_THROWS_AS(model(nan, 5), std::invalid_argument);
  CHECK_THROWS_AS(model.derivative(0.0, 5), std::invalid_argument);
  CHECK_THROWS(BselSunayama23({1.2}, {1.0}, 0.1, {2.5}));
  CHECK_THROWS(BselSunayama23({1.2, 1.4}, {1.0, 2.0}, 0.1,
                              {2.0, 2.0}));
}

TEST_CASE("BselModels dispatches explicit tags and reads DataBlock values")
{
  cosmosis::DataBlock source;
  source.put_val("boost_selection_costanzi26", "A", 0.10);
  source.put_val("boost_selection_costanzi26", "alpha", 0.92);
  source.put_val("boost_selection_costanzi26", "beta", -0.53);
  source.put_val("boost_selection_costanzi26", "gamma", 4.1);
  BselModels const model = BselModels::from_datablock(
    source, "Costanzi26", "boost_selection_costanzi26");
  CHECK(model(1.0, 0, 40.0, 0.3) > 1.0);
  CHECK_THROWS(BselModels::parse_tag("unknown"));

  source.put_val("custom_sunayama", "pi0", std::vector<double>{1.2, 1.4});
  source.put_val("custom_sunayama", "r0", std::vector<double>{1.0, 2.0});
  source.put_val("custom_sunayama", "c", 0.1);
  source.put_val("custom_sunayama", "lambda_bin",
                 std::vector<double>{2.0, 5.0});
  BselModels const sunayama = BselModels::from_datablock(
    source, "Sunayama23", "custom_sunayama");
  CHECK(sunayama(4.0, 5, 0.0, 0.0) ==
        Approx(2.4 + 0.1 * std::log(2.0)).epsilon(2e-15));
  CHECK(sunayama.derivative(4.0, 5, 0.0, 0.0) ==
        Approx(0.025).epsilon(2e-15));
  CHECK_THROWS(BselModels::from_datablock(
    source, "Sunayama23", "missing_section"));
}

TEST_CASE("Selected Shear1h2hMax consumers preserve local profile outputs")
{
  double const profile = 7.0;
  BselCostanzi26 const costanzi(0.10, 0.92, -0.53, 4.1);
  BselSunayama23 const sunayama({0.45}, {3.5}, -0.1, {4.0});
  Shear1h2hMaxSelCostanzi26 const selected_costanzi(costanzi);
  Shear1h2hMaxSelSunayama23 const selected_sunayama(sunayama);

  CHECK(selected_costanzi(profile, 1.0, 4, 40.0, 0.3) ==
        Approx(profile * costanzi(1.0, 40.0, 0.3)));
  CHECK(selected_sunayama(profile, 1.0, 4, 40.0, 0.3) ==
        Approx(profile * sunayama(1.0, 4)));
}

TEST_CASE("BselDeltaSigmaOperator integrates analytic linear branch to machine precision")
{
  BselSunayama23 const sunayama({0.45}, {3.5}, -0.1, {0.0});
  BselModels const model(sunayama);
  BselDeltaSigmaOperator const op(model, 20);

  double const sigma0 = 50.0;
  double const delta_sigma = 100.0;
  double const R = 1.5; // R < r0 = 3.5

  auto mean_sigma_fn = [sigma0](double) { return sigma0; };

  double const expected_b = 1.0 + 0.45 * (R / 3.5);
  double const expected_msel_over_pi_r2 = (1.0 / 3.0) * sigma0 * (0.45 / 3.5) * R;
  double const expected_dsigma_sel = expected_b * delta_sigma - expected_msel_over_pi_r2;

  double const result = op.apply(delta_sigma, mean_sigma_fn, R, 0, 0.0, 0.0);
  CHECK(result == Approx(expected_dsigma_sel).epsilon(1e-14));
}

TEST_CASE("BselDeltaSigmaOperator flat limit reproduces DeltaSigma exactly")
{
  BselSunayama23 const sunayama_flat({0.0}, {3.5}, 0.0, {0.0});
  BselModels const model_flat(sunayama_flat);
  BselDeltaSigmaOperator const op(model_flat);
  double const delta_sigma = 123.456;
  double const res = op.apply(delta_sigma, [](double) { return 50.0; }, 2.0, 0, 0.0, 0.0);
  CHECK(res == Approx(delta_sigma).epsilon(1e-15));
}

TEST_CASE("BselDeltaSigmaOperator validates positive radius and n_gl")
{
  BselCostanzi26 const costanzi(0.10, 0.92, -0.53, 4.1);
  BselModels const model(costanzi);
  BselDeltaSigmaOperator const op(model);

  CHECK_THROWS_AS(op.apply(10.0, [](double) { return 50.0; }, 0.0, 0, 40.0, 0.3),
                  std::invalid_argument);
  CHECK_THROWS_AS(op.apply(10.0, [](double) { return 50.0; }, -1.0, 0, 40.0, 0.3),
                  std::invalid_argument);
  CHECK_THROWS_AS(BselDeltaSigmaOperator(model, 0), std::invalid_argument);
}

TEST_CASE("BselDeltaSigmaOperator loads correctly from DataBlock")
{
  cosmosis::DataBlock source;
  source.put_val("boost_selection_costanzi26", "A", 0.10);
  source.put_val("boost_selection_costanzi26", "alpha", 0.92);
  source.put_val("boost_selection_costanzi26", "beta", -0.53);
  source.put_val("boost_selection_costanzi26", "gamma", 4.1);

  auto op_costanzi = BselDeltaSigmaOperator::from_datablock(source, "Costanzi26");
  CHECK(op_costanzi.model().tag() == BselModels::Tag::Costanzi26);

  source.put_val("bsel_profile_sunayama23", "pi0", std::vector<double>{0.45});
  source.put_val("bsel_profile_sunayama23", "r0", std::vector<double>{3.5});
  source.put_val("bsel_profile_sunayama23", "c", -0.1);
  source.put_val("bsel_profile_sunayama23", "lambda_bin", std::vector<double>{0.0});

  auto op_sunayama = BselDeltaSigmaOperator::from_datablock(source, "Sunayama23");
  CHECK(op_sunayama.model().tag() == BselModels::Tag::Sunayama23);
}

TEST_CASE("BselDeltaSigmaOperator matches direct GSL numerical quadrature")
{
  // Analytic smooth profile: barSigma(<r) = Sigma0 / (1 + r / rc)
  // Sigma(r) = Sigma0 * (2 + r/rc) / [2 (1 + r/rc)^2]
  // DeltaSigma(r) = Sigma0 * (r/rc) / [2 (1 + r/rc)^2]
  double const sigma0 = 100.0;
  double const rc = 1.5;

  auto bar_sigma = [sigma0, rc](double r) {
    return sigma0 / (1.0 + r / rc);
  };
  auto sigma = [sigma0, rc](double r) {
    double const x = r / rc;
    return sigma0 * (2.0 + x) / (2.0 * (1.0 + x) * (1.0 + x));
  };
  auto delta_sigma = [sigma0, rc](double r) {
    double const x = r / rc;
    return sigma0 * x / (2.0 * (1.0 + x) * (1.0 + x));
  };

  BselCostanzi26 const costanzi(0.03, 0.92, -0.53, 4.1);
  BselDeltaSigmaOperator const op_c(BselModels(costanzi), 20);

  BselSunayama23 const sunayama({0.45}, {3.5}, -0.1, {0.0});
  BselDeltaSigmaOperator const op_s(BselModels(sunayama), 20);

  double const lob = 25.0;
  double const z = 0.275;

  gsl_integration_workspace* ws = gsl_integration_workspace_alloc(1000);

  std::vector<double> const radii = {0.2, 0.5, 1.0, 2.0, 3.5, 5.0};
  for (double const R : radii) {
    // 1. Costanzi26
    struct ParamsC {
      BselCostanzi26 const* model;
      decltype(sigma) const* sig_fn;
      double lob, z;
    } params_c{&costanzi, &sigma, lob, z};

    gsl_function F_c;
    F_c.function = [](double r, void* p) -> double {
      auto const* p_c = static_cast<ParamsC*>(p);
      return r * (*p_c->model)(r, p_c->lob, p_c->z) * (*p_c->sig_fn)(r);
    };
    F_c.params = &params_c;

    double quad_val_c = 0.0, err_c = 0.0;
    gsl_integration_qag(&F_c, 0.0, R, 1e-12, 1e-9, 1000, GSL_INTEG_GAUSS21, ws,
                        &quad_val_c, &err_c);
    double const bds_direct_c =
      (2.0 / (R * R)) * quad_val_c - costanzi(R, lob, z) * sigma(R);
    double const bds_op_c =
      op_c.apply(delta_sigma(R), bar_sigma, R, 0, lob, z);

    CHECK(bds_op_c == Approx(bds_direct_c).epsilon(1e-6));

    // 2. Sunayama23 for R <= r0 (r0 = 3.5)
    if (R <= 3.5) {
      struct ParamsS {
        BselSunayama23 const* model;
        decltype(sigma) const* sig_fn;
      } params_s{&sunayama, &sigma};

      gsl_function F_s;
      F_s.function = [](double r, void* p) -> double {
        auto const* p_s = static_cast<ParamsS*>(p);
        return r * (*p_s->model)(r, 0) * (*p_s->sig_fn)(r);
      };
      F_s.params = &params_s;

      double quad_val_s = 0.0, err_s = 0.0;
      gsl_integration_qag(&F_s, 0.0, R, 1e-12, 1e-9, 1000, GSL_INTEG_GAUSS21,
                          ws, &quad_val_s, &err_s);
      double const bds_direct_s =
        (2.0 / (R * R)) * quad_val_s - sunayama(R, 0) * sigma(R);
      double const bds_op_s =
        op_s.apply(delta_sigma(R), bar_sigma, R, 0, 0.0, 0.0);

      CHECK(bds_op_s == Approx(bds_direct_s).epsilon(1e-6));
    }
  }

  gsl_integration_workspace_free(ws);
}

