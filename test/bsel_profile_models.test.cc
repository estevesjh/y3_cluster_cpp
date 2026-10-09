// Unit tests for the tagged Bsel kernels and BselModels dispatcher.
#include "catch2/catch.hpp"

#include "cosmosis/datablock/datablock.hh"
#include "systematics/bsel_profile.hh"

#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

using y3_cluster::BselCostanzi26;
using y3_cluster::BselModels;
using y3_cluster::BselSunayama23;

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
  source.put_val("bsel_profile_costanzi26", "A", 0.10);
  source.put_val("bsel_profile_costanzi26", "alpha", 0.92);
  source.put_val("bsel_profile_costanzi26", "beta", -0.53);
  source.put_val("bsel_profile_costanzi26", "gamma", 4.1);
  BselModels const model = BselModels::from_datablock(
    source, "Costanzi26", "bsel_profile_costanzi26");
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
