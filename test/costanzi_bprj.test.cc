// Costanzi-2026 B_prj(R) model
// (src/pipelines/systematics/costanzi_bprj/cpp/costanzi_bprj_t.hh,
// y3_cluster::CostanziBprj_t): closed form at R = R0, the R -> 0 / R >> R0
// power-law limits, golden values shared with costanzi_bprj.test.py
// (transcribed independently from the App. C formula of arXiv:2604.05833),
// the values-file DataBlock constructor (default + custom section) and the
// gamma > 0 guard.
#include "catch2/catch.hpp"

#include "cosmosis/datablock/datablock.hh"

#include "pipelines/systematics/costanzi_bprj/cpp/costanzi_bprj_t.hh"

#include <cmath>
#include <stdexcept>

using y3_cluster::CostanziBprj_t;

namespace {

  double const LOB = 40.0;
  double const Z = 0.3;

  struct Pin {
    double R;
    double B;
  };
  // (R, lob = 40, z = 0.3) -> B, paper formula transcribed independently.
  Pin const SIGMA_PINS[] = {{0.5, 1.091982576562725},
                            {1.0, 1.091255050071865},
                            {3.0, 1.0581193649579876}};
  Pin const DSIGMA_PINS[] = {{0.5, 1.0031265151008693},
                             {1.0, 1.022544206891198},
                             {3.0, 1.105348390209914}};

  void
  put_params(cosmosis::DataBlock& s, char const* section, CostanziBprj_t const& m)
  {
    s.put_val(section, "A", m.A());
    s.put_val(section, "alpha", m.alpha());
    s.put_val(section, "beta", m.beta());
    s.put_val(section, "gamma", m.gamma());
  }

} // namespace

TEST_CASE("CostanziBprj_t r0 is R_lambda (1+z)")
{
  CHECK(CostanziBprj_t::r0(LOB, Z) == Approx(1.082319169622435).epsilon(1e-14));
  CHECK(CostanziBprj_t::r0(100.0, 0.0) == Approx(1.0).epsilon(1e-15));
}

TEST_CASE("CostanziBprj_t matches the independently transcribed golden values")
{
  auto const sig = CostanziBprj_t::sigma();
  auto const dsg = CostanziBprj_t::dsigma();
  for (auto const& p : SIGMA_PINS)
    CHECK(sig(p.R, LOB, Z) == Approx(p.B).epsilon(1e-12));
  for (auto const& p : DSIGMA_PINS)
    CHECK(dsg(p.R, LOB, Z) == Approx(p.B).epsilon(1e-12));
}

TEST_CASE("CostanziBprj_t closed form at R0 and power-law limits")
{
  double const r0 = CostanziBprj_t::r0(LOB, Z);
  for (auto const& m : {CostanziBprj_t::sigma(), CostanziBprj_t::dsigma()}) {
    double const want =
      m.A() * std::pow(2.0, (m.beta() - m.alpha()) / m.gamma()) + 1.0;
    CHECK(m(r0, LOB, Z) == Approx(want).epsilon(1e-14));
    CHECK(m(0.0, LOB, Z) == 1.0);  // alpha > 0: no correction at R = 0
  }
  auto const m = CostanziBprj_t::sigma();
  double const xo = 1e6, xi = 1e-3;
  CHECK(m(xo * r0, LOB, Z) - 1.0 ==
        Approx(m.A() * std::pow(xo, m.beta())).epsilon(1e-9));
  CHECK(m(xi * r0, LOB, Z) - 1.0 ==
        Approx(m.A() * std::pow(xi, m.alpha())).epsilon(1e-9));
}

TEST_CASE("CostanziBprj_t reads the values-file section from the DataBlock")
{
  cosmosis::DataBlock s;
  put_params(s, "costanzi_bprj", CostanziBprj_t::sigma());
  put_params(s, "bprj_dsigma", CostanziBprj_t::dsigma());

  CostanziBprj_t const from_default(s);
  CostanziBprj_t const from_custom(s, "bprj_dsigma");
  for (auto const& p : SIGMA_PINS)
    CHECK(from_default(p.R, LOB, Z) == Approx(p.B).epsilon(1e-12));
  for (auto const& p : DSIGMA_PINS)
    CHECK(from_custom(p.R, LOB, Z) == Approx(p.B).epsilon(1e-12));
  CHECK_THROWS(CostanziBprj_t(s, "missing_section"));
}

TEST_CASE("CostanziBprj_t rejects gamma <= 0")
{
  CHECK_THROWS_AS(CostanziBprj_t(0.1, 0.1, -0.5, 0.0), std::invalid_argument);
  CHECK_THROWS_AS(CostanziBprj_t(0.1, 0.1, -0.5, -1.0), std::invalid_argument);
}
