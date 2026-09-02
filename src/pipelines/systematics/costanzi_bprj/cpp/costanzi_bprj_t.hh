// Costanzi-2026 selection-bias correction B_prj(R) (arXiv:2604.05833, App. C).
//
// The correction multiplies the max halo lensing model:
//
//   Sigma_corr(R) = B_prj(R) Sigma_max(R),   Sigma_max = max(Sigma_1h, Sigma_2h)
//
//   B_prj(R) = A (R/R0)^alpha [1 + (R/R0)^gamma]^((beta - alpha)/gamma) + 1
//   R0       = R_lambda(lob) (1 + z)         comoving Mpc/h
//
// A sets the amplitude, alpha/beta the inner/outer slopes, R0 the transition
// scale and gamma its smoothness.  R must be comoving Mpc/h like R0.  The same
// form describes the bias on DeltaSigma(R); only the parameter values differ
// (sigma() / dsigma() below).
//
// In a pipeline the parameters live in the CosmoSIS values file and are read
// by the DataBlock constructor:
//
//   [costanzi_bprj]
//   A = 0.10
//   alpha = 0.1
//   beta = -0.53
//   gamma = 4.1
//
// Python twin: ../python/costanzi_bprj.py (CostanziBprj).
#ifndef Y3_CLUSTER_CPP_COSTANZI_BPRJ_T_HH
#define Y3_CLUSTER_CPP_COSTANZI_BPRJ_T_HH

#include "cosmosis/datablock/datablock.hh"
#include "pipelines/shared/lensing_helpers.hh"  // y3_pipelines::R_lambda

#include <cmath>
#include <stdexcept>
#include <string>

namespace y3_cluster {

  class CostanziBprj_t {
  public:
    CostanziBprj_t(double A, double alpha, double beta, double gamma)
      : _A(A), _alpha(alpha), _beta(beta), _gamma(gamma)
    {
      // gamma divides (beta - alpha); > 0 = smooth transition.
      if (!(gamma > 0.0))
        throw std::invalid_argument("CostanziBprj_t: gamma must be > 0");
    }

    explicit CostanziBprj_t(cosmosis::DataBlock& sample,
                            std::string const& section = "costanzi_bprj")
      : CostanziBprj_t(sample.view<double>(section, "A"),
                       sample.view<double>(section, "alpha"),
                       sample.view<double>(section, "beta"),
                       sample.view<double>(section, "gamma"))
    {}

    // Paper best fits.  NOTE: arXiv:2604.05833 App. C quotes alpha = 0.92 for
    // Sigma; alpha = 0.1 here follows the owner's spec (2026-09-01) -- confirm
    // against the published version before sampling around it.
    static CostanziBprj_t sigma()  { return {0.10, 0.1, -0.53, 4.1}; }   // Sigma(R)
    static CostanziBprj_t dsigma() { return {0.12, 4.11, 0.18, 1.82}; }  // DeltaSigma(R)

    // Transition scale R0 = R_lambda(lob) (1 + z) [comoving Mpc/h].
    static double
    r0(double lob, double z)
    {
      return y3_pipelines::R_lambda(lob) * (1.0 + z);
    }

    // B_prj at comoving radius R [Mpc/h] for a cluster of richness lob at z.
    double
    operator()(double R, double lob, double z) const
    {
      double const x = R / r0(lob, z);
      return _A * std::pow(x, _alpha) *
               std::pow(1.0 + std::pow(x, _gamma), (_beta - _alpha) / _gamma) +
             1.0;
    }

    double A() const { return _A; }
    double alpha() const { return _alpha; }
    double beta() const { return _beta; }
    double gamma() const { return _gamma; }

  private:
    double _A, _alpha, _beta, _gamma;
  };

}

#endif
