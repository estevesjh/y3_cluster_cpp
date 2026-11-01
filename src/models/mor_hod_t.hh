// HOD mass-observable relation P(λ^true | M, z, φ) — continuous shifted-
// Poisson form from docs/richness_selection_function.tex §5.2
// (Eq. cont_poisson_shifted):
//
//   λ^cen  = 1[M ≥ Mmin]
//   μ_sat  = ((M - Mmin)/(M1 - Mmin))^α · ((1+z)/(1+z_pivot))^ε
//   δ      = (σ_intr · μ_sat)^2
//   ν      = μ_sat + δ
//   x      = ltr + δ
//   P(ltr | M, z; φ) = exp[-ν + (x - 1) ln(ν) - lnΓ(x)]     for x > 0
//
// The "-1"/Γ(x) of the Γ-continued Poisson pmf is the central: ltr - 1 + δ is
// Poisson-like with mean ν, so <ltr> = 1 + μ_sat.  This is the documented
// Costanzi form (docs math/index.md), identical to p_operator_t.hh and
// mor_shifted_poisson_t.cuh.  Below Mmin μ_sat = 0 and the density is the
// narrow Gaussian at λ^cen = 0.  (Before 2026-11-01 x = ltr - λ^cen + δ, which
// counted the central twice and put <ltr> one unit high.)
//
// Closed form; no discrete k-sum; recovers the Poisson limit as
// σ_intr → 0 and extends smoothly to non-integer ltr. Matches the
// Poisson⊗Gauss form to ~0.1% per the doc; pipelines that need the
// per-k evaluator (src/models/p_operator_t.hh) still use the
// accessors below (`poisson_truncation`, `sigma_lambda`, `mu_sat`).
//
// φ = (log10 Mmin, log10 M1, α, ε, σ_λ), read from the `cluster_mor`
// datablock section.
//
// Parameterization: the class reads EITHER
//   - log10_Mmin AND log10_M1                (emulator's native form), OR
//   - log10_Mmin AND log10_ratio             (MCMC-friendly form, where
//                                              log10_ratio = log10(M1/Mmin))
// If both are present, log10_ratio wins. The ratio form lets samplers
// use a clean boxcar prior on each of (log10_Mmin, log10_ratio), instead
// of Costanzi's conditional box M1/Mmin ∈ [10, 30]. Y1 pipelines already
// use this convention (see y1_mock_values.ini: mor_logMmin + mor_logRatio).
//
// Mass convention: operator() treats lnM as raw natural log of the halo
// mass in h^-1 M_sun (i.e. M = exp(lnM)).
//
// Note: HMF_t uses a different convention internally — it multiplies the
// `mass_function/m_h` array by `(Ω_M - Ω_ν)` before taking log, so when
// this integrand (or the emulator-backed one) calls `hmf(lnM, zt)` with
// raw lnM, the HMF table is indexed at a shifted mass node. This affects
// the absolute physics normalization but NOT the emu-vs-full comparison
// (both modules see identical hmf values for identical lnM inputs).
// TODO: unify the mass convention across HMF / MOR / emulator feeder.
#ifndef Y3_CLUSTER_CPP_MOR_HOD_T_HH
#define Y3_CLUSTER_CPP_MOR_HOD_T_HH

#include "cosmosis/datablock/datablock.hh"

#include <algorithm>
#include <cmath>
#include <vector>

namespace y3_cluster {

  class MOR_HOD_t {
  public:
    static constexpr double Z_PIVOT_DEFAULT = 0.45;
    static constexpr double POISSON_TOL = 1e-8;
    static constexpr double FALLBACK_SIGMA = 1.0e-3;
    static constexpr double MIN_SIGMA = 1.0e-6;

    MOR_HOD_t(double log10_Mmin,
              double log10_M1,
              double alpha,
              double epsilon,
              double sigma_lambda,
              double z_pivot = Z_PIVOT_DEFAULT)
      : _log10_Mmin(log10_Mmin)
      , _log10_M1(log10_M1)
      , _alpha(alpha)
      , _epsilon(epsilon)
      , _sigma_lambda(sigma_lambda)
      , _z_pivot(z_pivot)
    {}

    explicit MOR_HOD_t(cosmosis::DataBlock& sample)
      : _log10_Mmin(sample.view<double>("cluster_mor", "log10_Mmin"))
      , _log10_M1(_read_log10_M1(sample, _log10_Mmin))
      , _alpha(sample.view<double>("cluster_mor", "alpha"))
      , _epsilon(sample.view<double>("cluster_mor", "epsilon"))
      , _sigma_lambda(sample.view<double>("cluster_mor", "sigma_lambda"))
      , _z_pivot(Z_PIVOT_DEFAULT)
    {
      if (sample.has_val("cluster_mor", "z_pivot")) {
        _z_pivot = sample.view<double>("cluster_mor", "z_pivot");
      }
    }

    // P(lt | M, z; φ) — shifted-Poisson continuous form.
    // lnM is raw natural log of M in h^-1 M_sun (M = exp(lnM)).
    //
    // Support: ltr >= 0 and x > 0.  The continuous density spreads the
    // zero-satellite state over ltr in [0, 1) above Mmin, as the other
    // shifted-Poisson evaluators do.
    double
    operator()(double lt, double lnM, double zt) const
    {
      if (lt < 0.0) return 0.0;

      double const M = std::exp(lnM);
      double const lcentral = _lambda_central(M);
      double const mu = _mu_sat(M, zt);

      // μ_sat → 0 branch: collapse to narrow Gaussian δ(lt - lcentral).
      if (mu <= POISSON_TOL) {
        double const dx = (lt - lcentral) / FALLBACK_SIGMA;
        return std::exp(-0.5 * dx * dx) /
               (std::sqrt(2.0 * M_PI) * FALLBACK_SIGMA);
      }

      double const delta = (_sigma_lambda * mu) * (_sigma_lambda * mu);
      double const nu = mu + delta;
      double const x = lt + delta;
      if (x <= 0.0) return 0.0;

      double const log_P = -nu + (x - 1.0) * std::log(nu) - std::lgamma(x);
      return std::exp(log_P);
    }

    // Accessors used by batched evaluators (e.g. P_operator) that
    // hoist the Poisson-weight loop out of the lt loop, or swap in a
    // shifted-Poisson variant (Costanzi-2026 form).
    double log10_Mmin()     const { return _log10_Mmin; }
    double log10_M1()       const { return _log10_M1;   }
    double alpha()          const { return _alpha;      }
    double epsilon()        const { return _epsilon;    }
    double sigma_lambda()   const { return _sigma_lambda; }
    double z_pivot()        const { return _z_pivot;    }
    double mu_sat(double M, double z) const { return _mu_sat(M, z); }
    void   poisson_truncation(double mu, int& k_lo, int& k_hi) const
    {
      _poisson_truncation(mu, k_lo, k_hi);
    }

    // Linear-space mean mu_eff = lambda_central + mu_sat (leading order).
    // Used by callers (richness selection) that need a quadrature bracket
    // around the PDF support.
    double
    ltr_mean(double lnM, double zt) const
    {
      double const M = std::exp(lnM);
      return _lambda_central(M) + _mu_sat(M, zt);
    }

    // Linear-space scatter, sigma_eff = sqrt(mu_sat + (sigma_lambda mu_sat)^2).
    // See docs/richness_selection_function.tex §5.2 (Eq. hod_variance).
    double
    ltr_sigma(double lnM, double zt) const
    {
      double const M = std::exp(lnM);
      double const mu = _mu_sat(M, zt);
      return std::sqrt(std::max(mu + _sigma_lambda * _sigma_lambda * mu * mu,
                                0.0));
    }

  private:
    double _log10_Mmin;
    double _log10_M1;
    double _alpha;
    double _epsilon;
    double _sigma_lambda;
    double _z_pivot;

    double
    _lambda_central(double M) const
    {
      double const Mmin = std::pow(10.0, _log10_Mmin);
      return (M >= Mmin) ? 1.0 : 0.0;
    }

    double
    _mu_sat(double M, double z) const
    {
      double const Mmin = std::pow(10.0, _log10_Mmin);
      double const M1 = std::pow(10.0, _log10_M1);
      double const dM1 = M1 - Mmin;
      if (dM1 <= 0.0) return 0.0; // invalid: emulator prior forbids this
      double const dM = std::max(M - Mmin, 0.0);
      if (dM <= 0.0) return 0.0;
      double const base = dM / dM1;
      double const redshift_evolution =
        std::pow((1.0 + z) / (1.0 + _z_pivot), _epsilon);
      return std::pow(base, _alpha) * redshift_evolution;
    }

    // Reads log10_M1 from cluster_mor, allowing two forms:
    //   - "log10_ratio" (preferred for MCMC): M1 = Mmin * 10^ratio
    //   - "log10_M1"   (emulator-native)
    // If both are present, log10_ratio wins.
    static double
    _read_log10_M1(cosmosis::DataBlock& sample, double log10_Mmin)
    {
      if (sample.has_val("cluster_mor", "log10_ratio")) {
        return log10_Mmin + sample.view<double>("cluster_mor", "log10_ratio");
      }
      return sample.view<double>("cluster_mor", "log10_M1");
    }

    // Match kernels/mor_hod.py:91-105.
    static void
    _poisson_truncation(double mu, int& k_lo, int& k_hi)
    {
      double const mu_safe = std::max(mu, 0.0);
      double const n_sigma = 12.0;
      double const half =
        std::max(20.0, n_sigma * std::sqrt(std::max(mu_safe, 1.0)));
      k_lo = std::max(0, static_cast<int>(std::floor(mu_safe - half)));
      k_hi = static_cast<int>(std::ceil(mu_safe + half));
    }
  };

} // namespace y3_cluster

#endif
