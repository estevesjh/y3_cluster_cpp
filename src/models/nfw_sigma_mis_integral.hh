// Miscentered NFW Sigma / DeltaSigma by DIRECT integration of the analytic
// Wright & Brainerd (2000) NFW profile, using GSL ADAPTIVE quadrature
// (gsl_integration_qags).  Drop-in for the "single" lookup-table classes
// NFW_SIGMA_MIS / NFW_DSIGMA_MIS (same operator()(r, rmis, lnM),
// set_rho_ref, set_concentration_table, conc_at, delta_sigma).
//
//   Sigma_mis(R|R_mis)      = (1/pi) int_0^pi Sigma_NFW(s(phi)) dphi,
//                             s(phi) = sqrt(R^2 + R_mis^2 - 2 R R_mis cos phi)
//   Sigmabar_mis(<R|R_mis)  = (2/R^2) int_0^R Sigma_mis(R'|R_mis) R' dR'
//   DeltaSigma_mis(R|R_mis) = Sigmabar_mis(<R) - Sigma_mis(R)
//
// WHY GSL ADAPTIVE (not fixed Gauss-Legendre, not the log table):
//  * The phi integrand has an integrable LOG singularity at phi->0 when
//    R ~ R_mis (s->|R-R_mis|->0, Sigma_NFW ~ -ln s).  A fixed-node GL rule
//    either misses it or returns inf/NaN; QAGS refines into it.
//  * The radial Sigmabar integrand Sigma_mis(R') peaks sharply at R'=R_mis
//    (when R_mis<R); QAGS (split at R_mis) resolves the cusp.
//  * DeltaSigma_mis goes NEGATIVE for R_mis > R (halo offset beyond the
//    aperture: Sigmabar(<R) < Sigma(R)).  The shipped log_deltasigma_single
//    table is exp(spline(...)) and STRUCTURALLY cannot represent this; it
//    zeroes the R_mis>=R lobe, so the projection mean-field ("rnd") term
//    never cancels in DeltaSigma.  This class produces the correct negative
//    lobe, restoring the cancellation.
//
// *** STOPGAP ***  Intended for the R_mis >= R tail where the log table
// fails; the fast table stays valid for R_mis < R (positive lobe, matches
// this integral to <1e-4).  The proper fix is to regenerate the off-center
// tables in a SIGNED (not log) representation over the (x, x_mis) range the
// projection actually samples.
//
// Units/convention identical to NFW_SIGMA_MIS (pipeline M200c recipe):
// Sigma [Msun/h/pc^2], R,R_mis [cMpc/h], M via lnM [Msun/h] (M_200c),
// delta_c = (200 c^3/3)/(ln(1+c) - c/(1+c)),
// r_200 = cbrt(3 exp(lnM)/(800 pi rhoc)), r_s = r_200/c,
// coeff = 2 r_s delta_c rho_ref * 1e-12.
#ifndef Y3_CLUSTER_NFW_SIGMA_MIS_INTEGRAL_HH
#define Y3_CLUSTER_NFW_SIGMA_MIS_INTEGRAL_HH

#include <cmath>
#include <optional>
#include <stdexcept>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>

#include "utils/interp_1d.hh"

namespace y3_cluster {

  class NFW_SIGMA_MIS_INTEGRAL {
   public:
    // n_ws: GSL workspace size (max subintervals).  eps: relative tol.
    explicit NFW_SIGMA_MIS_INTEGRAL(double c = 4.0,
                                    double rhoc = 2.77533742639e+11,
                                    std::size_t n_ws = 512,
                                    double eps = 1.0e-6)
      : _c(c), _rhoc(rhoc), _rho_b(rhoc), _eps(eps), _n_ws(n_ws)
    {
      gsl_set_error_handler_off();   // return codes instead of abort()
      _ws_phi = gsl_integration_workspace_alloc(n_ws);
      _ws_rad = gsl_integration_workspace_alloc(n_ws);
      if (!_ws_phi || !_ws_rad)
        throw std::runtime_error("NFW_SIGMA_MIS_INTEGRAL: GSL alloc failed");
    }

    ~NFW_SIGMA_MIS_INTEGRAL()
    {
      if (_ws_phi) gsl_integration_workspace_free(_ws_phi);
      if (_ws_rad) gsl_integration_workspace_free(_ws_rad);
    }

    // Non-copyable (owns GSL workspaces); movable.
    NFW_SIGMA_MIS_INTEGRAL(NFW_SIGMA_MIS_INTEGRAL const&) = delete;
    NFW_SIGMA_MIS_INTEGRAL& operator=(NFW_SIGMA_MIS_INTEGRAL const&) = delete;
    NFW_SIGMA_MIS_INTEGRAL(NFW_SIGMA_MIS_INTEGRAL&& o) noexcept
      : _c(o._c), _rhoc(o._rhoc), _rho_b(o._rho_b), _eps(o._eps),
        _n_ws(o._n_ws), _c_tab(std::move(o._c_tab)),
        _ws_phi(o._ws_phi), _ws_rad(o._ws_rad)
    { o._ws_phi = nullptr; o._ws_rad = nullptr; }
    NFW_SIGMA_MIS_INTEGRAL& operator=(NFW_SIGMA_MIS_INTEGRAL&&) = delete;

    // UNIFIED rho_m convention: one density for boundary AND amplitude
    // (haloModel/rho_m_ref). Normalization factors live OUTSIDE.
    void set_rho_ref(double rho) { _rho_b = rho; }
    void set_concentration_table(Interp1D t) { _c_tab = std::move(t); }
    double conc_at(double lnM) const { return _c_tab ? _c_tab->clamp(lnM) : _c; }

    // Miscentered Sigma at projected radius r, halo offset rmis, ln-mass lnM.
    double
    operator()(double r, double rmis, double lnM) const
    {
      double rs, coeff, cc;
      profile_consts_(lnM, rs, coeff, cc);
      return sigma_mis_(r, rmis, rs, coeff);
    }

    // Miscentered DeltaSigma = Sigmabar(<r) - Sigma(r).  Signed (negative
    // for rmis > r), unlike the log table.
    double
    delta_sigma(double r, double rmis, double lnM) const
    {
      double rs, coeff, cc;
      profile_consts_(lnM, rs, coeff, cc);
      return sigmabar_(r, rmis, rs, coeff) - sigma_mis_(r, rmis, rs, coeff);
    }

   private:
    // ---- profile constants (pipeline M200c recipe) ---------------------
    // r_s [cMpc/h]; coeff = 2 r_s delta_c rho_ref / 1e12 [Msun/h/pc^2].
    void
    profile_consts_(double lnM, double& rs, double& coeff, double& c) const
    {
      c = conc_at(lnM);
      double const delta_c =
        (200.0 * c * c * c / 3.0) / (std::log(1.0 + c) - c / (1.0 + c));
      double const r_200 = std::cbrt(3.0 * std::exp(lnM) / (800.0 * M_PI * _rho_b));
      rs = r_200 / c;
      coeff = 2.0 * rs * delta_c * _rho_b * 1.0e-12;
    }

    // Centered analytic NFW Sigma (Wright & Brainerd 2000), regularized so
    // the log divergence at r->0 is finite (QAGS may probe very small r).
    double
    sigma_centered_(double r, double rs, double coeff) const
    {
      double x = std::abs(r) / rs;
      if (x < 1.0e-8) x = 1.0e-8;   // regularize integrable log singularity
      if (x < 1.0 - 1.0e-9)
        return coeff / (x * x - 1.0) *
               (1.0 - 2.0 / std::sqrt(1.0 - x * x) *
                        std::atanh(std::sqrt((1.0 - x) / (1.0 + x))));
      if (x <= 1.0 + 1.0e-9) return coeff / 3.0;
      return coeff / (x * x - 1.0) *
             (1.0 - 2.0 / std::sqrt(x * x - 1.0) *
                      std::atan(std::sqrt((x - 1.0) / (1.0 + x))));
    }

    // ---- GSL integrands (C callbacks + params) -------------------------
    struct Params {
      NFW_SIGMA_MIS_INTEGRAL const* self;
      double rs, coeff, R, rmis;
    };

    static double
    phi_integrand_(double phi, void* pv)
    {
      auto const* p = static_cast<Params const*>(pv);
      double const s = std::sqrt(p->R * p->R + p->rmis * p->rmis -
                                 2.0 * p->R * p->rmis * std::cos(phi));
      return p->self->sigma_centered_(s, p->rs, p->coeff);
    }

    static double
    rad_integrand_(double Rp, void* pv)
    {
      auto const* p = static_cast<Params const*>(pv);
      return p->self->sigma_mis_(Rp, p->rmis, p->rs, p->coeff) * Rp;
    }

    // Azimuthally-averaged (miscentered) Sigma via QAGS over [0, pi].
    double
    sigma_mis_(double R, double rmis, double rs, double coeff) const
    {
      if (rmis <= 0.0) return sigma_centered_(R, rs, coeff);
      Params p{this, rs, coeff, R, rmis};
      gsl_function F{&phi_integrand_, &p};
      double res = 0.0, err = 0.0;
      gsl_integration_qags(&F, 0.0, M_PI, 0.0, _eps, _n_ws, _ws_phi, &res, &err);
      return res / M_PI;   // (1/pi) int_0^pi  ==  (1/2pi) int_0^2pi (even)
    }

    // Sigmabar(<R) = (2/R^2) int_0^R Sigma_mis(R') R' dR'.  Split the radial
    // integral at R'=rmis (cusp) when 0 < rmis < R.
    double
    sigmabar_(double R, double rmis, double rs, double coeff) const
    {
      Params p{this, rs, coeff, R, rmis};
      gsl_function F{&rad_integrand_, &p};
      double res = 0.0, err = 0.0;
      if (rmis > 0.0 && rmis < R) {
        double a = 0.0, b = 0.0;
        gsl_integration_qags(&F, 0.0, rmis, 0.0, _eps, _n_ws, _ws_rad, &a, &err);
        gsl_integration_qags(&F, rmis, R, 0.0, _eps, _n_ws, _ws_rad, &b, &err);
        res = a + b;
      } else {
        gsl_integration_qags(&F, 0.0, R, 0.0, _eps, _n_ws, _ws_rad, &res, &err);
      }
      return (2.0 / (R * R)) * res;
    }

    double _c, _rhoc, _rho_b, _eps;
    std::size_t _n_ws;
    std::optional<Interp1D> _c_tab;
    gsl_integration_workspace* _ws_phi = nullptr;
    gsl_integration_workspace* _ws_rad = nullptr;
  };

}  // namespace y3_cluster
#endif
