// Miscentered Sigma NFW profile (CPU), two-halo Sigma_prj's inner NFW.
// Mirrors src/models/nfw_dsigma_mis.hh with the "log_sigma" 2-D lookup
// (not "log_deltasigma") from data/nfw_off_center/.
#ifndef Y3_CLUSTER_NFW_SIGMA_MIS_HH
#define Y3_CLUSTER_NFW_SIGMA_MIS_HH

#include <cmath>
#include <optional>
#include <string>

#include "fmt/core.h"

#include "utils/interp_1d.hh"
#include "utils/interp_2d.hh"
#include "utils/read_vector.hh"

namespace y3_cluster {
  // Shared defaults with nfw_dsigma_mis.hh (avoid redefining CONC, RHOC).
  inline std::string const NFW_SIG_GAMMA = "gamma";

  namespace nfw_sig_detail {
    inline std::string logx_file(std::string const& k) {
      return fmt::format("nfw_off_center/table_1000_1e-03_5e+03_{}_logx.txt", k);
    }
    inline std::string logxmis_file(std::string const& k) {
      return fmt::format("nfw_off_center/table_1000_1e-03_5e+03_{}_logxmis.txt", k);
    }
    inline std::string log_sigma_file(std::string const& k) {
      return fmt::format("nfw_off_center/table_1000_1e-03_5e+03_log_sigma_{}.txt", k);
    }
  }

  class NFW_SIGMA_MIS {
   public:
    NFW_SIGMA_MIS(double c, double rhoc, std::string const& kernel)
      : _c(c), _rhoc(rhoc),
        _nfwProfile(read_vector(nfw_sig_detail::logx_file(kernel)),
                    read_vector(nfw_sig_detail::logxmis_file(kernel)),
                    read_vector(nfw_sig_detail::log_sigma_file(kernel)))
    {}

    NFW_SIGMA_MIS()
      : NFW_SIGMA_MIS(4.0, 2.77533742639e+11, NFW_SIG_GAMMA)
    {}

    // UNIFIED rho_m convention (2026-08-24 decision): use `rho` --
    // haloModel/rho_m_ref = Omega_m rho_crit,0 (1+z_density)^3, the same
    // density first_halo_term builds the centred tables with -- for BOTH
    // the halo boundary r_200 = [3M/(800 pi rho)]^(1/3) (a mean-density
    // boundary) and the amplitude rho_s = delta_c * rho. Production call
    // sites use this. Pure normalization factors (e.g. legacy Omega_m,
    // the physical-density (1+z)^2) are applied OUTSIDE by the caller.
    void set_rho_ref(double rho) { _rho_b = rho; }

    // Optional per-mass concentration c(lnM) -- see
    // NFW_DSIGMA_MIS::set_concentration_table (issue #13).
    void set_concentration_table(Interp1D t) { _c_tab = std::move(t); }
    double conc_at(double lnM) const
    {
      return _c_tab ? _c_tab->clamp(lnM) : _c;
    }

    // Miscentered Sigma at projected radius r with halo offset rmis.
    // lnM is raw natural log of M in M_sun/h (M_200m).
    double
    operator()(double r, double rmis, double lnM) const
    {
      double const c       = conc_at(lnM);
      double const delta_c = (200.0 * c * c * c / 3.0) /
                             (std::log(1.0 + c) - c / (1.0 + c));
      double const r_200   = std::cbrt(3.0 * std::exp(lnM) / (800.0 * M_PI * _rho_b));
      double const r_s     = r_200 / c;
      double const x       = r / r_s;
      double const xmis    = rmis / r_s;

      double const log_sig = _nfwProfile.clamp(std::log(x), std::log(xmis));
      double const norm    = 2.0 * r_s * delta_c * _rho_b;
      return norm * std::exp(log_sig) * 1e-12;    // -> M_sun / h / pc^2
    }

   private:
    double const _c;
    double const _rhoc;
    double       _rho_b{_rhoc};   // boundary+amplitude density (set_rho_ref)
    std::optional<Interp1D> _c_tab;
    Interp2D _nfwProfile;
  };

  // Available miscentering kernels for NFW_SIGMA_MIS / NFW_DSIGMA_MIS.
  inline std::string const NFW_SIG_SINGLE = "single";
}  // namespace y3_cluster

#endif
