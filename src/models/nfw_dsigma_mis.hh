// Off-Centered Sigma NFW Profile
// Uses an interpolation table (look at data/nfw_off_center/)
// Assumes that datablock has rho_c, concetration
#ifndef Y3_CLUSTER_NFW_DSIGMA_MIS_HH
#define Y3_CLUSTER_NFW_DSIGMA_MIS_HH

#include <algorithm>
#include <iostream>
#include <math.h>
#include <string>

#include "fmt/core.h"

#include <optional>

#include "cosmosis/datablock/datablock.hh"

#include "utils/interp_1d.hh"
#include "utils/interp_2d.hh"
#include "utils/make_interp_2d.hh"
#include "utils/make_interp_1d.hh"
#include "utils/read_vector.hh"

namespace y3_cluster {
  // Default concentration value
  double const CONC = 4.0;

  // Critical density in Msun/Mpc^3
  double const RHOC = 2.77533742639e+11;

  // selects the miscentering kernel ('single','gamma')
  std::string const GAMMA = "gamma";

  // Helper functions to construct filenames needed to read the interpolation
  // table information.
    static inline std::string
    logx_file(std::string const& kernel)
    {
      return fmt::format("nfw_off_center/table_1000_1e-03_5e+03_{}_logx.txt",
                         kernel);
    }

    static inline std::string
    logxmis_file(std::string const& kernel)
    {
      return fmt::format("nfw_off_center/table_1000_1e-03_5e+03_{}_logxmis.txt",
                         kernel);
    }

    static inline std::string
    log_dsigma_file(std::string const& kernel)
    {
      return fmt::format(
        "nfw_off_center/table_1000_1e-03_5e+03_log_deltasigma_{}.txt", kernel);
    }

  // Available miscentering kernels for NFW_DSIGMA_MIS.
  std::string const SINGLE = "single";

  class NFW_DSIGMA_MIS {

    public:
    NFW_DSIGMA_MIS(double c, double rhoc, std::string const& kernel)
      : _c(c),
        _rhoc(rhoc),
        _rho_mult(1.0),
        _nfwProfile(read_vector(logx_file(kernel)),
                    read_vector(logxmis_file(kernel)),
                    read_vector(log_dsigma_file(kernel)))
    { }

    NFW_DSIGMA_MIS()
    : _c(CONC),
      _rhoc(RHOC),
      _rho_mult(1.0),
      _nfwProfile(read_vector(logx_file(GAMMA)),
                  read_vector(logxmis_file(GAMMA)),
                  read_vector(log_dsigma_file(GAMMA)))

    { }

    // See NFW_SIGMA_MIS::set_rho_mult.  Set to Omega_m per sample to
    // use rho_mean instead of rho_crit in the rho_s normalisation.
    void set_rho_mult(double m) { _rho_mult = m; }

    // Optional per-mass concentration c(lnM) (issue #13): the lookup
    // tables are universal in x = r/r_s, so ALL concentration
    // dependence is analytic (r_s = r_200/c and delta_c(c)).  When a
    // table is set (e.g. haloModel/{lnM, concentration} = Child18 at
    // one_halo_z), it overrides the fixed _c; clamped outside its
    // lnM range.  Default: unset -> legacy fixed c (= 4).
    void set_concentration_table(Interp1D t) { _c_tab = std::move(t); }
    double conc_at(double lnM) const
    {
      return _c_tab ? _c_tab->clamp(lnM) : _c;
    }

    // Scale radius r_s(M) [cMpc/h], exposed for callers that need it
    // directly (e.g. the frozen-physics r_s(M)-anchored amplitude
    // drift a_b(z)). Additive-only: duplicates the same formula
    // operator() computes inline, so any future change to that formula
    // must be mirrored here.
    double
    r_s(double lnM) const
    {
      double const r_200 = std::cbrt(3.0 * std::exp(lnM) / (800.0 * M_PI * _rhoc));
      return r_200 / conc_at(lnM);
    }

    double
    operator()(double r, double rmis, double lnM) const
    {
      double const rho_crit = _rhoc;
      double const c = conc_at(lnM);
      double const delta_c = (200.0 * c * c * c / 3.0) / (std::log(1.0 + c) - c / (1.0 + c));
      double const r_200 = std::cbrt(3.0 * std::exp(lnM) / (800.0 * M_PI * rho_crit));
      double const r_s = r_200 / c;

      double const x = r / r_s;
      double const xmis = rmis / r_s;

      double const log_unfw = _nfwProfile.clamp(std::log(x), std::log(xmis));

      // normalization term defined in Wright & Brainerd 2000
      double const norm = 2 * r_s * delta_c * rho_crit * _rho_mult;
      double const nfw = norm * std::exp(log_unfw);

      // Conversion from Msun/Mpc^2 to Msun/h pc^2
      return nfw*1e-12;
    }

  private:
    double const _c;
    double const _rhoc;
    double       _rho_mult;
    std::optional<Interp1D> _c_tab;
    Interp2D _nfwProfile;
  };
}
#endif

// USED FOR DEBUGGING
// using printf debugs r200 and r_s
// printf("r200: %f, r_s: %f\n", r_200, r_s);
// printf("delta_c: %f, rho_crit: %f\n", delta_c, rho_crit);
// printf("norm: %f\n", norm);
// printf("conc: %f\n", _c);