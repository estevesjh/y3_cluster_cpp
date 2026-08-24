// Off-Centered Sigma NFW Profile
// ----------- edited by Arwa but not sure about it ---------------------
// Uses an interpolation table (look at data/nfw_off_center/)
// Assumes that datablock has rho_c, concetration
#ifndef Y3_CLUSTER_NFW_DSIGMA_MIS
#define Y3_CLUSTER_NFW_DSIGMA_MIS

#include <algorithm>
#include <iostream>
#include <math.h>
#include <string>

#include "fmt/core.h"

#include "cosmosis/datablock/datablock.hh"

#include "utils/cuda_interp_1d.cuh"
#include "utils/cuda_interp_2d.cuh"
#include "utils/make_interp_1d.hh"
#include "utils/read_vector.hh"

namespace y3_cuda {
  // Default concentration value
  double const DSIGMA_MIS_CONC = 4.0;
  double const DSIGMA_MIS_RHOC = 2.77533742639e+11;
  std::string const DSIGMA_MIS_GAMMA = "gamma";

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

  class NFW_DSIGMA_MIS {

    public:
    NFW_DSIGMA_MIS(double c, double rhoc, std::string const& kernel)
      : _c(c),
        _rhoc(rhoc),
        _rho_mult(1.0),
        _nfwProfile(read_vector(logx_file(kernel)),
                    read_vector(logxmis_file(kernel)),
                    read_vector(log_dsigma_file(kernel))),
        _c_tab(std::vector<double>{0.0, 100.0}, std::vector<double>{c, c})
    { }


    NFW_DSIGMA_MIS()
    : _c(DSIGMA_MIS_CONC),
      _rhoc(DSIGMA_MIS_RHOC),
      _rho_mult(1.0),
      _nfwProfile(read_vector(logx_file(DSIGMA_MIS_GAMMA)),
                  read_vector(logxmis_file(DSIGMA_MIS_GAMMA)),
                  read_vector(log_dsigma_file(DSIGMA_MIS_GAMMA))),
      _c_tab(std::vector<double>{0.0, 100.0},
             std::vector<double>{DSIGMA_MIS_CONC, DSIGMA_MIS_CONC})
    { }

    // issue #13/#14: per-mass concentration c(lnM) = haloModel/concentration
    // (Child18 x concentration_amplitude), mirroring the CPU
    // y3_cluster::NFW_DSIGMA_MIS::set_concentration_table. The lookup table is
    // universal in x = r/r_s, so this only changes the analytic r_s = r_200/c
    // and delta_c(c) per call. Host-only (builds the device table); after this,
    // conc_at(lnM) uses the table on both host and device. Without it, the
    // legacy fixed c=4 default is used (cross-backend identity pins stay valid).
    void
    set_concentration_table(std::vector<double> const& lnM,
                            std::vector<double> const& conc)
    {
      gpu_support::Interp1D t(lnM, conc);
      _c_tab.swap(t);
      _use_ctab = true;
    }

    // Datablock convenience overload (host-only): reads the c(lnM) columns
    // from `section`, like make_Interp1D. Call in set_sample when
    // use_halo_model_conc is on.
    void
    set_concentration_table(cosmosis::DataBlock& sample,
                            char const* section = "haloModel",
                            char const* lnM_name = "lnM",
                            char const* conc_name = "concentration")
    {
      set_concentration_table(
        sample.view<std::vector<double>>(section, lnM_name),
        sample.view<std::vector<double>>(section, conc_name));
    }

    __host__ __device__ double
    conc_at(double lnM) const
    {
      return _use_ctab ? _c_tab.clamp(lnM) : _c;
    }

    // DONE (issue #13/#14): set_concentration_table / conc_at above mirror the
    // CPU nfw_dsigma_mis.hh. Default (no table set) keeps the fixed c=4 that the
    // cross-backend identity pins test, so they stay valid; the GPU projection
    // sites now honor use_halo_model_conc like the CPU ones. NOTE: needs a
    // Perlmutter GPU build (nvcc compile on login node; ctest on a GPU node).
    // TODO: Implement different operator in case of rhocz(zt)
    // Ask Marc How to make _c and _rhoc be functional forms in any case

    NFW_DSIGMA_MIS(cosmosis::DataBlock& sample)
        : _c(y3_cluster::make_Interp1D(sample,
                                        "haloModel",
                                        "lnM",
                                        "concentration")
                .clamp(14.0)),
            _rhoc(y3_cluster::make_Interp1D(sample,
                                            "haloModel",
                                            "z",
                                            "rhoc")
                    .clamp(0.0)),
            _rho_mult(1.0),
            _nfwProfile(read_vector(logx_file(DSIGMA_MIS_GAMMA)),
                        read_vector(logxmis_file(DSIGMA_MIS_GAMMA)),
                        read_vector(log_dsigma_file(DSIGMA_MIS_GAMMA))),
            _c_tab(std::vector<double>{0.0, 100.0},
                   std::vector<double>{_c, _c})
        { }

    // Multiplier applied to rho_s (= rho_crit * delta_c).  Set to
    // Omega_m after reading cosmological_parameters to switch from
    // the rho_crit-based normalisation to rho_mean-based, matching
    // the CPU y3_cluster::NFW_DSIGMA_MIS (nfw_dsigma_mis.hh) and the
    // Python reference (richness_selection.nfw.NFWMiscentered).
    // Default is 1.0 (legacy rho_crit behaviour) so existing GPU
    // callers that never call this are unaffected.
    void set_rho_mult(double m) { _rho_mult = m; }

    __device__ __host__ double
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
    gpu_support::Interp2D _nfwProfile;
    gpu_support::Interp1D _c_tab;        // per-mass c(lnM); device-safe (by-value)
    bool _use_ctab = false;             // false -> use fixed _c (default c=4)
  };
}
#endif

// USED FOR DEBUGGING
// using printf debugs r200 and r_s
// printf("r200: %f, r_s: %f\n", r_200, r_s);
// printf("delta_c: %f, rho_crit: %f\n", delta_c, rho_crit);
// printf("norm: %f\n", norm);
// printf("conc: %f\n", _c);