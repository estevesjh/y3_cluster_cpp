// Off-Centered Sigma NFW Profile
// ----------- edited by Arwa but not sure about it ---------------------
// Uses an interpolation table (look at data/nfw_off_center/)
// Assumes that datablock has rho_c, concetration
#ifndef Y3_CLUSTER_NFW_SIGMA_MIS
#define Y3_CLUSTER_NFW_SIGMA_MIS

#include <algorithm>
#include <iostream>
#include <math.h>
#include <string>

#include "fmt/core.h"

#include "cosmosis/datablock/datablock.hh"

#include "utils/cuda_interp_2d.cuh"
#include "utils/make_interp_1d.hh"
#include "utils/read_vector.hh"

namespace y3_cuda {
  // Default concentration value
  inline double const CONC = 4.0;
  inline double const RHOC = 2.77533742639e+11;
  inline std::string const GAMMA = "gamma";

  class NFW_SIGMA_MIS {

    // Helper functions to construct filenames needed to read the interpolation
    // table information. This is so we can initialize the tables correctly,
    // rather than attempting to default-construct them and then replace the
    // default- constructed table with a correctly initialized one.

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
    log_sigma_file(std::string const& kernel)
    {
      return fmt::format(
        "nfw_off_center/table_1000_1e-03_5e+03_log_sigma_{}.txt", kernel);
    }

  public:
    NFW_SIGMA_MIS(double c, double rhoc, std::string const& kernel)
      : _c(c)
      , _rhoc(rhoc)
      , _nfwProfile(read_vector(logx_file(kernel)),
                    read_vector(logxmis_file(kernel)),
                    read_vector(log_sigma_file(kernel)))
    {}

    NFW_SIGMA_MIS()
      : _c(CONC)
      , _rhoc(RHOC)
      , _nfwProfile(read_vector(logx_file(GAMMA)),
                    read_vector(logxmis_file(GAMMA)),
                    read_vector(log_sigma_file(GAMMA)))

    {}

    // In case, we envolve the NFW profile with redshift
    // TODO: Implement Mass-Concentration Relation
    // TODO: Implement different operator in case of rhocz(zt)
    // Ask Marc How to make _c and _rhoc be functional forms in any case
    NFW_SIGMA_MIS(cosmosis::DataBlock& sample)
      : _c(y3_cluster::make_Interp1D(sample,
                                     "haloModel",
                                     "lnM",
                                     "concentration")
             .clamp(14.0))
      , _rhoc(
          y3_cluster::make_Interp1D(sample, "haloModel", "z", "rhoc")
            .clamp(0.0))
      , _nfwProfile(read_vector(logx_file(GAMMA)),
                    read_vector(logxmis_file(GAMMA)),
                    read_vector(log_sigma_file(GAMMA)))
    {}


    // UNIFIED rho_m convention (2026-08-24 decision): use `rho` --
    // haloModel/rho_m_ref = Omega_m rho_crit,0 (1+z_density)^3, identical
    // to the density first_halo_term builds the centred tables with --
    // for BOTH the halo boundary r_200 = [3M/(800 pi rho)]^(1/3) and the
    // amplitude rho_s = delta_c * rho. Host-only (call before the device
    // copy). Production call sites use this; pure normalization factors
    // (legacy Omega_m, the physical (1+z)^2) are applied OUTSIDE.
    void set_rho_ref(double rho) { _rho_b = rho; }

    __device__ __host__ double
    operator()(double r, double rmis, double lnM) const
    {
      double const delta_c = (200.0 * _c * _c * _c / 3.0) / (std::log(1.0 + _c) - _c / (1.0 + _c));
      double const r_200 =
        std::cbrt(3.0 * std::exp(lnM) / (800.0 * M_PI * _rho_b));
      double const r_s = r_200 / _c;

      // normalization term defined in Wright & Brainerd 2000
      double const norm = 2 * r_s * delta_c * _rho_b;

      double const x = r / r_s;
      double const xmis = rmis / r_s;
      double const log_unfw = _nfwProfile.clamp(std::log(x), std::log(xmis));
      double const nfw = norm * std::exp(log_unfw);

      // Conversion from Msun/Mpc^2 to Msun/h pc^2
      return nfw*1e-12;
    }

  private:
    double const _c;
    double const _rhoc;
    double       _rho_b{_rhoc};   // boundary+amplitude density (set_rho_ref)
    gpu_support::Interp2D _nfwProfile;
  };
}
#endif
