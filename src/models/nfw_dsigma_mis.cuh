// Off-Centered Sigma NFW Profile
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

#include "utils/cuda_interp_2d.cuh"
#include "utils/make_interp_1d.hh"
#include "utils/read_vector.hh"

namespace y3_cuda {
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

  class NFW_DSIGMA_MIS {

    public:
    NFW_DSIGMA_MIS(double c, double rhoc, std::string const& kernel)
      : _c(c),
        _rhoc(rhoc),
        _nfwProfile(read_vector(logx_file(kernel)),
                    read_vector(logxmis_file(kernel)),
                    read_vector(log_dsigma_file(kernel)))
    { }

    NFW_DSIGMA_MIS()
    : _c(CONC),
      _rhoc(RHOC),
      _nfwProfile(read_vector(logx_file(GAMMA)),
                  read_vector(logxmis_file(GAMMA)),
                  read_vector(log_dsigma_file(GAMMA)))

    { }

    // TODO(issue #13, needs a Perlmutter CUDA build to implement+test):
    // mirror the CPU use_halo_model_conc option (src/models/
    // nfw_dsigma_mis.hh::set_concentration_table / conc_at). The lookup
    // table is universal in x = r/r_s, so the change is purely analytic
    // (r_s = r_200/c, delta_c(c)); the conc table is
    // haloModel/{lnM, concentration} (Child18 at one_halo_z), device
    // side via quad::Interp1D with clamp. Until then the CUDA prj
    // modules support ONLY the legacy fixed c=4 default -- which is
    // what every cross-backend identity pin tests, so they stay valid;
    // do NOT set use_halo_model_conc=T on a GPU module.
    // TODO: Implement different operator in case of rhocz(zt)
    // NFW_DSIGMA_MIS(cosmosis::DataBlock& sample)
    // : _c(y3_cluster::make_Interp1D(sample,"haloModel","lnM","concentration").clamp(14.0))
    // , _rhoc(y3_cluster::make_Interp1D(sample,"haloModel","z","rhoc").clamp(0.0))
    // , _nfwProfile(read_vector(logx_file(GAMMA)),
    //               read_vector(logxmis_file(GAMMA)),
    //               read_vector(log_dsigma_file(GAMMA)))
    // { }

    __device__ __host__ double
    operator()(double r, double rmis, double lnM) const 
    {
      double const rho_crit = _rhoc;
      double const delta_c = (200.0 * _c * _c * _c / 3.0) / (std::log(1.0 + _c) - _c / (1.0 + _c));
      double const r_200 = std::cbrt(3.0 * std::exp(lnM) / (800.0 * M_PI * rho_crit));
      double const r_s = r_200 / _c;

      double const x = r / r_s;
      double const xmis = rmis / r_s;

      double const log_unfw = _nfwProfile.clamp(std::log(x), std::log(xmis));
      
      // normalization term defined in Wright & Brainerd 2000
      double const norm = 2 * r_s * delta_c * rho_crit ;
      double const nfw = norm * std::exp(log_unfw);

      // Conversion from Msun/Mpc^2 to Msun/h pc^2
      return nfw*1e-12;
    }

  private:
    double const _c;
    double const _rhoc;
    gpu_support::Interp2D _nfwProfile;
  };
}
#endif

// USED FOR DEBUGGING
// using printf debugs r200 and r_s
// printf("r200: %f, r_s: %f\n", r_200, r_s);
// printf("delta_c: %f, rho_crit: %f\n", delta_c, rho_crit);
// printf("norm: %f\n", norm);
// printf("conc: %f\n", _c);