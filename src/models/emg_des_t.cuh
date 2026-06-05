#ifndef Y3_CLUSTER_CPP_EMG_DES_T_CUH
#define Y3_CLUSTER_CPP_EMG_DES_T_CUH

#include "cosmosis/datablock/datablock.hh"
#include "common/cuda/Interp1D.cuh"
#include "utils/make_interp_1d.cuh"
#include "utils/primitives.cuh"

#include <vector>

namespace y3_cuda {

namespace emg_constants {
  __host__ __device__ constexpr double SQRT2 = 1.4142135623730951;
}

class EMG_DES_t {
private:
  quad::Interp1D a_mu_;
  quad::Interp1D b_mu_;
  quad::Interp1D a_sig_;
  quad::Interp1D b_sig_;
  quad::Interp1D a_tau_;
  quad::Interp1D b_tau_;
  quad::Interp1D a_fprj_;
  quad::Interp1D b_fprj_;

public:
  size_t get_device_mem_footprint()
  {
    size_t size = 0;
    size += a_mu_.get_device_mem_footprint();
    size += b_mu_.get_device_mem_footprint();
    size += a_sig_.get_device_mem_footprint();
    size += b_sig_.get_device_mem_footprint();
    size += a_tau_.get_device_mem_footprint();
    size += b_tau_.get_device_mem_footprint();
    size += a_fprj_.get_device_mem_footprint();
    size += b_fprj_.get_device_mem_footprint();
    return size;
  }

  explicit EMG_DES_t(cosmosis::DataBlock& sample)
    : a_mu_(make_Interp1D(sample, "plob_ltr_params", "z", "a_mu"))
    , b_mu_(make_Interp1D(sample, "plob_ltr_params", "z", "b_mu"))
    , a_sig_(make_Interp1D(sample, "plob_ltr_params", "z", "a_sig"))
    , b_sig_(make_Interp1D(sample, "plob_ltr_params", "z", "b_sig"))
    , a_tau_(make_Interp1D(sample, "plob_ltr_params", "z", "a_tau"))
    , b_tau_(make_Interp1D(sample, "plob_ltr_params", "z", "b_tau"))
    , a_fprj_(make_Interp1D(sample, "plob_ltr_params", "z", "a_fprj"))
    , b_fprj_(make_Interp1D(sample, "plob_ltr_params", "z", "b_fprj"))
  {}

  __host__ __device__ void
  get_params(double ltr, double z,
             double& mu, double& sigma, double& tau, double& fprj) const
  {
    double const ltr_safe = fmax(ltr, 0.5);

    double const a_mu_z = a_mu_.clamp(z);
    double const b_mu_z = b_mu_.clamp(z);
    double const a_sig_z = a_sig_.clamp(z);
    double const b_sig_z = b_sig_.clamp(z);
    double const a_tau_z = a_tau_.clamp(z);
    double const b_tau_z = b_tau_.clamp(z);
    double const a_fprj_z = a_fprj_.clamp(z);
    double const b_fprj_z = b_fprj_.clamp(z);

    mu = a_mu_z + b_mu_z * ltr_safe;
    sigma = b_sig_z * pow(ltr_safe, a_sig_z);
    tau = b_tau_z / pow(ltr_safe, a_tau_z);

    double const denom = pow(1.0 + exp(-ltr_safe), a_fprj_z);
    fprj = fmin(1.0, b_fprj_z / fmax(denom, 1e-10));

    sigma = fmax(sigma, 1e-6);
    tau = fmax(tau, 1e-6);
    fprj = fmax(0.0, fmin(1.0, fprj));
  }

  // P(lambda_ob | lambda_tr, z)
  // Direct PDF:
  // (1 - fprj) * Gaussian + fprj * EMG
  __host__ __device__ double
  operator()(double lob, double ltr, double z) const
  {
    double mu, sigma, tau, fprj;
    get_params(ltr, z, mu, sigma, tau, fprj);

    double const gauss =
        y3_cuda::gaussian(lob, mu, sigma);

    double const exp_arg =
        0.5 * tau * (2.0 * mu + tau * sigma * sigma - 2.0 * lob);

    double const erfc_arg =
        (mu + tau * sigma * sigma - lob)
        / (emg_constants::SQRT2 * sigma);

    double const emg =
        0.5 * tau * exp(exp_arg) * erfc(erfc_arg);

    return (1.0 - fprj) * gauss + fprj * emg;
  }

  __host__ __device__ double
  pdf_with_params(double lob, double ltr, double z,
                  double& mu_out, double& sigma_out,
                  double& tau_out, double& fprj_out) const
  {
    get_params(ltr, z, mu_out, sigma_out, tau_out, fprj_out);
    return (*this)(lob, ltr, z);
  }
};

} // namespace y3_cuda

#endif

