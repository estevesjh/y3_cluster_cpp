// sigma_prj_gpu_t.cuh -- GPU integrand for Costanzi-2026 projection shear.
// -------------------- in progress --------------------
// This is the CUDA-side analogue of sigma_prj_t.hh.  It deliberately does
// NOT use the old shortcut
//      DSigma_hh * bias * b_sel * SigmaCritInv
//
// Physics matched to sigma_prj_t.hh / ShearPrjEvaluator:
//
//   Sigma_prj(R | lob,zob) = int dtheta dz dlnM
//       2*pi*sin(theta) * dV/dz/dOmega(z) * w_z(z,zob) * n(M,z)
//       * [ 1 + b(M,z) b_sel(theta|lob,zob) xi_NL(DeltaChi,zob) ]
//       * Sigma_mis(R | M,z, Rmis = theta * D_A(zob))
//
// and the same expression with DeltaSigma_mis for DSigma_prj.
//
// The WeightF template selects one of the four pieces:
//   SigmaRndWeight, SigmaClWeight, DSigmaRndWeight, DSigmaClWeight.
// A wrapper module should integrate the four pieces and then publish
//   sigma_prj/{vals,rnd,cl}
//   dsigma_prj/{vals,rnd,cl}
//   shear_prj/{vals,rnd,cl}, with shear_prj = dsigma_prj * SigmaCritInv.

#ifndef Y3_CLUSTER_CPP_SIGMA_PRJ_GPU_T_CUH
#define Y3_CLUSTER_CPP_SIGMA_PRJ_GPU_T_CUH

#include "cosmosis/datablock/datablock.hh"
#include "cosmosis/datablock/ndarray.hh"

#include "common/cuda/Volume.cuh"
#include "common/cuda/Interp2D.cuh"

#include "utils/datablock_reader.hh"
#include "utils/make_grid_points.hh"
#include "utils/make_interp_1d.cuh"
#include "utils/make_interp_2d.cuh"

#include "models/dv_do_dz_t.cuh"
#include "models/hmf_t.cuh"
#include "models/sigma_photoz_table_t.cuh" // This replaces SIGMA_PHOTOZ_DES_t which uses a polynomial fit 

// These two are the GPU equivalents of the CPU headers used by sigma_prj_t.hh:
//   models/nfw_sigma_mis.hh
//   models/nfw_dsigma_mis.hh
// They must evaluate the SINGLE miscentering kernel, not the DES gamma miscentering distribution.
#include "models/nfw_sigma_mis.cuh"
#include "models/nfw_dsigma_mis.cuh"

#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>

namespace y3_cuda {

namespace sigma_prj_gpu_detail {

static constexpr double PI = 3.14159265358979323846;
static constexpr int MAX_BSEL = 256;

__host__ __device__ inline double
R_lambda(double lob_center)
{
  return pow(lob_center / 100.0, 0.2);
}

__host__ __device__ inline double
lob_center_from_bin(int bin)
{
  double const centres[4] = {25.0, 37.5, 52.5, 130.0};
  return (bin >= 0 && bin < 4) ? centres[bin] : 25.0;
}

__host__ __device__ inline double
photoz_weight(double z, double zob, double sigma_z)
{
  double const u = (z - zob) / sigma_z;
  return (fabs(u) < 1.0) ? (1.0 - u * u) : 0.0;
}

__host__ __device__ inline double
theta_excl_at_z(double chi_z, double chi_o, double R_excl)
{
  double const denom = 2.0 * chi_z * chi_o + 1.0e-30;
  double cos_ex = (chi_z * chi_z + chi_o * chi_o - R_excl * R_excl) / denom;
  if (cos_ex >= 1.0 - 1.0e-12) return 0.0;
  if (cos_ex < -1.0) cos_ex = -1.0;
  return acos(cos_ex);
}

__host__ __device__ inline double
bsel_sigmoid(double theta, double theta_lam, double B_small, double B_large)
{
  double const k_sig  = 2.5 / theta_lam;
  double const theta0 = 0.5 * theta_lam;
  double const sgm = 1.0 / (1.0 + exp(-k_sig * (theta - theta0)));
  return B_small + (B_large - B_small) * sgm;
}

}  // namespace sigma_prj_gpu_detail

// -----------------------------------------------------------------------------
// Weight functors.  These reproduce the CPU decomposition:
//   rnd: the "1" term
//   cl : the b(M,z) * b_sel(theta) * xi_NL term, with exclusion applied
// -----------------------------------------------------------------------------
struct SigmaRndWeight {
  __host__ __device__ static double apply(double common,
                                          double Sigma_v,
                                          double /*DSigma_v*/,
                                          double /*bias*/, double /*bsel*/,
                                          double /*xi*/, bool /*has_cl*/)
  {
    return common * Sigma_v;
  }
};

struct SigmaClWeight {
  __host__ __device__ static double apply(double common,
                                          double Sigma_v,
                                          double /*DSigma_v*/,
                                          double bias, double bsel,
                                          double xi, bool has_cl)
  {
    return has_cl ? common * bias * bsel * xi * Sigma_v : 0.0;
  }
};

struct DSigmaRndWeight {
  __host__ __device__ static double apply(double common,
                                          double /*Sigma_v*/,
                                          double DSigma_v,
                                          double /*bias*/, double /*bsel*/,
                                          double /*xi*/, bool /*has_cl*/)
  {
    return common * DSigma_v;
  }
};

struct DSigmaClWeight {
  __host__ __device__ static double apply(double common,
                                          double /*Sigma_v*/,
                                          double DSigma_v,
                                          double bias, double bsel,
                                          double xi, bool has_cl)
  {
    return has_cl ? common * bias * bsel * xi * DSigma_v : 0.0;
  }
};

// total = rnd + cl in one pass, i.e. the literal [1 + b*b_sel*xi_NL]
// bracket of the Costanzi-2026 Sigma_prj integrand (validation_report.tex
// Eq. sigma_prj) applied to DSigma_mis. Use this instead of DSigmaRndWeight
// when shear_prj/vals should be the full projection term (rnd+cl), not just
// the uncorrelated background piece.
struct DSigmaTotalWeight {
  __host__ __device__ static double apply(double common,
                                          double /*Sigma_v*/,
                                          double DSigma_v,
                                          double bias, double bsel,
                                          double xi, bool has_cl)
  {
    double const cl = has_cl ? bias * bsel * xi : 0.0;
    return common * (1.0 + cl) * DSigma_v;
  }
};

// -----------------------------------------------------------------------------
// SigmaPrjGPU<WeightF>
//
// Integration variables: (theta, z, lnM)
// Grid point: (lambda_bin, zo_low, zo_high, R)
// -----------------------------------------------------------------------------
template <class WeightF>
class SigmaPrjGPU {
public:
  using grid_t       = y3_cluster::grid_t<4>;
  using grid_point_t = grid_t::value_type;

private:
  using volume_t = quad::Volume<double, 3>;

  std::optional<y3_cuda::HMF_t>       hmf_;
  std::optional<y3_cuda::DV_DO_DZ_t>  dv_do_dz_;

  quad::Interp2D hmb_;    // haloModel/bias(lnM,z)
  quad::Interp2D xi_nl_;  // xi_nl(r,z)
  quad::Interp1D chi_;    // distances/d_c(z)

  // GPU NFW miscentering kernels.  These must be the SINGLE kernel used by
  // projection, matching sigma_prj_t.hh lines where NFW_SIG_SINGLE/SINGLE are
  // selected.
  std::optional<NFW_SIGMA_MIS>  sigma_mis_;
  std::optional<NFW_DSIGMA_MIS> dsigma_mis_;

  // bSelMargGPU output is a flattened wall grid, not unique axes:
  //   b_sel_marginalised/{lob,zob,b_small,b_large}
  double b_lob_[sigma_prj_gpu_detail::MAX_BSEL];
  double b_zob_[sigma_prj_gpu_detail::MAX_BSEL];
  double b_small_[sigma_prj_gpu_detail::MAX_BSEL];
  double b_large_[sigma_prj_gpu_detail::MAX_BSEL];
  int n_bsel_ = 0;

  double omega_m_ = 0.286;
  double h0_      = 0.7;

  // Current grid point.
  int lob_bin_ = 0;
  double lob_center_ = 25.0;
  double zo_low_ = 0.0;
  double zo_high_ = 0.0;
  double zob_ = 0.0;
  double R_ = 0.0;

public:
  explicit SigmaPrjGPU(cosmosis::DataBlock& /*cfg*/) {}

  void set_sample(cosmosis::DataBlock& sample)
  {
    hmf_.emplace(sample);
    dv_do_dz_.emplace(sample);

    hmb_   = make_Interp2D(sample, "haloModel", "lnM", "z", "bias");
    xi_nl_ = make_Interp2D(sample, "xi_nl", "r", "z", "xi_nl");
    chi_   = make_Interp1D(sample, "distances", "z", "d_c");

    omega_m_ = sample.view<double>("cosmological_parameters", "omega_M");
    h0_      = sample.view<double>("cosmological_parameters", "h0");

    // Same normalisation convention as CPU sigma_prj_t.hh:
    // projection uses rho_mean = Omega_m * rho_crit.
    //
    // PORT NOTE (des_y3 merge): this originally read
    //     sigma_mis_->set_rho_mult(omega_m_);
    //     dsigma_mis_->set_rho_mult(omega_m_);
    // set_rho_mult was REMOVED from NFW_{,D}SIGMA_MIS by the unified
    // rho_m reference-density convention (6f812e2). Under the old hybrid
    // the halo boundary r_200 came from rho_crit while the amplitude
    // carried an extra Omega_m factor; the unified convention uses ONE
    // density for both, published always-on as haloModel/rho_m_ref =
    // Omega_m * rho_crit,0 * (1+one_halo_z_density)^3.
    //
    // With the default one_halo_z_density = 0, rho_m_ref is numerically
    // Omega_m * rho_crit,0 -- i.e. exactly the "rho_mean = Omega_m *
    // rho_crit" this comment asks for -- so the AMPLITUDE is unchanged
    // and only the r_200 boundary moves onto the same density. That
    // boundary change is deliberate: it is the fix for the 56-86%
    // radial-series offset (docs/known_issues/
    // radial_series_vs_full_ltmz_defect.md), and it is what makes this
    // module comparable to ShearPrjGl/ShearPrjCore, which read the same
    // haloModel/rho_m_ref.
    double const rho_ref =
      sample.has_val("haloModel", "rho_m_ref")
        ? sample.view<double>("haloModel", "rho_m_ref")
        : omega_m_ * 2.77533742639e+11;
    sigma_mis_.emplace(4.0, 2.77533742639e+11, std::string("single"));
    dsigma_mis_.emplace(4.0, 2.77533742639e+11, std::string("single"));
    sigma_mis_->set_rho_ref(rho_ref);
    dsigma_mis_->set_rho_ref(rho_ref);

    for (int i = 0; i < sigma_prj_gpu_detail::MAX_BSEL; ++i) {
      b_lob_[i] = 0.0;
      b_zob_[i] = 0.0;
      b_small_[i] = 1.0;
      b_large_[i] = 1.0;
    }

    if (!sample.has_val("b_sel_marginalised", "b_small") ||
        !sample.has_val("b_sel_marginalised", "b_large") ||
        !sample.has_val("b_sel_marginalised", "lob") ||
        !sample.has_val("b_sel_marginalised", "zob")) {
      throw std::runtime_error(
          "SigmaPrjGPU needs b_sel_marginalised/{lob,zob,b_small,b_large}. "
          "Run bSelMargGPU before shear_prj.");
    }

    auto const& nd_lob = sample.view<cosmosis::ndarray<double>>(
        "b_sel_marginalised", "lob");
    auto const& nd_zob = sample.view<cosmosis::ndarray<double>>(
        "b_sel_marginalised", "zob");
    auto const& nd_s = sample.view<cosmosis::ndarray<double>>(
        "b_sel_marginalised", "b_small");
    auto const& nd_l = sample.view<cosmosis::ndarray<double>>(
        "b_sel_marginalised", "b_large");

    int const n = static_cast<int>(nd_s.size());
    n_bsel_ = (n < sigma_prj_gpu_detail::MAX_BSEL)
            ? n : sigma_prj_gpu_detail::MAX_BSEL;

    auto it_lob = nd_lob.begin();
    auto it_zob = nd_zob.begin();
    auto it_s   = nd_s.begin();
    auto it_l   = nd_l.begin();
    for (int i = 0; i < n_bsel_; ++i) {
      b_lob_[i]   = *it_lob++;
      b_zob_[i]   = *it_zob++;
      b_small_[i] = *it_s++;
      b_large_[i] = *it_l++;
    }
  }

  void set_grid_point(grid_point_t const& pt)
  {
    lob_bin_    = static_cast<int>(pt[0]);
    lob_center_ = sigma_prj_gpu_detail::lob_center_from_bin(lob_bin_);
    zo_low_     = pt[1];
    zo_high_    = pt[2];
    zob_        = 0.5 * (zo_low_ + zo_high_);
    R_          = pt[3];
  }

  size_t get_device_mem_footprint()
  {
    size_t size = 0;
    if (hmf_) size += hmf_->get_device_mem_footprint();
    if (dv_do_dz_) size += dv_do_dz_->get_device_mem_footprint();
    size += hmb_.get_device_mem_footprint();
    size += xi_nl_.get_device_mem_footprint();
    size += chi_.get_device_mem_footprint();
    //if (sigma_mis_) size += sigma_mis_->get_device_mem_footprint();
    //if (dsigma_mis_) size += dsigma_mis_->get_device_mem_footprint();
    return size;
  }

  __host__ __device__ double
  operator()(double theta, double z, double lnM) const
  {
    using namespace sigma_prj_gpu_detail;

    // DES photo-z projection kernel w_z(z,zob), same parabolic form as CPU.
    SIGMA_PHOTOZ_TABLE_t sigma_z_model;
    double const sig_z = sigma_z_model(z);;
    double const wz = photoz_weight(z, zob_, sig_z);
    if (wz <= 0.0) return 0.0;

    double const chi_o = chi_.clamp(zob_) * h0_;
    double const chi_z = chi_.clamp(z)    * h0_;
    double const D_A_o = chi_o / (1.0 + zob_);

    double const Rlam   = R_lambda(lob_center_);
    double const R_excl = Rlam * (1.0 + zob_);
    double const theta_lam = Rlam * (1.0 + zob_) / chi_o;

    double const sin_th = sin(theta);
    double const cos_th = cos(theta);

    double const dV   = (*dv_do_dz_)(z);
    double const hmfv = (*hmf_)(lnM, z);
    double const common = 2.0 * PI * sin_th * dV * wz * hmfv;
    if (common == 0.0) return 0.0;

    // Correct projection geometry: Rmis = theta * D_A(zob).
    double const Rmis = theta * D_A_o;
    double const Sigma_v  = (*sigma_mis_ )(R_, Rmis, lnM);
    double const DSigma_v = (*dsigma_mis_)(R_, Rmis, lnM);

    // Exclusion + clustering piece.
    bool has_cl = false;
    double xi_val = 0.0;
    double bias_val = 0.0;
    double bsel_val = 1.0;

    double const theta_excl = theta_excl_at_z(chi_z, chi_o, R_excl);
    if (theta > theta_excl) {
      double const dchi = sqrt(fmax(chi_z * chi_z + chi_o * chi_o
                              - 2.0 * chi_z * chi_o * cos_th, 0.0));
      xi_val = xi_nl_.clamp(dchi, zob_);
      bias_val = hmb_.clamp(lnM, z);

      double Bsmall = 1.0;
      double Blarge = 1.0;
      get_b_asymptotes_(Bsmall, Blarge);
      bsel_val = bsel_sigmoid(theta, theta_lam, Bsmall, Blarge);
      has_cl = true;
    }

    return WeightF::apply(common, Sigma_v, DSigma_v,
                          bias_val, bsel_val, xi_val, has_cl);
  }

  __host__ __device__ void
  get_b_asymptotes_(double& Bsmall, double& Blarge) const
  {
    // bSelMargGPU writes flattened (lob,zob) rows.  Pick the closest row.
    // This is robust for the current wall-grid; if a later bSel module writes
    // unique axes, this can be upgraded to z interpolation like CPU.
    double best = 1.0e300;
    int best_i = -1;
    for (int i = 0; i < n_bsel_; ++i) {
      double const dl = fabs(b_lob_[i] - lob_center_);
      double const dz = fabs(b_zob_[i] - zob_);
      double const score = dl / 100.0 + dz;
      if (score < best) {
        best = score;
        best_i = i;
      }
    }
    if (best_i >= 0) {
      Bsmall = b_small_[best_i];
      Blarge = b_large_[best_i];
    }
  }

  static char const* module_label() { return "shear_prj"; }

  static std::vector<volume_t>
  make_integration_volumes(cosmosis::DataBlock& cfg)
  {
    volume_t vol;
    char const* label = module_label();

    vol.lows [0] = cfg.view<double>(label, "theta_low");
    vol.highs[0] = cfg.view<double>(label, "theta_high");
    vol.lows [1] = cfg.view<double>(label, "zt_low");
    vol.highs[1] = cfg.view<double>(label, "zt_high");
    vol.lows [2] = cfg.view<double>(label, "lnm_low");
    vol.highs[2] = cfg.view<double>(label, "lnm_high");

    // Zipped mode needs one integration volume per grid point.
    // The shear_prj grid is the wall-of-numbers list:
    //   (lambda_bin, zo_low, zo_high, radii)
    auto const& radii = cfg.view<std::vector<double>>(label, "radii");
    return std::vector<volume_t>(radii.size(), vol);
  }

  static grid_t make_grid_points(cosmosis::DataBlock& cfg)
  {
    return y3_cluster::make_grid_points_wall_of_numbers(
        cfg, module_label(), "lambda_bin", "zo_low", "zo_high", "radii");
  }
  
};

}  // namespace y3_cuda

#endif
