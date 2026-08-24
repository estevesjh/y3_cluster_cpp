// Projection shear — full (theta, z, lnM)-resolved reference, CUDA/PAGANI.
//
// Follows the standard CUDA integration-module template
// (DEFINE_COSMOSIS_CUDA_INTEGRATION_MODULE, same as gt_card_gpu and the
// des_y3 explicit 3d GPU modules): one adaptive PAGANI triple integral
// per wall point over the continuous projection integrand,
//
//   dSigma_prj(R) = ∫dθ ∫dzt ∫dlnM  2π sinθ · dV/dΩdz(zt) ·
//       w_pz(zt; z_ob) · n(M, zt) ·
//       [ 1 + b(M, zt) · b_sel(θ) · ξ_NL(|Δχ|, z_ob) · 1(θ > θ_excl(zt)) ]
//       · ΔΣ_mis(R | M, θ·D_A(z_ob))                    [single kernel]
//
// — exactly the observable the fixed-GL fixed-GL backends compute
// (exact z, ξ_NL frozen at z_ob, analytic b_sel plateaus, no Ω(z),
// slab exclusion on the clustered channel only), with the adaptive
// quadrature of its CPU precedent ShearPrjCuhre. Every table rides to
// the device as quad::Interp1D / quad::Interp2D: χ(z) = d_c·h0, σ_z(z)
// (the compiled z_kernel table), halo bias, ξ_NL, and the single-offset
// NFW look-up inside y3_cuda::NFW_DSIGMA_MIS (its CUDA reader predates
// set_rho_mult, so the Ω_m mean-density factor is applied in the
// integrand, host-convention identical).
//
// Grid: the 180-point zipped wall (lambda_bin, zo_low, zo_high, radii);
// volumes: per-row (theta, zt, lnm) bounds. Output:
// dsigmaprjfastmassgpu/{vals, errors, probs, status, nregions}
// (ΔΣ_prj total = rnd + cl, Msun/(h pc^2)).
// Status: reference backend. Production remains ShearPrjFrozenPhysics.
#ifndef Y3_CLUSTER_CPP_DSIGMA_PRJ_3D_GPU_T_CUH
#define Y3_CLUSTER_CPP_DSIGMA_PRJ_3D_GPU_T_CUH

#include "utils/cuda_module_macros.cuh"
#include "utils/datablock_reader.hh"
#include "utils/make_cuda_integration_volumes.cuh"
#include "utils/make_grid_points.hh"
#include "utils/make_interp_1d.cuh"
#include "utils/make_interp_2d.cuh"

#include "cosmosis/datablock/datablock.hh"
#include "cosmosis/datablock/datablock_status.h"
#include "cubacpp/integration_result.hh"
#include "common/cuda/Volume.cuh"

#include "models/dv_do_dz_t.cuh"
#include "models/hmf_t.cuh"
#include "models/nfw_dsigma_mis.cuh"
#include "models/z_kernel_data.hh"

#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>

class DSigmaPrj3dGpu {
public:
  using grid_t = y3_cluster::grid_t<4>;
  using grid_point_t = grid_t::value_type;

private:
  using volume_t = quad::Volume<double, 3>;

  std::vector<double> lob_centers_;
  double h0_{0.7}, omega_m_{0.3};

  std::optional<y3_cuda::HMF_t> hmf_;
  std::optional<y3_cuda::DV_DO_DZ_t> dv_do_dz_;
  std::optional<quad::Interp1D> chi_;      // d_c(z), Mpc (x h0 in use)
  std::optional<quad::Interp1D> sigma_z_;  // compiled z_kernel table
  std::optional<quad::Interp2D> bias_;     // b(lnM, z)
  std::optional<quad::Interp2D> xi_nl_;    // xi_NL(r, z)
  std::optional<y3_cuda::NFW_DSIGMA_MIS> dsigma_mis_;   // 'single'

  // Host-side copies for set_grid_point geometry.
  std::vector<double> h_dist_z_, h_d_c_;
  std::vector<double> h_bs_lob_, h_bs_zob_, h_b_small_, h_b_large_;

  // Per-wall-point scalars (host-set, device-read).
  double zob_{0}, chi_o_{0}, d_a_o_{0}, r_excl_{0}, cur_R_{0};
  double b_small_{0}, b_large_{0}, k_sig_{0}, theta0_{0};

  double
  host_chi(double z) const
  {
    double const zc = std::min(std::max(z, h_dist_z_.front()),
                               h_dist_z_.back());
    auto const it = std::upper_bound(h_dist_z_.begin(), h_dist_z_.end(), zc);
    std::size_t const i =
      std::min<std::size_t>(std::max<long>(1, it - h_dist_z_.begin()),
                            h_dist_z_.size() - 1);
    double const t = (zc - h_dist_z_[i - 1]) /
                     (h_dist_z_[i] - h_dist_z_[i - 1]);
    return (h_d_c_[i - 1] + t * (h_d_c_[i] - h_d_c_[i - 1])) * h0_;
  }

public:
  explicit DSigmaPrj3dGpu(cosmosis::DataBlock& cfg)
  {
    lob_centers_ = {25.0, 37.5, 52.5, 130.0};
    if (cfg.has_val(module_label(), "lob_centers"))
      lob_centers_ = get_vector_double(cfg, module_label(), "lob_centers");
    if (lob_centers_.empty())
      throw std::runtime_error("DSigmaPrj3dGpu: lob_centers empty");
  }

  void
  set_sample(cosmosis::DataBlock& s)
  {
    hmf_.emplace(s);
    dv_do_dz_.emplace(s);
    chi_.emplace(make_Interp1D(s, "distances", "z", "d_c"));
    auto const zk_z = y3_cluster::z_kernel_z();
    auto const zk_s = y3_cluster::z_kernel_sigma();
    sigma_z_.emplace(zk_z.data(), zk_s.data(), zk_z.size());
    bias_.emplace(make_Interp2D(s, "haloModel", "lnM", "z", "bias"));
    xi_nl_.emplace(make_Interp2D(s, "xi_nl", "r", "z", "xi_nl"));
    dsigma_mis_.emplace(4.0, 2.77533742639e+11, "single");
    h0_ = s.view<double>("cosmological_parameters", "h0");
    omega_m_ = s.view<double>("cosmological_parameters", "omega_M");

    using doubles = std::vector<double>;
    h_dist_z_ = s.view<doubles>("distances", "z");
    h_d_c_ = s.view<doubles>("distances", "d_c");
    h_bs_lob_ = s.view<doubles>("b_sel_marginalised", "lob");
    h_bs_zob_ = s.view<doubles>("b_sel_marginalised", "zob");
    auto flat = [&s](char const* key) {
      auto const nd = s.view<cosmosis::ndarray<double>>(
        "b_sel_marginalised", key);
      return std::vector<double>(nd.begin(), nd.end());
    };
    h_b_small_ = flat("b_small");
    h_b_large_ = flat("b_large");
  }

  void
  set_grid_point(grid_point_t const& pt)
  {
    int const lob_bin = static_cast<int>(pt[0]);
    zob_ = 0.5 * (pt[1] + pt[2]);
    cur_R_ = pt[3];
    double const lobc = lob_centers_.at(lob_bin % lob_centers_.size());
    chi_o_ = host_chi(zob_);
    d_a_o_ = chi_o_ / (1.0 + zob_);
    r_excl_ = std::pow(lobc / 100.0, 0.2) * (1.0 + zob_);

    // b_sel plateaus: bracket zob in the 3-node table, linear, clamped
    // (interp_b_asymptotes_ semantics).
    int const n_zob = static_cast<int>(h_bs_zob_.size());
    int const n_lob = static_cast<int>(h_bs_lob_.size());
    int j = 0;
    while (j + 1 < n_zob && h_bs_zob_[j + 1] < zob_) ++j;
    int const j0 = (j + 1 < n_zob) ? j : n_zob - 2;
    int const j1 = j0 + 1;
    double f = (h_bs_zob_[j1] > h_bs_zob_[j0])
                 ? (zob_ - h_bs_zob_[j0]) /
                     (h_bs_zob_[j1] - h_bs_zob_[j0])
                 : 0.0;
    f = std::min(1.0, std::max(0.0, f));
    b_small_ = (1 - f) * h_b_small_[j0 * n_lob + lob_bin] +
               f * h_b_small_[j1 * n_lob + lob_bin];
    b_large_ = (1 - f) * h_b_large_[j0 * n_lob + lob_bin] +
               f * h_b_large_[j1 * n_lob + lob_bin];
    double const theta_lam =
      std::pow(lobc / 100.0, 0.2) * (1.0 + zob_) / chi_o_;
    k_sig_ = 2.5 / theta_lam;
    theta0_ = 0.5 * theta_lam;
  }

  // First variable is ln(theta): the DSigma_mis feature at
  // theta_R = R/D_A is ~1e-5 wide in linear theta for the smallest
  // radii, and PAGANI silently returned 0 for ~20% of wall points on a
  // linear-theta volume (statuses all converged) -- the same reason the
  // fixed-GL core builds its theta grid in log theta. The exp/Jacobian
  // makes every feature O(1) wide in the integration variable.
  __host__ __device__ double
  operator()(double lntheta, double zt, double lnM) const
  {
    double const theta = exp(lntheta);
    double const sig = sigma_z_->clamp(zt);
    double const u = (zt - zob_) / sig;
    if (fabs(u) >= 1.0) return 0.0;
    double const w_pz = 1.0 - u * u;

    double const chi_z = chi_->clamp(zt) * h0_;
    double const dchi2 = chi_z * chi_z + chi_o_ * chi_o_ -
                         2.0 * chi_z * chi_o_ * cos(theta);
    double const dchi = sqrt(fmax(dchi2, 0.0));

    // LoS slab exclusion (clustered channel only).
    double const denom = 2.0 * chi_z * chi_o_ + 1.0e-30;
    double cos_ex = (chi_z * chi_z + chi_o_ * chi_o_ -
                     r_excl_ * r_excl_) / denom;
    double theta_excl = 0.0;
    if (cos_ex < 1.0 - 1.0e-12) {
      cos_ex = fmax(cos_ex, -1.0);
      theta_excl = acos(cos_ex);
    }
    double cl = 0.0;
    if (theta > theta_excl) {
      double const bsel =
        b_small_ + (b_large_ - b_small_) /
                     (1.0 + exp(-k_sig_ * (theta - theta0_)));
      cl = bias_->clamp(lnM, zt) * bsel * xi_nl_->clamp(dchi, zob_);
    }

    double const dsmis =
      omega_m_ * (*dsigma_mis_)(cur_R_, theta * d_a_o_, lnM);
    return theta * 2.0 * M_PI * sin(theta) * (*dv_do_dz_)(zt) * w_pz *
           (*hmf_)(lnM, zt) * (1.0 + cl) * dsmis;   // x theta: dtheta = theta dlntheta
  }

  static char const* module_label() { return "DSigmaPrj3dGpu"; }

  static std::vector<volume_t>
  make_integration_volumes(cosmosis::DataBlock& cfg)
  {
    return y3_cuda::make_integration_volumes_wall_of_numbers(
      cfg, module_label(), "lntheta", "zt", "lnm");
  }

  static grid_t
  make_grid_points(cosmosis::DataBlock& cfg)
  {
    return y3_cluster::make_grid_points_wall_of_numbers(
      cfg, module_label(), "lambda_bin", "zo_low", "zo_high", "radii");
  }
};

#endif
