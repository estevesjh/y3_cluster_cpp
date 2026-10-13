// Max-model lensing profile by direct projection of xi_max — no approximations.
// C++ counterpart of operator_max1h2h_gl.py.
//
// For every halo (lnM, z):
//
//   xi_max(r)     = max(xi_nfw(r, lnM), b(lnM, z) * xi_hh(r, z))
//   Sigma(R)      = 2 rho_m int_0^l_max xi_max(sqrt(R^2 + l^2)) dl
//   barSigma(<R)  = (2 / R^2) int_0^R R' Sigma(R') dR'
//   DeltaSigma(R) = barSigma(<R) - Sigma(R)
//
// Both integrals use fixed Gauss-Legendre nodes:
//   line of sight: l = R sinh(u), u in [0, acosh(r_max / R)], dl = r du with
//                  r = R cosh(u), so the integral stops at the 3D radius r_max.
//   aperture:      R' = R t^2, t in [0, 1], so (2/R^2) R' dR' = 4 t^3 dt.
//
// Inputs: xi_nfw(r, lnM), bias(lnM, z), xi_hh(r, z) as callables.  rho_m is
// the mean matter density (M_sun h^2 / Mpc^3); `unit` = 1e-12 returns
// M_sun h / pc^2, the haloModel Sigma/dSigma convention.
//
// The *_grid methods fill out[k * n_z + q] for every (lnM_k, z_q) node in one
// pass: xi_nfw and xi_hh are interpolated once per line-of-sight node and
// combined by the max, so the cost is n_los * (n_lnM + n_z) interpolations
// plus n_los * n_lnM * n_z max/multiply-adds per radius.
#ifndef Y3_CLUSTER_CPP_OPERATOR_MAX1H2H_GL_HH
#define Y3_CLUSTER_CPP_OPERATOR_MAX1H2H_GL_HH

#include "pipelines/shared/sel_gl_weights.hh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace y3_cluster {

class OperatorMax1h2h_gl {
public:
  using Fn2 = std::function<double(double, double)>;
  static constexpr std::size_t DEFAULT_N_LOS = 64;
  static constexpr std::size_t DEFAULT_N_APERTURE = 32;
  static constexpr double UNIT_MPC2_TO_PC2 = 1.0e-12;

  OperatorMax1h2h_gl(Fn2 xi_nfw, Fn2 bias, Fn2 xi_hh,
                     double rho_m, double r_max,
                     std::size_t n_los = DEFAULT_N_LOS,
                     std::size_t n_aperture = DEFAULT_N_APERTURE,
                     double unit = UNIT_MPC2_TO_PC2)
    : xi_nfw_(std::move(xi_nfw)), bias_(std::move(bias)),
      xi_hh_(std::move(xi_hh)), rho_m_(rho_m), r_max_(r_max), unit_(unit)
  {
    if (!(rho_m_ > 0.0) || !(r_max_ > 0.0) || n_los < 1 || n_aperture < 1)
      throw std::invalid_argument(
        "OperatorMax1h2h_gl: needs rho_m > 0, r_max > 0, n_los, n_aperture >= 1");
    y3_pipelines::gl_nodes(0.0, 1.0, n_los, t_los_, w_los_);
    y3_pipelines::gl_nodes(0.0, 1.0, n_aperture, t_ap_, w_ap_);
  }

  // Aperture nodes t_j and weights 4 w_j t_j^3: barSigma(<R) =
  // sum_j aperture_weights()[j] * Sigma(R * t_j^2).
  std::vector<double> const& aperture_nodes() const { return t_ap_; }
  std::vector<double> aperture_weights() const
  {
    std::vector<double> out(t_ap_.size());
    for (std::size_t j = 0; j != t_ap_.size(); ++j)
      out[j] = 4.0 * w_ap_[j] * t_ap_[j] * t_ap_[j] * t_ap_[j];
    return out;
  }

  // Sigma(R | lnM_k, z_q) -> out[k * n_z + q].
  void surface_density_grid(double R, std::vector<double> const& lnM,
                            std::vector<double> const& z,
                            std::vector<double>& out) const
  {
    std::size_t const n_k = lnM.size(), n_q = z.size();
    out.assign(n_k * n_q, 0.0);
    if (!(R > 0.0))
      throw std::invalid_argument("OperatorMax1h2h_gl requires R > 0");
    if (R >= r_max_)
      return;

    std::vector<double> b(n_k * n_q);
    for (std::size_t k = 0; k != n_k; ++k)
      for (std::size_t q = 0; q != n_q; ++q)
        b[k * n_q + q] = bias_(lnM[k], z[q]);

    // u in [0, u_max] mapped from the unit GL nodes; dl = r du.
    double const u_max = std::acosh(r_max_ / R);
    std::vector<double> one(n_k), two(n_q);
    for (std::size_t l = 0; l != t_los_.size(); ++l) {
      double const r = R * std::cosh(u_max * t_los_[l]);
      double const wr = u_max * w_los_[l] * r;
      for (std::size_t k = 0; k != n_k; ++k) one[k] = xi_nfw_(r, lnM[k]);
      for (std::size_t q = 0; q != n_q; ++q) two[q] = xi_hh_(r, z[q]);
      for (std::size_t k = 0; k != n_k; ++k) {
        double* row = &out[k * n_q];
        double const* brow = &b[k * n_q];
        for (std::size_t q = 0; q != n_q; ++q)
          row[q] += wr * std::max(one[k], brow[q] * two[q]);
      }
    }
    double const scale = 2.0 * rho_m_ * unit_;
    for (double& v : out) v *= scale;
  }

  // barSigma(<R | lnM_k, z_q) -> out[k * n_z + q].
  void mean_surface_density_grid(double R, std::vector<double> const& lnM,
                                 std::vector<double> const& z,
                                 std::vector<double>& out) const
  {
    out.assign(lnM.size() * z.size(), 0.0);
    std::vector<double> const wa = aperture_weights();
    std::vector<double> sigma;
    for (std::size_t j = 0; j != t_ap_.size(); ++j) {
      surface_density_grid(R * t_ap_[j] * t_ap_[j], lnM, z, sigma);
      for (std::size_t i = 0; i != out.size(); ++i)
        out[i] += wa[j] * sigma[i];
    }
  }

  // DeltaSigma(R | lnM_k, z_q) -> out[k * n_z + q].
  void excess_surface_density_grid(double R, std::vector<double> const& lnM,
                                   std::vector<double> const& z,
                                   std::vector<double>& out) const
  {
    std::vector<double> sigma;
    mean_surface_density_grid(R, lnM, z, out);
    surface_density_grid(R, lnM, z, sigma);
    for (std::size_t i = 0; i != out.size(); ++i)
      out[i] -= sigma[i];
  }

  double surface_density(double R, double lnM, double z) const
  {
    std::vector<double> out;
    surface_density_grid(R, {lnM}, {z}, out);
    return out[0];
  }

  double mean_surface_density(double R, double lnM, double z) const
  {
    std::vector<double> out;
    mean_surface_density_grid(R, {lnM}, {z}, out);
    return out[0];
  }

  double excess_surface_density(double R, double lnM, double z) const
  {
    std::vector<double> out;
    excess_surface_density_grid(R, {lnM}, {z}, out);
    return out[0];
  }

private:
  Fn2 xi_nfw_, bias_, xi_hh_;
  double rho_m_, r_max_, unit_;
  std::vector<double> t_los_, w_los_, t_ap_, w_ap_;
};

} // namespace y3_cluster

#endif
