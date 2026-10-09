// Lensing observables for the Shear1h2hMax profile.
//
// The profile constructs
//   xi_hm(r) = max[xi_1h(r), b(M,z) xi_nl(r,z)]
// and projects it into the following *comoving* surface-density quantities,
// in the same mass/area convention as haloModel/{Sigma_hh,dSigma_hh}:
//
//   Sigma(R)       = projected surface density,
//   barSigma(<R)   = 2/R^2 integral_0^R Sigma(R') R' dR',
//   DeltaSigma(R)  = barSigma(<R) - Sigma(R),
//   gamma_T(R)     = DeltaSigma(R) Sigma_crit^{-1}.
//
// `surface_density` returns Sigma, `mean_surface_density` returns barSigma,
// `excess_surface_density` returns DeltaSigma, and `tangential_shear` returns
// gamma_T only when the caller supplies Sigma_crit^{-1}.  The profile never
// hides lens/source geometry or silently converts DeltaSigma into shear.
//
// For R >= r_trans the line of sight is entirely in the two-halo regime and
// the exact DeltaSigma result is DeltaSigma_2h + C0/R^2.  For R < r_trans,
// nonsingular fixed-order GL integrals after y = R sinh(u) provide the Sigma
// and barSigma corrections.
#ifndef Y3_CLUSTER_CPP_SHEAR1H2H_MAX_PROFILE_HH
#define Y3_CLUSTER_CPP_SHEAR1H2H_MAX_PROFILE_HH

#include "cosmosis/datablock/datablock.hh"
#include "cosmosis/datablock/ndarray.hh"
#include "pipelines/shared/sel_gl_weights.hh"
#include "utils/interp_1d.hh"
#include "utils/interp_2d.hh"
#include "utils/make_interp_1d.hh"
#include "utils/make_interp_2d.hh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

class Shear1h2hMaxProfile {
  using Interp2D = y3_cluster::Interp2D;
  using Interp1D = y3_cluster::Interp1D;

  static constexpr double PI = 3.1415926535897932384626433832795;
  static constexpr double MSUN_PER_MPC2_TO_MSUN_PER_PC2 = 1.0e-12;
  static constexpr std::size_t DEFAULT_N_ROOT = 96;
  static constexpr std::size_t DEFAULT_N_GL = 20;
  static constexpr std::size_t DEFAULT_N_OFFSET = 24;
  static constexpr std::size_t DEFAULT_N_PHI = 64;
  static constexpr std::size_t DEFAULT_N_APERTURE = 8;
  static constexpr double DEFAULT_Q_MAX = 12.0;

  Interp2D bias_;
  Interp2D xi_nl_;
  Interp2D m3d_nl_;
  Interp2D sigma_bar_hh_;
  Interp2D dsigma_hh_;
  std::vector<double> r_xi_;
  std::vector<double> r_sigma_;
  std::optional<Interp1D> concentration_;
  std::size_t n_root_;
  std::size_t n_gl_;
  std::size_t n_offset_;
  std::size_t n_phi_;
  std::size_t n_aperture_;
  double q_max_;
  double rho_ref_;
  mutable std::map<std::pair<double, double>, double> transition_cache_;

  struct ProjectionCorrections {
    double sigma;
    double mean_sigma;
    double delta_sigma;
  };

  static Interp2D
  make_sanitized(cosmosis::DataBlock& sample,
                 char const* section, char const* x_key,
                 char const* y_key, char const* value_key)
  {
    using doubles = std::vector<double>;
    auto const x = sample.view<doubles>(section, x_key);
    auto const y = sample.view<doubles>(section, y_key);
    auto const& nd = sample.view<cosmosis::ndarray<double>>(
      section, value_key);
    std::vector<double> values(nd.begin(), nd.end());
    if (values.size() != x.size() * y.size())
      throw std::runtime_error(
        "Shear1h2hMaxProfile: interpolation table extents mismatch");
    for (double& value : values)
      if (!std::isfinite(value)) value = 0.0;
    return Interp2D(x, y, values);
  }

  double
  concentration(double lnM) const
  {
    return concentration_ ? concentration_->clamp(lnM) : 4.0;
  }

  void
  nfw_parameters(double lnM, double& rs, double& rho_s,
                 double& r200) const
  {
    double const c = concentration(lnM);
    r200 = std::cbrt(3.0 * std::exp(lnM) /
                     (800.0 * PI * rho_ref_));
    rs = r200 / c;
    double const delta_c = (200.0 * c * c * c / 3.0) /
                           (std::log1p(c) - c / (1.0 + c));
    rho_s = delta_c * rho_ref_;
  }

  struct NfwShapes {
    double f;
    double g;
  };

  static NfwShapes
  nfw_shapes(double x)
  {
    x = std::max(x, 1.0e-12);
    if (std::abs(x - 1.0) <= 1.0e-8)
      return {1.0 / 3.0, 1.0 + std::log(0.5)};
    if (x < 1.0) {
      double const root = std::sqrt((1.0 - x) / (1.0 + x));
      double const a = std::atanh(root) / std::sqrt(1.0 - x * x);
      return {(1.0 - 2.0 * a) / (x * x - 1.0),
              std::log(x / 2.0) + 2.0 * a};
    }
    double const root = std::sqrt((x - 1.0) / (1.0 + x));
    double const a = std::atan(root) / std::sqrt(x * x - 1.0);
    return {(1.0 - 2.0 * a) / (x * x - 1.0),
            std::log(x / 2.0) + 2.0 * a};
  }

  double
  xi_1h(double r, double lnM) const
  {
    double rs, rho_s, r200;
    nfw_parameters(lnM, rs, rho_s, r200);
    double const x = std::max(r / rs, 1.0e-15);
    return rho_s / (rho_ref_ * x * (1.0 + x) * (1.0 + x)) - 1.0;
  }

  double
  m3d_1h(double r, double lnM) const
  {
    double rs, rho_s, r200;
    nfw_parameters(lnM, rs, rho_s, r200);
    double const x = r / rs;
    return 4.0 * PI * rho_s * rs * rs * rs *
             (std::log1p(x) - x / (1.0 + x)) -
           (4.0 / 3.0) * PI * rho_ref_ * r * r * r;
  }

  double
  xi_2h(double r, double lnM, double z) const
  {
    return bias_.clamp(lnM, z) * xi_nl_.clamp(r, z);
  }

  double
  m3d_2h(double r, double lnM, double z) const
  {
    return bias_.clamp(lnM, z) * m3d_nl_.clamp(r, z);
  }

  double
  transition_radius(double lnM, double z) const
  {
    auto const key = std::make_pair(lnM, z);
    auto const found = transition_cache_.find(key);
    if (found != transition_cache_.end()) return found->second;

    double rs, rho_s, r200;
    nfw_parameters(lnM, rs, rho_s, r200);
    double const lo = std::max(r_xi_.front(), 0.25 * r200);
    double const hi = r_xi_.back();
    double prev_r = lo;
    double prev_f = xi_1h(prev_r, lnM) - xi_2h(prev_r, lnM, z);
    double a = lo, b = hi;
    bool bracketed = false;
    for (std::size_t i = 1; i != n_root_; ++i) {
      double const t = static_cast<double>(i) /
                       static_cast<double>(n_root_ - 1);
      double const r = lo * std::pow(hi / lo, t);
      double const f = xi_1h(r, lnM) - xi_2h(r, lnM, z);
      if (std::isfinite(prev_f) && std::isfinite(f) &&
          prev_f >= 0.0 && f <= 0.0) {
        a = prev_r;
        b = r;
        bracketed = true;
        break;
      }
      prev_r = r;
      prev_f = f;
    }
    if (!bracketed) {
      double const fallback = std::clamp(2.0 * r200,
                                         r_xi_.front(), r_xi_.back());
      transition_cache_.emplace(key, fallback);
      return fallback;
    }

    double fa = xi_1h(a, lnM) - xi_2h(a, lnM, z);
    for (int i = 0; i != 60; ++i) {
      double const mid = std::sqrt(a * b);
      double const fm = xi_1h(mid, lnM) - xi_2h(mid, lnM, z);
      if (fa * fm > 0.0) {
        a = mid;
        fa = fm;
      } else {
        b = mid;
      }
    }
    double const result = std::sqrt(a * b);
    transition_cache_.emplace(key, result);
    return result;
  }

  ProjectionCorrections
  projection_corrections(double radius, double lnM, double z,
                         double r_trans) const
  {
    double const delta_m_trans =
      m3d_1h(r_trans, lnM) - m3d_2h(r_trans, lnM, z);
    double const outer_tail = delta_m_trans / (PI * radius * radius) *
                              MSUN_PER_MPC2_TO_MSUN_PER_PC2;
    if (radius >= r_trans)
      return {0.0, outer_tail, outer_tail};

    std::vector<double> nodes, weights;
    double const u_max = std::acosh(r_trans / radius);
    y3_pipelines::gl_nodes(0.0, u_max, n_gl_, nodes, weights);
    double sigma_diff = 0.0;
    double mantle = 0.0;
    for (std::size_t i = 0; i != nodes.size(); ++i) {
      double const u = nodes[i];
      double const y = radius * std::sinh(u);
      double const r = radius * std::cosh(u);
      double const delta_xi = xi_1h(r, lnM) - xi_2h(r, lnM, z);
      double const jacobian = r;
      sigma_diff += weights[i] * jacobian * delta_xi;
      mantle += weights[i] * jacobian * delta_xi * y / (r + y);
    }
    sigma_diff *= 2.0 * rho_ref_;
    mantle *= 4.0 * rho_ref_;
    double const delta_m = m3d_1h(radius, lnM) -
                           m3d_2h(radius, lnM, z);
    double const unit_conversion = MSUN_PER_MPC2_TO_MSUN_PER_PC2;
    return {
      sigma_diff * unit_conversion,
      (delta_m / (PI * radius * radius) + mantle) * unit_conversion,
      (delta_m / (PI * radius * radius) + mantle - sigma_diff) *
        unit_conversion};
  }

  double
  convolved_surface_density(double radius, double r_mis,
                             double lnM, double z,
                             bool residual) const
  {
    radius = std::max(radius, 1.0e-8);
    if (r_mis <= 0.0)
      return residual ? residual_surface_density(radius, lnM, z)
                      : surface_density(radius, lnM, z);

    std::vector<double> edges{0.0, 1.0, 4.0, 8.0, 16.0, q_max_};
    double const q_break = radius / r_mis;
    if (q_break > 0.0 && q_break < q_max_)
      edges.push_back(q_break);
    for (double& edge : edges)
      edge = std::clamp(edge, 0.0, q_max_);
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

    std::vector<double> phi_t, phi_w;
    y3_pipelines::gl_nodes(0.0, 1.0, n_phi_, phi_t, phi_w);
    double total = 0.0;
    for (std::size_t panel = 0; panel + 1 < edges.size(); ++panel) {
      double const left = edges[panel];
      double const right = edges[panel + 1];
      if (right <= left) continue;
      std::vector<double> q, wq;
      y3_pipelines::gl_nodes(left, right, n_offset_, q, wq);
      for (std::size_t iq = 0; iq != q.size(); ++iq) {
        double const offset = r_mis * q[iq];
        double angular = 0.0;
        for (std::size_t ip = 0; ip != phi_t.size(); ++ip) {
          double const phi = PI * phi_t[ip] * phi_t[ip];
          double const separation = std::sqrt(
            (radius - offset) * (radius - offset) +
            4.0 * radius * offset *
              std::sin(0.5 * phi) * std::sin(0.5 * phi));
          double const value = residual
            ? residual_surface_density(separation, lnM, z)
            : surface_density(separation, lnM, z);
          angular += 2.0 * phi_t[ip] * phi_w[ip] * value;
        }
        total += wq[iq] * q[iq] * std::exp(-q[iq]) * angular;
      }
    }
    return total;
  }

  double
  convolved_residual_surface_density(double radius, double r_mis,
                                      double lnM, double z) const
  {
    return convolved_surface_density(radius, r_mis, lnM, z, true);
  }

  double
  miscentered_excess_surface_density_impl(
    double R, double r_mis, double lnM, double z, bool residual) const
  {
    std::vector<double> t, wt;
    y3_pipelines::gl_nodes(0.0, 1.0, n_aperture_, t, wt);
    double mean = 0.0;
    for (std::size_t i = 0; i != t.size(); ++i) {
      double const radius = std::max(R * t[i] * t[i], 1.0e-8);
      mean += 4.0 * wt[i] * t[i] * t[i] * t[i] *
              convolved_surface_density(radius, r_mis, lnM, z, residual);
    }
    double const edge = convolved_surface_density(
      std::max(R, 1.0e-8), r_mis, lnM, z, residual);
    return mean - edge;
  }

public:
  explicit Shear1h2hMaxProfile(
    cosmosis::DataBlock& sample,
    std::size_t n_gl = DEFAULT_N_GL,
    std::size_t n_root = DEFAULT_N_ROOT,
    std::size_t n_offset = DEFAULT_N_OFFSET,
    std::size_t n_phi = DEFAULT_N_PHI,
    std::size_t n_aperture = DEFAULT_N_APERTURE,
    double q_max = DEFAULT_Q_MAX)
    : bias_(y3_cluster::make_Interp2D(
        sample, "haloModel", "lnM", "z", "bias"))
    , xi_nl_(y3_cluster::make_Interp2D(
        sample, "xi_nl", "r", "z", "xi_nl"))
    , m3d_nl_(y3_cluster::make_Interp2D(
        sample, "xi_nl", "r", "z", "m3d_nl"))
    , sigma_bar_hh_(make_sanitized(
        sample, "haloModel", "r_sigma", "z", "Sigma_bar_hh"))
    , dsigma_hh_(make_sanitized(
        sample, "haloModel", "r_sigma", "z", "dSigma_hh"))
    , r_xi_(sample.view<std::vector<double>>("xi_nl", "r"))
    , r_sigma_(sample.view<std::vector<double>>("haloModel", "r_sigma"))
    , n_root_(n_root)
    , n_gl_(n_gl)
    , n_offset_(n_offset)
    , n_phi_(n_phi)
    , n_aperture_(n_aperture)
    , q_max_(q_max)
    , rho_ref_(sample.view<double>("haloModel", "rho_m_ref"))
  {
    if (n_gl_ < 1)
      throw std::invalid_argument(
        "Shear1h2hMaxProfile: n_gl must be positive");
    if (n_root_ < 2)
      throw std::invalid_argument(
        "Shear1h2hMaxProfile: n_root must be at least 2");
    if (n_offset_ < 1 || n_phi_ < 1 || n_aperture_ < 1 || q_max_ <= 0.0)
      throw std::invalid_argument(
        "Shear1h2hMaxProfile: invalid miscentering quadrature");
    if (sample.has_val("haloModel", "concentration"))
      concentration_.emplace(y3_cluster::make_Interp1D(
        sample, "haloModel", "lnM", "concentration"));
  }

  double
  nfw_surface_density(double R, double lnM) const
  {
    double rs, rho_s, r200;
    nfw_parameters(lnM, rs, rho_s, r200);
    NfwShapes const shape = nfw_shapes(R / rs);
    return 2.0 * rho_s * rs * shape.f *
           MSUN_PER_MPC2_TO_MSUN_PER_PC2;
  }

  double
  nfw_excess_surface_density(double R, double lnM) const
  {
    double rs, rho_s, r200;
    nfw_parameters(lnM, rs, rho_s, r200);
    double const x = std::max(R / rs, 1.0e-12);
    NfwShapes const shape = nfw_shapes(x);
    return 2.0 * rho_s * rs *
           (2.0 * shape.g / (x * x) - shape.f) *
           MSUN_PER_MPC2_TO_MSUN_PER_PC2;
  }

  double
  residual_surface_density(double R, double lnM, double z) const
  {
    return surface_density(R, lnM, z) - nfw_surface_density(R, lnM);
  }

  double
  residual_excess_surface_density(double R, double lnM, double z) const
  {
    return excess_surface_density(R, lnM, z) -
           nfw_excess_surface_density(R, lnM);
  }

  double
  miscentered_residual_excess_surface_density(
    double R, double r_mis, double lnM, double z) const
  {
    return miscentered_excess_surface_density_impl(
      R, r_mis, lnM, z, true);
  }

  double
  miscentered_excess_surface_density(
    double R, double r_mis, double lnM, double z) const
  {
    return miscentered_excess_surface_density_impl(
      R, r_mis, lnM, z, false);
  }

  double
  surface_density(double R, double lnM, double z) const
  {
    double const radius = std::max(R, r_sigma_.front());
    double const r_trans = transition_radius(lnM, z);
    double const bias = bias_.clamp(lnM, z);
    double const sigma_2h = bias * (sigma_bar_hh_.clamp(radius, z) -
                                    dsigma_hh_.clamp(radius, z));
    return sigma_2h + projection_corrections(
      radius, lnM, z, r_trans).sigma;
  }

  double
  mean_surface_density(double R, double lnM, double z) const
  {
    double const radius = std::max(R, r_sigma_.front());
    double const r_trans = transition_radius(lnM, z);
    double const bias = bias_.clamp(lnM, z);
    double const mean_sigma_2h = bias * sigma_bar_hh_.clamp(radius, z);
    return mean_sigma_2h + projection_corrections(
      radius, lnM, z, r_trans).mean_sigma;
  }

  double
  excess_surface_density(double R, double lnM, double z) const
  {
    double const radius = std::max(R, r_sigma_.front());
    double const r_trans = transition_radius(lnM, z);
    double const bias = bias_.clamp(lnM, z);
    double const dsigma_2h = bias * dsigma_hh_.clamp(radius, z);
    return dsigma_2h + projection_corrections(
      radius, lnM, z, r_trans).delta_sigma;
  }

  double
  tangential_shear(double R, double lnM, double z,
                   double sigma_crit_inv) const
  {
    return excess_surface_density(R, lnM, z) * sigma_crit_inv;
  }
};

#endif
