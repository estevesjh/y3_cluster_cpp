// Lensing observables for the Shear1h2hMax profile.
//
// The profile constructs
//   xi_hm(r) = max[xi_1h(r), b(M,z) xi_nl(r,z)]
// with transition radius r_trans defined by:
//   xi_1h(r_trans) = b(M,z) xi_nl(r_trans, z).
//
// Line-of-sight projection yields the comoving lensing observables:
//   Sigma(R)      = 2 rho_m \int_0^\infty xi_hm(sqrt(R^2 + y^2)) dy
//   barSigma(<R)  = (2 / R^2) \int_0^R Sigma(R') R' dR'
//   DeltaSigma(R) = barSigma(<R) - Sigma(R)
//   gamma_T(R)    = DeltaSigma(R) Sigma_crit^{-1}.
//
// Evaluating the exact line-of-sight projection of the non-linear 3D maximum
// at every radial scale is computationally expensive during MCMC.
// The projected excess surface density DeltaSigma_max is approximated around transition radius
// and is assembled by building a piecewise function:
//
//   For R < r_trans:
//     DeltaSigma(R) = DeltaSigma_1h(R) + Delta_gap^1 * (R / r_trans)^2
//     Sigma(R)      = Sigma_1h(R) + Delta_gap^0 + 2 * Delta_gap^1 * (1 - (R / r_trans)^2)
//
//   For R >= r_trans:
//     DeltaSigma(R) = b * DeltaSigma_hh(R) + (C_0 / R^2) * 1e-12
//     Sigma(R)      = b * (Sigma_bar_hh(R) - DeltaSigma_hh(R))
//
//   Mean surface density:
//     barSigma(<R)  = Sigma(R) + DeltaSigma(R)
//
// where C_0 = Delta M_3D(r_trans) / pi (in Msun/h),
// Delta_gap^0 = Sigma_max(r_trans) - Sigma_1h(r_trans), and
// Delta_gap^1 = DeltaSigma_max(r_trans) - DeltaSigma_1h(r_trans).
//
// If r_trans is not in the datablock, the profile falls back to the pointwise
// max(1h, 2h) model, matching Python's MaxMixtureProfile.
#ifndef Y3_CLUSTER_CPP_SHEAR1H2H_MAX_PROFILE_HH
#define Y3_CLUSTER_CPP_SHEAR1H2H_MAX_PROFILE_HH

#include "cosmosis/datablock/datablock.hh"
#include "cosmosis/datablock/ndarray.hh"
#include "pipelines/shared/sel_gl_weights.hh"
#include "pipelines/systematics/selection_boost/Bsel.hh"
#include "utils/interp_1d.hh"
#include "utils/interp_2d.hh"
#include "utils/make_interp_1d.hh"
#include "utils/make_interp_2d.hh"

#include <algorithm>
#include <cmath>
#include <cstddef>
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
  Interp2D sigma_bar_hh_;
  Interp2D dsigma_hh_;
  std::vector<double> r_sigma_;
  std::optional<Interp1D> concentration_;
  std::size_t n_offset_;
  std::size_t n_phi_;
  std::size_t n_aperture_;
  double q_max_;
  double rho_ref_;
  std::optional<Interp2D> r_trans_;
  std::optional<Interp2D> c0_trans_;
  std::optional<Interp2D> delta_gap_0_;
  std::optional<Interp2D> delta_gap_1_;

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
    , sigma_bar_hh_(make_sanitized(
        sample, "haloModel", "r_sigma", "z", "Sigma_bar_hh"))
    , dsigma_hh_(make_sanitized(
        sample, "haloModel", "r_sigma", "z", "dSigma_hh"))
    , r_sigma_(sample.view<std::vector<double>>("haloModel", "r_sigma"))
    , n_offset_(n_offset)
    , n_phi_(n_phi)
    , n_aperture_(n_aperture)
    , q_max_(q_max)
    , rho_ref_(sample.view<double>("haloModel", "rho_m_ref"))
  {
    (void)n_gl;
    (void)n_root;
    if (n_offset_ < 1 || n_phi_ < 1 || n_aperture_ < 1 || q_max_ <= 0.0)
      throw std::invalid_argument(
        "Shear1h2hMaxProfile: invalid miscentering quadrature");
    if (sample.has_val("haloModel", "concentration"))
      concentration_.emplace(y3_cluster::make_Interp1D(
        sample, "haloModel", "lnM", "concentration"));
    if (sample.has_val("haloModel", "r_trans")) {
      r_trans_.emplace(y3_cluster::make_Interp2D(
        sample, "haloModel", "lnM", "z", "r_trans"));
      c0_trans_.emplace(y3_cluster::make_Interp2D(
        sample, "haloModel", "lnM", "z", "c0_trans"));
      delta_gap_0_.emplace(y3_cluster::make_Interp2D(
        sample, "haloModel", "lnM", "z", "delta_gap_0"));
      delta_gap_1_.emplace(y3_cluster::make_Interp2D(
        sample, "haloModel", "lnM", "z", "delta_gap_1"));
    }
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
    double const bias = bias_.clamp(lnM, z);
    double const sigma_2h = bias * (sigma_bar_hh_.clamp(radius, z) -
                                    dsigma_hh_.clamp(radius, z));
    if (r_trans_ && delta_gap_0_ && delta_gap_1_) {
      double const rt = r_trans_->clamp(lnM, z);
      if (radius >= rt) {
        return sigma_2h;
      } else {
        double const s1h = nfw_surface_density(radius, lnM);
        double const gap0 = delta_gap_0_->clamp(lnM, z);
        double const gap1 = delta_gap_1_->clamp(lnM, z);
        double const r_ratio = radius / rt;
        return s1h + gap0 + 2.0 * gap1 * (1.0 - r_ratio * r_ratio);
      }
    }
    double const s1h = nfw_surface_density(radius, lnM);
    return std::max(s1h, sigma_2h);
  }

  double
  excess_surface_density(double R, double lnM, double z) const
  {
    double const radius = std::max(R, r_sigma_.front());
    double const bias = bias_.clamp(lnM, z);
    double const dsigma_2h = bias * dsigma_hh_.clamp(radius, z);
    if (r_trans_ && c0_trans_ && delta_gap_1_) {
      double const rt = r_trans_->clamp(lnM, z);
      if (radius >= rt) {
        double const c0 = c0_trans_->clamp(lnM, z);
        return dsigma_2h + (c0 / (radius * radius)) * MSUN_PER_MPC2_TO_MSUN_PER_PC2;
      } else {
        double const dsigma_1h = nfw_excess_surface_density(radius, lnM);
        double const gap1 = delta_gap_1_->clamp(lnM, z);
        double const r_ratio = radius / rt;
        return dsigma_1h + gap1 * r_ratio * r_ratio;
      }
    }
    double const dsigma_1h = nfw_excess_surface_density(radius, lnM);
    return std::max(dsigma_1h, dsigma_2h);
  }

  double
  mean_surface_density(double R, double lnM, double z) const
  {
    double const radius = std::max(R, r_sigma_.front());
    return surface_density(radius, lnM, z) + excess_surface_density(radius, lnM, z);
  }

  double
  tangential_shear(double R, double lnM, double z,
                   double sigma_crit_inv) const
  {
    return excess_surface_density(R, lnM, z) * sigma_crit_inv;
  }

  double
  selected_excess_surface_density(
    double R, double lnM, double z,
    y3_cluster::BselDeltaSigmaOperator const& bsel_op,
    int bin_index, double lob) const
  {
    return bsel_op.apply(
      excess_surface_density(R, lnM, z),
      [this, lnM, z](double r) { return mean_surface_density(r, lnM, z); },
      R, bin_index, lob, z);
  }

  double
  miscentered_selected_excess_surface_density(
    double R, double r_mis, double lnM, double z,
    y3_cluster::BselDeltaSigmaOperator const& bsel_op,
    int bin_index, double lob) const
  {
    return bsel_op.apply(
      miscentered_excess_surface_density(R, r_mis, lnM, z),
      [this, lnM, z](double r) { return mean_surface_density(r, lnM, z); },
      R, bin_index, lob, z);
  }
};

#endif
