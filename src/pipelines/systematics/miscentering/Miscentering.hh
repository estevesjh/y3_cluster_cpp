// Gamma-kernel miscentering of an excess surface density, fixed Gauss-Legendre.
//
// For a halo whose centred surface density is Sigma(r) (the full max-model
// profile, Bsel-weighted when selection is on), a fraction f_mis of clusters
// is centred at a projected offset R_s drawn from the gamma distribution
//
//   P(R_s | sigma) = R_s / sigma^2 exp(-R_s / sigma),
//   sigma_b = tau_mis R_lambda(lambda_b),   R_lambda = (lambda / 100)^0.2.
//
// (E1) Sigma_mis(R)  = int dR_s P(R_s) (1/pi) int_0^pi dtheta
//                        Sigma(sqrt(R^2 + R_s^2 + 2 R R_s cos theta))
// (E2) DSigma_mis(R) = int_0^1 4 t^3 Sigma_mis(R t^2) dt - Sigma_mis(R)
// (E4) O_obs         = (1 - f_mis) O_cen + f_mis O_mis
//
// Discretisation (all nodes fixed GL on [0, 1]):
//   offsets: two panels split at R_s = R (the ring passes through the halo
//            centre there): R_s = R u, and R_s = R + x_max sigma u;
//   angle:   theta = pi (1 - v^2), clustering nodes at theta = pi;
//   aperture: R' = R t^2 with weights 4 w t^3 (as OperatorMax1h2h_gl).
// On a log-uniform radial grid r_m with Sigma linear in ln r between nodes,
// (E1)+(E2) collapse to one weight row per (sigma, R):
//   DSigma_mis(R) = sum_m L_m Sigma(r_m),
// built by excess_surface_density_row().  Ring radii reach 2 R + x_max sigma,
// which the grid must cover.
//
// Below the grid floor r_lo the profile must keep its core mass: clamping
// Sigma below 0.1 Mpc/h biases DSigma_mis by 23% at R = 0.2 and 1% at R = 5
// for an NFW (2026-10-13 note, section 7).  Either start the grid inside the
// cusp (~1e-4), or give it a core slot: ring nodes with r < r_lo then
// collect into one extra row entry, the uniform-disk core
// Sigma_core = M_2d(<r_lo) / (pi r_lo^2) (haloModel/m2d_core).  With
// r_lo = 1e-3 the disk is as accurate as the exact NFW interior (1.4e-5).
// Validated on analytic NFW against a Hankel-space truth, which needs no
// Sigma -> DSigma aperture integral, and against cluster-toolkit;
// MiscenteringCuhre is the adaptive full-integral reference.
#ifndef Y3_CLUSTER_CPP_SYSTEMATICS_MISCENTERING_HH
#define Y3_CLUSTER_CPP_SYSTEMATICS_MISCENTERING_HH

#include "cosmosis/datablock/datablock.hh"
#include "cubacpp/cuhre.hh"
#include "cubacpp/integration_volume.hh"
#include "cubacpp/turn_off_cuba_forking.hh"
#include "gsl/gsl_spline.h"
#include "pipelines/shared/lensing_helpers.hh"
#include "pipelines/shared/sel_gl_weights.hh"
#include "pipelines/systematics/selection_boost/Bsel.hh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <optional>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace y3_cluster {

// Analytic NFW (Wright & Brainerd 2000) — the validation profile of the
// tests and the Cuhre reference.  M = M200m with
// r_200 = [3 M / (800 pi rho_m)]^(1/3) and rho_s = delta_c rho_m — the
// haloModel / OperatorMax1h2h_gl convention.  Output in M_sun h / pc^2.
class NfwWrightBrainerd {
public:
  NfwWrightBrainerd(double lnM, double c, double rho_m)
  {
    constexpr double PI = 3.14159265358979323846;
    double const r200 = std::cbrt(3.0 * std::exp(lnM) / (800.0 * PI * rho_m));
    double const delta_c =
      (200.0 / 3.0) * c * c * c / (std::log1p(c) - c / (1.0 + c));
    r_s_ = r200 / c;
    sigma_s_ = 2.0 * delta_c * rho_m * r_s_ * 1.0e-12;
  }

  double r_s() const { return r_s_; }

  double surface_density(double r) const { return sigma_s_ * F(r / r_s_); }

  // barSigma(<r) - Sigma(r), with barSigma = 2 sigma_s g(x) / x^2.
  double excess_surface_density(double r) const
  {
    double const x = r / r_s_;
    return sigma_s_ * (2.0 * g(x) / (x * x) - F(x));
  }

private:
  // 2/sqrt(|1 - x^2|) {atanh, atan}(sqrt(|1 - x| / (1 + x))); -> 1 at x = 1.
  static double h(double x)
  {
    if (x < 1.0)
      return 2.0 / std::sqrt(1.0 - x * x) *
             std::atanh(std::sqrt((1.0 - x) / (1.0 + x)));
    if (x > 1.0)
      return 2.0 / std::sqrt(x * x - 1.0) *
             std::atan(std::sqrt((x - 1.0) / (x + 1.0)));
    return 1.0;
  }

  // Sigma / sigma_s = (1 - h) / (x^2 - 1); series near x = 1 (0/0 there).
  static double F(double x)
  {
    double const d = x - 1.0;
    if (std::abs(d) < 1.0e-3)
      return 1.0 / 3.0 - 0.4 * d + (13.0 / 35.0) * d * d -
             (20.0 / 63.0) * d * d * d;
    return (1.0 - h(x)) / (x * x - 1.0);
  }

  // barSigma x^2 / (2 sigma_s) = ln(x / 2) + h(x).
  static double g(double x) { return std::log(0.5 * x) + h(x); }

  double r_s_{0.0}, sigma_s_{0.0};
};

// Log-uniform radial grid r_m = exp(ln r_lo + m dln), m = 0..n-1, with
// hat functions linear in ln r.  Queries above r_hi clamp.  Below r_lo they
// clamp to r_0, or — with core = true — go to the extra core slot n (one
// constant value for the whole disk r < r_lo); rows then have n + 1 entries.
class LogRadialGrid {
public:
  LogRadialGrid(double r_lo, double r_hi, std::size_t n, bool core = false)
    : ln_lo_(std::log(r_lo)), core_(core), r_(n)
  {
    if (!(r_lo > 0.0) || !(r_hi > r_lo) || n < 2)
      throw std::invalid_argument(
        "LogRadialGrid: needs 0 < r_lo < r_hi and n >= 2");
    dln_ = (std::log(r_hi) - ln_lo_) / static_cast<double>(n - 1);
    for (std::size_t m = 0; m != n; ++m)
      r_[m] = std::exp(ln_lo_ + dln_ * static_cast<double>(m));
  }

  std::size_t size() const { return r_.size(); }
  bool has_core() const { return core_; }
  std::size_t row_size() const { return r_.size() + (core_ ? 1 : 0); }
  double r(std::size_t m) const { return r_[m]; }
  double r_lo() const { return r_.front(); }
  double r_hi() const { return r_.back(); }
  double dln() const { return dln_; }

  // row += w h_m(r) for the two hat functions that are non-zero at ln r.
  void scatter(double ln_r, double w, double* row) const
  {
    double const u = (ln_r - ln_lo_) / dln_;
    std::size_t const last = r_.size() - 1;
    if (core_ && u < 0.0) { row[last + 1] += w; return; }
    if (!(u > 0.0)) { row[0] += w; return; }
    if (u >= static_cast<double>(last)) { row[last] += w; return; }
    auto const i = static_cast<std::size_t>(u);
    double const f = u - static_cast<double>(i);
    row[i] += w * (1.0 - f);
    row[i + 1] += w * f;
  }

private:
  double ln_lo_, dln_{0.0};
  bool core_;
  std::vector<double> r_;
};

// Natural cubic spline of y(ln r) (GSL cspline, C2 — no kinks at the knots,
// which stall adaptive cubature on linearly interpolated tables), clamped
// to the end values outside [r_0, r_n-1].  Copyable; evaluation is const.
class CubicSplineLnR {
public:
  CubicSplineLnR(std::vector<double> const& r, std::vector<double> const& y)
    : lo_(r.front()), hi_(r.back()),
      spline_(gsl_spline_alloc(gsl_interp_cspline, r.size()), &gsl_spline_free)
  {
    if (r.size() != y.size() || r.size() < 3 || !(r.front() > 0.0))
      throw std::invalid_argument(
        "CubicSplineLnR: needs >= 3 matching (r, y) points, r > 0");
    std::vector<double> ln_r(r.size());
    for (std::size_t i = 0; i != r.size(); ++i) ln_r[i] = std::log(r[i]);
    gsl_spline_init(spline_.get(), ln_r.data(), y.data(), r.size());
    y_lo_ = y.front();
    y_hi_ = y.back();
  }

  double r_lo() const { return lo_; }
  double r_hi() const { return hi_; }

  double operator()(double r) const
  {
    if (!(r > lo_)) return y_lo_;
    if (!(r < hi_)) return y_hi_;
    return gsl_spline_eval(spline_.get(), std::log(r), nullptr);
  }

private:
  double lo_, hi_, y_lo_{0.0}, y_hi_{0.0};
  std::shared_ptr<gsl_spline> spline_;
};

class MiscenteringOperator {
public:
  static constexpr std::size_t DEFAULT_N_OFFSET = 32;
  static constexpr std::size_t DEFAULT_N_THETA = 32;
  static constexpr std::size_t DEFAULT_N_APERTURE = 32;
  // Gamma tail beyond x_max sigma: probability (x_max + 1) exp(-x_max) = 4e-8.
  static constexpr double DEFAULT_X_MAX = 20.0;

  MiscenteringOperator(double f_mis, double tau_mis,
                       std::size_t n_offset = DEFAULT_N_OFFSET,
                       std::size_t n_theta = DEFAULT_N_THETA,
                       std::size_t n_aperture = DEFAULT_N_APERTURE,
                       double x_max = DEFAULT_X_MAX)
    : f_mis_(f_mis), tau_mis_(tau_mis), x_max_(x_max)
  {
    if (!(f_mis_ >= 0.0 && f_mis_ <= 1.0))
      throw std::invalid_argument("MiscenteringOperator: f_mis must be in [0, 1]");
    if (!(tau_mis_ > 0.0) || !std::isfinite(tau_mis_))
      throw std::invalid_argument("MiscenteringOperator: tau_mis must be > 0");
    if (n_offset < 1 || n_theta < 1 || n_aperture < 1 || !(x_max_ > 0.0))
      throw std::invalid_argument(
        "MiscenteringOperator: needs n_offset, n_theta, n_aperture >= 1, x_max > 0");
    constexpr double PI = 3.14159265358979323846;
    y3_pipelines::gl_nodes(0.0, 1.0, n_offset, u_, wu_);
    std::vector<double> v, wv, t, wt;
    y3_pipelines::gl_nodes(0.0, 1.0, n_theta, v, wv);
    y3_pipelines::gl_nodes(0.0, 1.0, n_aperture, t, wt);
    cos_theta_.resize(n_theta);
    w_theta_.resize(n_theta);
    for (std::size_t l = 0; l != n_theta; ++l) {
      cos_theta_[l] = std::cos(PI * (1.0 - v[l] * v[l]));
      w_theta_[l] = 2.0 * v[l] * wv[l];  // (1/pi) dtheta = 2 v dv
    }
    t2_.resize(n_aperture);
    w_ap_.resize(n_aperture);
    for (std::size_t j = 0; j != n_aperture; ++j) {
      t2_[j] = t[j] * t[j];
      w_ap_[j] = 4.0 * wt[j] * t[j] * t[j] * t[j];
    }
  }

  // Required: miscentering/f_mis and miscentering/tau_mis (view<double>
  // throws when absent — no fiducial fallback).
  static MiscenteringOperator from_datablock(
    cosmosis::DataBlock& s, std::string const& section = "miscentering",
    std::size_t n_offset = DEFAULT_N_OFFSET,
    std::size_t n_theta = DEFAULT_N_THETA,
    std::size_t n_aperture = DEFAULT_N_APERTURE)
  {
    return MiscenteringOperator(s.view<double>(section, "f_mis"),
                                s.view<double>(section, "tau_mis"),
                                n_offset, n_theta, n_aperture);
  }

  double f_mis() const { return f_mis_; }
  double tau_mis() const { return tau_mis_; }
  double x_max() const { return x_max_; }

  // sigma = tau_mis R_lambda(lambda^ob)  [h^-1 Mpc]
  double offset_scale(double lob) const
  {
    return tau_mis_ * y3_pipelines::R_lambda(lob);
  }

  // Largest ring radius (E1)+(E2) query for R: aperture R' <= R, offset
  // R_s <= R' + x_max sigma.
  double max_ring_radius(double R, double sigma) const
  {
    return 2.0 * R + x_max_ * sigma;
  }

  // (E1) by (D1), Sigma given as a callback r -> Sigma(r).
  template <typename SigmaFn>
  double surface_density(double R, double sigma, SigmaFn&& sigma_fn) const
  {
    double acc = 0.0;
    for_each_ring_node(R, sigma, [&](double r2, double w) {
      acc += w * sigma_fn(std::sqrt(r2));
    });
    return acc;
  }

  // (E1)+(E2) by (D1)+(D2), Sigma given as a callback.
  template <typename SigmaFn>
  double excess_surface_density(double R, double sigma, SigmaFn&& sigma_fn) const
  {
    double acc = -surface_density(R, sigma, sigma_fn);
    for (std::size_t j = 0; j != t2_.size(); ++j)
      acc += w_ap_[j] * surface_density(R * t2_[j], sigma, sigma_fn);
    return acc;
  }

  // (D4): row with DSigma_mis(R) = sum_m row[m] Sigma(r_m) (+ row[n]
  // Sigma_core with a core slot), Sigma linear in ln r on `grid`.  Same
  // nodes as excess_surface_density().
  void excess_surface_density_row(double R, double sigma,
                                  LogRadialGrid const& grid,
                                  std::vector<double>& row) const
  {
    if (max_ring_radius(R, sigma) > grid.r_hi() * (1.0 + 1.0e-12))
      throw std::out_of_range(
        "MiscenteringOperator: radial grid ends below 2 R + x_max sigma");
    row.assign(grid.row_size(), 0.0);
    auto add = [&](double Rp, double c) {
      for_each_ring_node(Rp, sigma, [&](double r2, double w) {
        grid.scatter(0.5 * std::log(r2), c * w, row.data());
      });
    };
    add(R, -1.0);
    for (std::size_t j = 0; j != t2_.size(); ++j)
      add(R * t2_[j], w_ap_[j]);
  }

  // (E4)
  double apply(double o_cen, double o_mis) const
  {
    return (1.0 - f_mis_) * o_cen + f_mis_ * o_mis;
  }

private:
  // (D1) nodes for Sigma_mis(Rp): visit(r^2, weight) with
  // Sigma_mis(Rp) = sum weight Sigma(r).
  template <typename Visit>
  void for_each_ring_node(double Rp, double sigma, Visit&& visit) const
  {
    if (!(Rp > 0.0) || !(sigma > 0.0))
      throw std::invalid_argument("MiscenteringOperator requires R > 0, sigma > 0");
    double const tail = x_max_ * sigma;
    auto ring = [&](double rs, double w) {
      double const a = Rp * Rp + rs * rs, b = 2.0 * Rp * rs;
      for (std::size_t l = 0; l != cos_theta_.size(); ++l)
        visit(a + b * cos_theta_[l], w * w_theta_[l]);
    };
    for (std::size_t i = 0; i != u_.size(); ++i) {
      double const rs1 = Rp * u_[i];
      ring(rs1, Rp * wu_[i] * gamma_pdf(rs1, sigma));
      double const rs2 = Rp + tail * u_[i];
      ring(rs2, tail * wu_[i] * gamma_pdf(rs2, sigma));
    }
  }

  static double gamma_pdf(double rs, double sigma)
  {
    return rs / (sigma * sigma) * std::exp(-rs / sigma);
  }

  double f_mis_, tau_mis_, x_max_;
  std::vector<double> u_, wu_, cos_theta_, w_theta_, t2_, w_ap_;
};

// Reference: the full miscentering integral, adaptive Cuhre, no fixed nodes,
// no tabulated Sigma_mis.  Same input contract as cluster-toolkit
// (C_miscentering.c): Sigma is a natural cubic spline in ln r of
// (r_tab, sigma_tab), the analytic core below r_tab[0], and 0 above
// r_tab.back().  Unlike cluster-toolkit, DSigma_mis is ONE integral —
// no spline of Sigma_mis and no power-law inner extrapolation:
//
//   DSigma_mis(R) = int_0^1 dt int_0^inf dx int_0^pi dtheta/pi  x e^-x
//                   [ 4 t^3 Sigma(r(R t^2, x sigma, theta)) - Sigma(r(R, x sigma, theta)) ],
//   r(R', R_s, theta) = sqrt(R'^2 + R_s^2 + 2 R' R_s cos theta),
//
// with the exact maps R' = R t^2 (aperture) and x = y / (1 - y) (offsets
// on [0, inf), no x_max cut).  Slow; for validating MiscenteringOperator.
class MiscenteringCuhre {
public:
  using Fn = std::function<double(double)>;

  struct Result {
    double value, error;
    long long neval;
    bool converged;
  };

  // Any continuous profile r -> Sigma(r) (e.g. the module's own max model).
  explicit MiscenteringCuhre(Fn sigma, double epsrel = 1.0e-6,
                             double epsabs = 0.0, long long maxeval = 50'000'000)
    : sigma_(std::move(sigma)), epsrel_(epsrel), epsabs_(epsabs), maxeval_(maxeval)
  {
    cubacpp::turn_off_cuba_forking();
  }

  // cluster-toolkit input: cubic spline in ln r of (r_tab, sigma_tab), core(r)
  // below r_tab[0], 0 above r_tab.back().
  MiscenteringCuhre(std::vector<double> const& r_tab,
                    std::vector<double> const& sigma_tab, Fn core,
                    double epsrel = 1.0e-6, double epsabs = 0.0,
                    long long maxeval = 50'000'000)
    : MiscenteringCuhre(spline_profile(r_tab, sigma_tab, std::move(core)),
                        epsrel, epsabs, maxeval)
  {}

  double sigma(double r) const { return sigma_(r); }

  // (E1): 2-D Cuhre over (y, theta).
  Result surface_density(double R, double sigma_off) const
  {
    auto f = [&](double y, double theta) {
      return offset_weight(y) * sigma(ring(R, sigma_off * offset_x(y), theta));
    };
    return run(f, cubacpp::IntegrationVolume<2>({0.0, 0.0}, {1.0, PI}), 1.0 / PI);
  }

  // (E1)+(E2): one 3-D Cuhre over (t, y, theta).
  Result excess_surface_density(double R, double sigma_off) const
  {
    auto f = [&](double t, double y, double theta) {
      double const rs = sigma_off * offset_x(y);
      return offset_weight(y) *
             (4.0 * t * t * t * sigma(ring(R * t * t, rs, theta)) -
              sigma(ring(R, rs, theta)));
    };
    return run(f, cubacpp::IntegrationVolume<3>({0.0, 0.0, 0.0}, {1.0, 1.0, PI}),
               1.0 / PI);
  }

private:
  static constexpr double PI = 3.14159265358979323846;

  static double ring(double Rp, double rs, double theta)
  {
    return std::sqrt(Rp * Rp + rs * rs + 2.0 * Rp * rs * std::cos(theta));
  }
  // x = y / (1 - y); P(R_s) dR_s = x e^-x dx, dx = dy / (1 - y)^2.
  static double offset_x(double y) { return y / (1.0 - y); }
  static double offset_weight(double y)
  {
    if (!(y < 1.0)) return 0.0;
    double const x = offset_x(y), d = 1.0 - y;
    return x * std::exp(-x) / (d * d);
  }

  template <typename F, std::size_t N>
  Result run(F const& f, cubacpp::IntegrationVolume<N> const& vol, double norm) const
  {
    cubacpp::Cuhre alg;
    alg.maxeval = maxeval_;
    auto const res = alg.integrate(f, epsrel_, epsabs_, vol);
    return {norm * res.value, norm * res.error, res.neval, res.converged()};
  }

  static Fn spline_profile(std::vector<double> const& r_tab,
                           std::vector<double> const& sigma_tab, Fn core)
  {
    CubicSplineLnR const spline(r_tab, sigma_tab);
    return [spline, core = std::move(core)](double r) {
      if (r < spline.r_lo()) return core(r);
      if (r > spline.r_hi()) return 0.0;
      return spline(r);
    };
  }

  Fn sigma_;
  double epsrel_, epsabs_;
  long long maxeval_;
};

// Stacked gamma miscentering for a z-resolved (lnM_k, z_q) consumer, the
// piece a stacked-shear module such as Shear1h2hMax calls:
//
//   O_mis(b, R) = sum_{k,q} W(b; k, q) DSigma_mis(R | lnM_k, z_q; sigma_b),
//   O           = (1 - f_mis) O_cen + f_mis O_mis                    (apply)
//
// of the consumer's centred profile Sigma_sel = B_sel Sigma (Bsel first,
// then miscentering), sigma_b = tau_mis R_lambda(lob_b).
//
// Method::gauss_legendre (fast, default): Sigma(r | k, q) is tabulated once
//   per sample on a LogRadialGrid r_m (core slot when the profile has a
//   core), MiscenteringOperator rows L_{slot,R} are built per (richness
//   slot, wall R) and cached while tau_mis and the grid floor are
//   unchanged, and the stack is folded in once per sample,
//     g_b[m] = sum_q B(r_m; z_q) sum_k W(b; k, q) Sigma(r_m | k, q),
//   so evaluate() is a dot product L . g_b.  B is evaluated once per model
//   row (Costanzi26: richness slot x z; Sunayama23: bin), not once per bin.
// Method::cuhre (reference): every DSigma_mis(R | k, q) by MiscenteringCuhre
//   of the pointwise profile; slow (use a coarse lnM x z grid).
//   Non-convergence is reported per sample on stderr, not fatal.
class Miscentering {
public:
  enum class Method { gauss_legendre, cuhre };

  struct Options {
    Method method = Method::gauss_legendre;
    std::size_t n_offset = MiscenteringOperator::DEFAULT_N_OFFSET;
    std::size_t n_theta = MiscenteringOperator::DEFAULT_N_THETA;
    std::size_t n_aperture = MiscenteringOperator::DEFAULT_N_APERTURE;
    std::size_t n_r = 1024;
    double cuhre_epsrel = 1.0e-6;
    long long cuhre_maxeval = 50'000'000;

    // Module-section keys: mis_method (gauss_legendre | cuhre),
    // n_mis_offset, n_mis_theta, n_mis_aperture, n_r_mis,
    // mis_cuhre_epsrel, mis_cuhre_maxeval.
    static Options from_config(cosmosis::DataBlock& cfg, char const* section)
    {
      Options o;
      if (cfg.has_val(section, "mis_method")) {
        auto const m = cfg.view<std::string>(section, "mis_method");
        if (m == "cuhre") o.method = Method::cuhre;
        else if (m != "gauss_legendre")
          throw std::invalid_argument("Miscentering: unknown mis_method " + m);
      }
      auto get_n = [&](char const* key, std::size_t& v) {
        if (cfg.has_val(section, key)) v = static_cast<std::size_t>(cfg.view<int>(section, key));
      };
      get_n("n_mis_offset", o.n_offset);
      get_n("n_mis_theta", o.n_theta);
      get_n("n_mis_aperture", o.n_aperture);
      get_n("n_r_mis", o.n_r);
      if (cfg.has_val(section, "mis_cuhre_epsrel"))
        o.cuhre_epsrel = cfg.view<double>(section, "mis_cuhre_epsrel");
      if (cfg.has_val(section, "mis_cuhre_maxeval"))
        o.cuhre_maxeval = cfg.view<int>(section, "mis_cuhre_maxeval");
      return o;
    }
  };

  // The consumer's centred profile on its (k, q) nodes, for one sample.
  struct Profile {
    double r_lo;                 // grid floor
    double r_max;                // profile valid up to (throws if exceeded)
    std::vector<double> core;    // (k, q) uniform-disk Sigma below r_lo; empty = none
    // Sigma(r | k, q) for every node: out[k * n_z + q] (gauss_legendre).
    std::function<void(double, std::vector<double>&)> grid;
    // Sigma(r | k, q) at one node, incl. the core (cuhre).
    std::function<double(double, std::size_t, std::size_t)> at;
  };

  // The consumer's stack.
  struct Stack {
    std::size_t n_bins, n_lnm, n_z;
    double const* weights;           // W: index (b * n_lnm + k) * n_z + q
    std::vector<double> const* z;    // z nodes (Bsel)
    BselModels const* bsel;          // nullptr without Bsel
  };

  Miscentering(Options options, std::vector<double> wall_radii,
               std::vector<double> lob_centers)
    : opt_(options), wall_(std::move(wall_radii)), lob_(std::move(lob_centers))
  {
    if (wall_.empty() || lob_.empty())
      throw std::invalid_argument("Miscentering: needs wall radii and lob_centers");
  }

  Method method() const { return opt_.method; }
  MiscenteringOperator const& op() const { return *op_; }

  // Reads miscentering/{f_mis, tau_mis} (required) and prepares O_mis on
  // the wall.  The Stack weights and the Profile functions must outlive
  // the next set_sample (off-wall evaluate() uses them).
  void set_sample(cosmosis::DataBlock& s, Stack const& stack, Profile const& profile)
  {
    stack_ = stack;
    profile_ = profile;
    op_.emplace(MiscenteringOperator::from_datablock(
      s, "miscentering", opt_.n_offset, opt_.n_theta, opt_.n_aperture));
    sigma_.resize(lob_.size());
    for (std::size_t j = 0; j != lob_.size(); ++j)
      sigma_[j] = op_->offset_scale(lob_[j]);
    double const r_hi = op_->max_ring_radius(
      *std::max_element(wall_.begin(), wall_.end()),
      *std::max_element(sigma_.begin(), sigma_.end()));
    if (r_hi > profile.r_max)
      throw std::runtime_error(
        "Miscentering: the profile must reach 2 max(r_perp) + x_max sigma_max");
    if (opt_.method == Method::cuhre)
      set_sample_cuhre();
    else
      set_sample_gauss_legendre(r_hi);
  }

  // O_mis(b, R): the stacked miscentred profile.
  double evaluate(int b, double R) const
  {
    std::size_t const j = static_cast<std::size_t>(b) % lob_.size();
    auto const it = std::find(wall_.begin(), wall_.end(), R);
    bool const on_wall = it != wall_.end();
    std::size_t const i = static_cast<std::size_t>(it - wall_.begin());
    if (opt_.method == Method::cuhre) {
      if (on_wall)
        return cuhre_[static_cast<std::size_t>(b) * wall_.size() + i];
      double const* w = stack_.weights + static_cast<std::size_t>(b) * stack_.n_lnm * stack_.n_z;
      int const key = stack_.bsel ? b : static_cast<int>(j);
      double acc = 0.0;
      for (std::size_t k = 0; k != stack_.n_lnm; ++k)
        for (std::size_t q = 0; q != stack_.n_z; ++q)
          acc += w[k * stack_.n_z + q] * cuhre_excess_surface_density(key, R, k, q);
      return acc;
    }
    std::vector<double> local;
    double const* row = nullptr;
    if (on_wall) {
      row = rows_[j * wall_.size() + i].data();
    } else {
      op_->excess_surface_density_row(R, sigma_[j], *grid_, local);
      row = local.data();
    }
    double const* g = &g_[static_cast<std::size_t>(b) * grid_->row_size()];
    double acc = 0.0;
    for (std::size_t m = 0; m != grid_->row_size(); ++m)
      acc += row[m] * g[m];
    return acc;
  }

  // O = (1 - f_mis) O_cen + f_mis O_mis
  double apply(double o_cen, double o_mis) const { return op_->apply(o_cen, o_mis); }

  struct CuhreStats {
    std::size_t n{0}, not_converged{0};
    long long neval{0};
    double max_rel_err{0.0};
  };
  CuhreStats const& cuhre_stats() const { return stats_; }

private:
  void set_sample_gauss_legendre(double r_hi)
  {
    std::size_t const n_wall = wall_.size(), n_lob = lob_.size();
    std::size_t const n_z = stack_.n_z, n_lnm = stack_.n_lnm, n_kq = n_lnm * n_z;
    bool const core = !profile_.core.empty();
    if (op_->tau_mis() != tau_cached_ || profile_.r_lo != r_lo_cached_) {
      grid_.emplace(profile_.r_lo, r_hi, opt_.n_r, core);
      rows_.assign(n_lob * n_wall, {});
      for (std::size_t j = 0; j != n_lob; ++j)
        for (std::size_t i = 0; i != n_wall; ++i)
          op_->excess_surface_density_row(wall_[i], sigma_[j], *grid_, rows_[j * n_wall + i]);
      tau_cached_ = op_->tau_mis();
      r_lo_cached_ = profile_.r_lo;
    }
    auto const& grid = *grid_;
    std::size_t const n_row = grid.row_size();
    // Radius of each row entry (the core slot evaluates B at r_lo).
    auto row_radius = [&](std::size_t m) { return m < grid.size() ? grid.r(m) : profile_.r_lo; };
    auto const B = bsel_table(n_row, row_radius);

    g_.assign(stack_.n_bins * n_row, 0.0);
    std::vector<double> sigma(n_kq), gq(n_z);
    for (std::size_t m = 0; m != n_row; ++m) {
      if (m == grid.size())
        sigma = profile_.core;
      else
        profile_.grid(grid.r(m), sigma);
      for (std::size_t b = 0; b != stack_.n_bins; ++b) {
        double const* w = stack_.weights + b * n_kq;
        std::fill(gq.begin(), gq.end(), 0.0);
        for (std::size_t k = 0; k != n_lnm; ++k)
          for (std::size_t q = 0; q != n_z; ++q)
            gq[q] += w[k * n_z + q] * sigma[k * n_z + q];
        double const* Bm = B.empty() ? nullptr : &B[bsel_key(b)][m * n_z];
        double sum = 0.0;
        for (std::size_t q = 0; q != n_z; ++q)
          sum += gq[q] * (Bm ? Bm[q] : 1.0);
        g_[b * n_row + m] = sum;
      }
    }
  }

  // B_sel(r_m; z_q) per distinct model row: Costanzi26 depends on the
  // richness slot and z, Sunayama23 on the bin only.  Empty without Bsel.
  // Layout [key][m * n_z + q].
  template <typename RowRadius>
  std::vector<std::vector<double>> bsel_table(std::size_t n_row, RowRadius&& row_radius) const
  {
    std::vector<std::vector<double>> B;
    if (!stack_.bsel)
      return B;
    auto const& model = *stack_.bsel;
    std::size_t const n_z = stack_.n_z;
    bool const by_bin = model.tag() == BselModels::Tag::Sunayama23;
    B.resize(by_bin ? stack_.n_bins : lob_.size());
    for (std::size_t key = 0; key != B.size(); ++key) {
      int const bin = static_cast<int>(key);  // Sunayama: key = bin
      double const lob = lob_[key % lob_.size()];
      B[key].resize(n_row * n_z);
      for (std::size_t m = 0; m != n_row; ++m) {
        double const r = row_radius(m);
        if (by_bin) {
          std::fill_n(&B[key][m * n_z], n_z, model(r, bin, lob, 0.0));
        } else {
          for (std::size_t q = 0; q != n_z; ++q)
            B[key][m * n_z + q] = model(r, bin, lob, (*stack_.z)[q]);
        }
      }
    }
    return B;
  }

  std::size_t bsel_key(std::size_t b) const
  {
    return stack_.bsel->tag() == BselModels::Tag::Sunayama23 ? b : b % lob_.size();
  }

  // Without Bsel the integral depends on the bin only through sigma (lob slot).
  void set_sample_cuhre()
  {
    std::size_t const n_wall = wall_.size(), n_z = stack_.n_z, n_kq = stack_.n_lnm * n_z;
    std::size_t const n_key = stack_.bsel ? stack_.n_bins : lob_.size();
    stats_ = {};
    std::vector<std::vector<double>> dmis(n_key * n_wall);  // [key][i][kq]
    for (std::size_t key = 0; key != n_key; ++key)
      for (std::size_t i = 0; i != n_wall; ++i) {
        auto& out = dmis[key * n_wall + i];
        out.resize(n_kq);
        for (std::size_t k = 0; k != stack_.n_lnm; ++k)
          for (std::size_t q = 0; q != n_z; ++q)
            out[k * n_z + q] = cuhre_excess_surface_density(static_cast<int>(key), wall_[i], k, q);
      }
    cuhre_.assign(stack_.n_bins * n_wall, 0.0);
    for (std::size_t b = 0; b != stack_.n_bins; ++b) {
      std::size_t const key = stack_.bsel ? b : b % lob_.size();
      double const* w = stack_.weights + b * n_kq;
      for (std::size_t i = 0; i != n_wall; ++i) {
        double acc = 0.0;
        for (std::size_t idx = 0; idx != n_kq; ++idx)
          acc += w[idx] * dmis[key * n_wall + i][idx];
        cuhre_[b * n_wall + i] = acc;
      }
    }
    std::fprintf(stderr,
                 "[Miscentering] mis_method = cuhre: %zu integrals, %.3g "
                 "evaluations, max rel error estimate %.2e, %zu at maxeval\n",
                 stats_.n, static_cast<double>(stats_.neval), stats_.max_rel_err,
                 stats_.not_converged);
  }

  // DSigma_mis(R | k, q) of B Sigma by MiscenteringCuhre; `key` is the bin
  // (with Bsel) or the lob slot (without).
  double cuhre_excess_surface_density(int key, double R, std::size_t k, std::size_t q) const
  {
    std::size_t const slot = static_cast<std::size_t>(key) % lob_.size();
    double const lob = lob_[slot];
    double const z = stack_.z ? (*stack_.z)[q] : 0.0;
    MiscenteringCuhre const ref(
      [&](double r) {
        double const b = stack_.bsel ? (*stack_.bsel)(std::max(r, profile_.r_lo), key, lob, z) : 1.0;
        return b * profile_.at(r, k, q);
      },
      opt_.cuhre_epsrel, 0.0, opt_.cuhre_maxeval);
    auto const res = ref.excess_surface_density(R, sigma_[slot]);
    // The profile's core step and r_trans corner can keep Cuhre just above
    // a tight epsrel: reported, not fatal.
    ++stats_.n;
    stats_.neval += res.neval;
    if (!res.converged) ++stats_.not_converged;
    if (res.value != 0.0)
      stats_.max_rel_err = std::max(stats_.max_rel_err, std::abs(res.error / res.value));
    return res.value;
  }

  Options opt_;
  std::vector<double> wall_, lob_;
  Stack stack_{};
  Profile profile_{};
  std::optional<MiscenteringOperator> op_;
  std::vector<double> sigma_;                     // per lob slot
  std::optional<LogRadialGrid> grid_;
  std::vector<std::vector<double>> rows_;         // L per (lob slot, wall R)
  double tau_cached_{std::numeric_limits<double>::quiet_NaN()};
  double r_lo_cached_{std::numeric_limits<double>::quiet_NaN()};
  std::vector<double> g_;                         // g_b[m]
  std::vector<double> cuhre_;                     // O_mis per (bin, wall R), cuhre
  mutable CuhreStats stats_;
};

}  // namespace y3_cluster

#endif
