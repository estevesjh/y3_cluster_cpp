// Traditional 1h+2h excess surface density via the max model — C++ backend.
//
//   DeltaSigma_tot(R, lnM, z | bin) = max(
//       DeltaSigma_cl(R, lnM | bin),
//       b(lnM, z) * DeltaSigma_hh(R, z) )
//
// the Y1-era SIG_MAX/KAPPA_MAX/GAMMA_MAX composition on the modern
// haloModel tables (same observable as the Python
// ../python/shear1h2h_max.py), stacked with the selection weight
//
//   O_ij(R) = int dz int dlnM  n dV/dOmegadz Omega Sigma_crit^-1
//             S_ij(lnM, z) DeltaSigma_tot(R, lnM, z).
//
// The output `shear1h2h_max/vals` is a stacked DeltaSigma, not Sigma and not
// gamma_T.  Tangential shear is obtained by applying the lens/source
// geometry: gamma_T(R) = DeltaSigma(R) Sigma_crit^{-1}.  This module already
// includes Sigma_crit^{-1} in its stack weight, as required by its historical
// observable contract; the profile helper exposes
// Sigma, barSigma, DeltaSigma, and gamma_T separately.
//
// Structure note: the two-halo term is z-dependent, so — unlike
// Shear1hGl, which reuses SelGlWeights's z-contracted weight — the
// redshift integral must stay inside the mass integral. This driver
// therefore builds the z-RESOLVED weight W2d(bin; lnM, z) on the same
// fixed GL nodes (same term composition as SelGlWeights, just without the
// z sum) and contracts (lnM, z) per (bin, R).
//
// Every table is read through the project's interpolation primitives
// (Interp2D / Interp1D with clamped queries): haloModel dSigma_nfw,
// bias and dSigma_hh, average_sigma_crit_inv, the selection tensor via
// SelFunction_t, and the miscentred NFW look-up via NFW_DSIGMA_MIS.
//
// dSigma_hh historically carried NaN over part of its (R, z) table
// (docs/known_issues/dsigma_hh_debug_flag.md — since FIXED in the
// producer); the sanitize-to-0 step is kept as a defensive guard, which
// is exact for a max model (max(1h, 0) = 1h wherever the 2h term were
// undefined). Requires compute_lensing_2h = T.
//
// Options: bin_index x r_perp cartesian grid (bin slow / R fast),
// zt_low/zt_high/lnm_low/lnm_high (required), n_lnm (96), n_z (64),
// lob_centers, include_miscentering (default T), max_xi (default F),
// n_gl (20), n_root (96) for the opt-in max_xi path, n_offset (24),
// n_phi (64), n_aperture (8), q_max (12),
// miscentering_method ("table" or "direct", default "table"), and
// use_nfw_table_residual (default T).
// f_mis and tau_mis are REQUIRED datablock values (miscentering/f_mis,
// miscentering/tau_mis): set_sample throws if the section is missing —
// no silent fallback to the Y3 fiducial defaults.
// Output: shear1h2h_max/vals — same section as the Python backend
// (interchangeable; never run both in one pipeline).
// Status: reference implementation of the traditional-shear arm.
#ifndef Y3_CLUSTER_CPP_SHEAR1H2H_MAX_T_HH
#define Y3_CLUSTER_CPP_SHEAR1H2H_MAX_T_HH

#include "cosmosis/datablock/datablock.hh"
#include "cosmosis/datablock/ndarray.hh"

#include "models/dv_do_dz_t.hh"
#include "models/hmf_t.hh"
#include "pipelines/shared/sel_gl_weights.hh"
#include "models/nfw_dsigma_mis.hh"
#include "models/omega_z_des.hh"
#include "models/sel_function_t.hh"
#include "shear1h2h_max_profile.hh"
#include "pipelines/shared/lensing_helpers.hh"
#include "utils/datablock_reader.hh"
#include "utils/interp_1d.hh"
#include "utils/interp_2d.hh"
#include "utils/make_grid_points.hh"
#include "utils/make_interp_2d.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

class Shear1h2hMax {
public:
  using grid_t = y3_cluster::grid_t<2>;
  using grid_point_t = grid_t::value_type;
  static constexpr std::size_t n_outputs = 1;

  explicit Shear1h2hMax(cosmosis::DataBlock& cfg)
    : N_lnm_(cfg.has_val(module_label(), "n_lnm")
               ? cfg.view<int>(module_label(), "n_lnm") : 96)
    , N_z_(cfg.has_val(module_label(), "n_z")
             ? cfg.view<int>(module_label(), "n_z") : 64)
    , zt_lo_(cfg.view<double>(module_label(), "zt_low"))
    , zt_hi_(cfg.view<double>(module_label(), "zt_high"))
    , lnm_lo_(cfg.view<double>(module_label(), "lnm_low"))
    , lnm_hi_(cfg.view<double>(module_label(), "lnm_high"))
    , include_mis_(cfg.has_val(module_label(), "include_miscentering")
                     ? cfg.view<int>(module_label(),
                                     "include_miscentering") != 0
                     : true)
    , max_xi_(cfg.has_val(module_label(), "max_xi")
                          ? cfg.view<int>(module_label(),
                                          "max_xi") != 0
                          : false)
    // TODO(#14): drop the hardcoded c=4 — once claude/issue-4-dsigma-hh-2h-term
    // merges, call dsigma_mis_.set_concentration_table(
    //   make_Interp1D(s, "haloModel", "lnM", "concentration"))
    // in set_sample so the miscentred term uses the same per-mass Child18
    // concentration as the centred dSigma_nfw table.
    , dsigma_mis_(y3_cluster::CONC, y3_cluster::RHOC, y3_cluster::GAMMA)
  {
    int const n_gl = cfg.has_val(module_label(), "n_gl")
      ? cfg.view<int>(module_label(), "n_gl") : 20;
    int const n_root = cfg.has_val(module_label(), "n_root")
      ? cfg.view<int>(module_label(), "n_root") : 96;
    if (n_gl < 1)
      throw std::invalid_argument("Shear1h2hMax: n_gl must be positive");
    if (n_root < 2)
      throw std::invalid_argument(
        "Shear1h2hMax: n_root must be at least 2");
    N_gl_ = static_cast<std::size_t>(n_gl);
    N_root_ = static_cast<std::size_t>(n_root);
    int const n_offset = cfg.has_val(module_label(), "n_offset")
      ? cfg.view<int>(module_label(), "n_offset") : 24;
    int const n_phi = cfg.has_val(module_label(), "n_phi")
      ? cfg.view<int>(module_label(), "n_phi") : 64;
    int const n_aperture = cfg.has_val(module_label(), "n_aperture")
      ? cfg.view<int>(module_label(), "n_aperture") : 8;
    q_max_ = cfg.has_val(module_label(), "q_max")
      ? cfg.view<double>(module_label(), "q_max") : 12.0;
    if (n_offset < 1 || n_phi < 1 || n_aperture < 1 || q_max_ <= 0.0)
      throw std::invalid_argument(
        "Shear1h2hMax: invalid miscentering quadrature");
    N_offset_ = static_cast<std::size_t>(n_offset);
    N_phi_ = static_cast<std::size_t>(n_phi);
    N_aperture_ = static_cast<std::size_t>(n_aperture);
    miscentering_method_ = cfg.has_val(
      module_label(), "miscentering_method")
      ? cfg.view<std::string>(module_label(), "miscentering_method")
      : "table";
    if (miscentering_method_ != "table" &&
        miscentering_method_ != "direct")
      throw std::invalid_argument(
        "Shear1h2hMax: miscentering_method must be 'table' or 'direct'");
    use_nfw_table_residual_ = cfg.has_val(
      module_label(), "use_nfw_table_residual")
      ? cfg.view<int>(module_label(), "use_nfw_table_residual") != 0 : true;
    y3_pipelines::gl_nodes(lnm_lo_, lnm_hi_, N_lnm_, lnm_x_, lnm_w_);
    y3_pipelines::gl_nodes(zt_lo_, zt_hi_, N_z_, z_x_, z_w_);
    lob_centers_ =
      y3_pipelines::read_lob_centers(cfg, module_label());
    if (lob_centers_.empty())
      throw std::runtime_error("Shear1h2hMax: lob_centers is empty");
  }

  void
  set_sample(cosmosis::DataBlock& s)
  {
    namespace w = y3_pipelines;

    y3_cluster::HMF_t const hmf(s);
    y3_cluster::DV_DO_DZ_t const dv(s);
    y3_cluster::OMEGA_Z_DES const omega(s);
    auto const sci = w::load_sigma_crit_inv(s);

    dsigma_nfw_.emplace(y3_cluster::make_Interp2D(
      s, "haloModel", "r_sigma", "lnM", "dSigma_nfw"));
    bias_.emplace(
      y3_cluster::make_Interp2D(s, "haloModel", "lnM", "z", "bias"));
    dsigma_hh_.emplace(make_sanitized_hh(s));

    // Required: no fallback to the fiducial defaults — a pipeline that
    // has not published the miscentering section must fail loudly.
    f_mis_ = include_mis_ ? s.view<double>("miscentering", "f_mis") : 0.0;
    double const tau_mis = s.view<double>("miscentering", "tau_mis");
    // UNIFIED rho_m convention (2026-08-24): boundary AND amplitude on
    // haloModel/rho_m_ref (same density as the centred dSigma_nfw table).
    dsigma_mis_.set_rho_ref(s.view<double>("haloModel", "rho_m_ref"));
    // Physical mean density (opt-in): exact per-z-node identity on the
    // 1-halo mixture ONLY (see evaluate; the 2-halo row is untouched).
    int phys = 0;
    if (s.has_val("haloModel", "one_halo_physical_density"))
      s.get_val("haloModel", "one_halo_physical_density", phys);
    phys_density_ = (phys != 0);

    if (max_xi_)
      profile_.emplace(s, N_gl_, N_root_, N_offset_, N_phi_,
                       N_aperture_, q_max_);

    // z-only factors (Sigma_crit_inv folded in, as the shear weight
    // requires) and the z-RESOLVED weight W2d(bin; lnM, z).
    std::vector<double> zfac(z_x_.size());
    for (std::size_t q = 0; q != z_x_.size(); ++q)
      zfac[q] = z_w_[q] * dv(z_x_[q]) * omega(z_x_[q]) * sci.clamp(z_x_[q]);

    int const n_bins = y3_pipelines::n_bins_from_block(s);
    n_bins_ = static_cast<std::size_t>(n_bins);
    w2d_.assign(n_bins_ * N_lnm_ * N_z_, 0.0);
    for (int b = 0; b != n_bins; ++b) {
      y3_cluster::SelFunction_t const sel(s, b);
      for (std::size_t k = 0; k != N_lnm_; ++k)
        for (std::size_t q = 0; q != N_z_; ++q)
          w2d_[(b * N_lnm_ + k) * N_z_ + q] =
            lnm_w_[k] * zfac[q] * hmf(lnm_x_[k], z_x_[q]) *
            sel(lnm_x_[k], z_x_[q]);
    }

    // b(lnM, z) on the node grid — bin-independent, so hoisted here
    // (this is the cache that keeps evaluate() free of interpolation
    // in the inner double loop).
    bias_kq_.assign(N_lnm_ * N_z_, 0.0);
    for (std::size_t k = 0; k != N_lnm_; ++k)
      for (std::size_t q = 0; q != N_z_; ++q)
        bias_kq_[k * N_z_ + q] = bias_->clamp(lnm_x_[k], z_x_[q]);

    r_mis_.assign(n_bins_, 0.0);
    for (std::size_t b = 0; b != n_bins_; ++b)
      r_mis_[b] = tau_mis * w::R_lambda(
                              lob_centers_[b % lob_centers_.size()]);
  }

  std::array<double, n_outputs>
  evaluate(grid_point_t const& pt) const
  {
    int const b = static_cast<int>(pt[0]);
    double const R = pt[1];
    if (b < 0 || static_cast<std::size_t>(b) >= n_bins_)
      throw std::out_of_range("Shear1h2hMax: bin_index out of range");

    // Per-(bin, R) profile rows: the 1-halo term is z-free (one value
    // per mass node), the 2-halo table row is mass-free (one per z
    // node). Both are interpolated once here, so the (lnM, z) double
    // sum below touches no interpolator.
    std::vector<double> one(N_lnm_), two(N_z_);
    if (!max_xi_)
      for (std::size_t k = 0; k != N_lnm_; ++k) {
        double const d_cen = dsigma_nfw_->clamp(R, lnm_x_[k]);
        double const d_mis = dsigma_mis_(R, r_mis_[b], lnm_x_[k]);
        one[k] = (1.0 - f_mis_) * d_cen + f_mis_ * d_mis;
      }
    for (std::size_t q = 0; q != N_z_; ++q)
      two[q] = dsigma_hh_->clamp(R, z_x_[q]);

    double acc = 0.0;
    double const* w2 = &w2d_[static_cast<std::size_t>(b) * N_lnm_ * N_z_];
    if (max_xi_) {
      for (std::size_t k = 0; k != N_lnm_; ++k) {
        double const* wrow = w2 + k * N_z_;
        for (std::size_t q = 0; q != N_z_; ++q) {
          double const qf = phys_density_ ? 1.0 + z_x_[q] : 1.0;
          double const centered = profile_->excess_surface_density(
            R * qf, lnm_x_[k], z_x_[q]) * qf * qf;
          if (miscentering_method_ == "direct" && f_mis_ != 0.0) {
            double const direct_mis =
              profile_->miscentered_excess_surface_density(
                R * qf, r_mis_[b] * qf, lnm_x_[k], z_x_[q]) * qf * qf;
            double const total = centered + f_mis_ * (
              direct_mis - centered);
            acc += wrow[q] * total;
            continue;
          }
          double const d_mis = dsigma_mis_(R * qf, r_mis_[b] * qf,
                                           lnm_x_[k]) * qf * qf;
          // The legacy approximation applies the tabulated correction to the
          // 1-halo leg only.  The residual mode below instead adds the
          // gamma-convolved Sigma_max - Sigma_NFW residual.
          double const d_cen = dsigma_nfw_->clamp(R * qf, lnm_x_[k]) * qf * qf;
          double total = centered + f_mis_ * (d_mis - d_cen);
          if (use_nfw_table_residual_ && f_mis_ != 0.0) {
            double const nfw_cen =
              profile_->nfw_excess_surface_density(
                R * qf, lnm_x_[k]) * qf * qf;
            double const residual_cen =
              profile_->residual_excess_surface_density(
                R * qf, lnm_x_[k], z_x_[q]) * qf * qf;
            double const residual_mis =
              profile_->miscentered_residual_excess_surface_density(
                R * qf, r_mis_[b] * qf, lnm_x_[k], z_x_[q]) * qf * qf;
            total = centered + f_mis_ * (
              d_mis + residual_mis - nfw_cen - residual_cen);
          }
          acc += wrow[q] * total;
        }
      }
      return {acc};
    }
    if (!phys_density_) {
      for (std::size_t k = 0; k != N_lnm_; ++k) {
        double const* wrow = w2 + k * N_z_;
        double const* brow = &bias_kq_[k * N_z_];
        double const one_k = one[k];
        for (std::size_t q = 0; q != N_z_; ++q)
          acc += wrow[q] * std::max(one_k, brow[q] * two[q]);
      }
      return {acc};
    }
    // Physical density: the 1-halo mixture becomes z-dependent through
    // the exact identity DSigma_phys(R|z) = (1+z)^2 DSigma_com(R (1+z)),
    // evaluated at every z node (slower; opt-in diagnostic mode). The
    // 2-halo row keeps its own convention.
    for (std::size_t k = 0; k != N_lnm_; ++k) {
      double const* wrow = w2 + k * N_z_;
      double const* brow = &bias_kq_[k * N_z_];
      for (std::size_t q = 0; q != N_z_; ++q) {
        double const qf = 1.0 + z_x_[q];
        double const d_cen = dsigma_nfw_->clamp(R * qf, lnm_x_[k]);
        double const d_mis =
          dsigma_mis_(R * qf, r_mis_[b] * qf, lnm_x_[k]);
        double const one_kq =
          (qf * qf) * ((1.0 - f_mis_) * d_cen + f_mis_ * d_mis);
        acc += wrow[q] * std::max(one_kq, brow[q] * two[q]);
      }
    }
    return {acc};
  }

  static char const* module_label() { return "Shear1h2hMax"; }

  static std::array<char const*, n_outputs>
  output_sections()
  {
    return {"shear1h2h_max"};
  }

  static grid_t
  make_grid_points(cosmosis::DataBlock& cfg)
  {
    return y3_cluster::make_grid_points_cartesian_product(
      cfg, module_label(), "bin_index", "r_perp");
  }

private:
  // haloModel/dSigma_hh through Interp2D, with the producer's NaNs
  // replaced by 0 first (docs/known_issues/dsigma_hh_debug_flag.md — since fixed;
  // guard kept). The datablock
  // ndarray is (n_z, n_r) row-major, which is exactly the column-major
  // (x = r_sigma fastest) layout Interp2D's vector constructor wants.
  static y3_cluster::Interp2D
  make_sanitized_hh(cosmosis::DataBlock& s)
  {
    using doubles = std::vector<double>;
    auto const r = s.view<doubles>("haloModel", "r_sigma");
    auto const z = s.view<doubles>("haloModel", "z");
    auto const& nd =
      s.view<cosmosis::ndarray<double>>("haloModel", "dSigma_hh");
    std::vector<double> vals(nd.begin(), nd.end());
    if (vals.size() != r.size() * z.size())
      throw std::runtime_error("Shear1h2hMax: dSigma_hh extents mismatch");
    for (auto& v : vals)
      if (!std::isfinite(v)) v = 0.0;
    return y3_cluster::Interp2D(r, z, vals);
  }

  std::size_t N_lnm_, N_z_;
  std::size_t N_gl_{20}, N_root_{96};
  std::size_t N_offset_{24}, N_phi_{64}, N_aperture_{8};
  double q_max_{12.0};
  double zt_lo_, zt_hi_, lnm_lo_, lnm_hi_;
  bool include_mis_;
  y3_cluster::NFW_DSIGMA_MIS dsigma_mis_;
  std::vector<double> lnm_x_, lnm_w_, z_x_, z_w_, lob_centers_;

  std::optional<y3_cluster::Interp2D> dsigma_nfw_, bias_, dsigma_hh_;
  double f_mis_{0.0};
  std::size_t n_bins_{0};
  bool phys_density_{false};
  bool max_xi_{false};
  std::string miscentering_method_{"table"};
  bool use_nfw_table_residual_{true};
  std::optional<Shear1h2hMaxProfile> profile_;
  std::vector<double> w2d_, bias_kq_, r_mis_;
};

#endif
