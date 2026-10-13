// Traditional 1h+2h excess surface density via the max model — C++ backend.
//
// Formulated via 3D halo exclusion and piecewise approximation:
// 1. In 3D real space, halo exclusion is defined on the correlation function:
//      xi_hm(r) = max(xi_1h(r), b(M, z) * xi_mm(r, z))
//    with transition radius r_trans defined by xi_1h(r_trans) = b(M, z) * xi_mm(r_trans, z).
// 2. The projected excess surface density DeltaSigma_max is approximated around transition
// radius and is assembled by building a piecewise function:
//      For R < r_trans:
//        DeltaSigma_1h(R) + Delta_gap^1 * (R / r_trans)^2
//      For R >= r_trans:
//        b * DeltaSigma_hh(R) + (C_0 / R^2) * 1e-12
//    where C_0 = Delta M_3D(r_trans) / pi (in Msun/h) and
//    Delta_gap^1 = DeltaSigma_max(r_trans) - DeltaSigma_1h(r_trans).
//    r_trans, c0_trans and delta_gap_1 are read from haloModel (published
//    by halo_model when compute_lensing_1h = compute_lensing_2h = T).
//
// Stacked with the selection weight
//
//   O_ij(R) = int dz int dlnM  n dV/dOmegadz Omega Sigma_crit^-1
//             S_ij(lnM, z) DeltaSigma_max(R, lnM, z).
//
// The output `shear1h2h_max/vals` is a stacked DeltaSigma, not Sigma and not
// gamma_T.  Tangential shear is obtained by applying the lens/source
// geometry: gamma_T(R) = DeltaSigma(R) Sigma_crit^{-1}.  This module already
// includes Sigma_crit^{-1} in its stack weight, as required by its historical
// observable contract.
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
// bias, dSigma_hh, r_trans, c0_trans, delta_gap_1, average_sigma_crit_inv,
// and the selection tensor via SelFunction_t.
//
// profile_method selects how DeltaSigma_max(R | lnM, z) is computed:
//   piecewise_approx (default): the r_trans / C_0 / Delta_gap^1 form above.
//   gauss_legendre: exact GL projection of
//     xi_max(r) = max(xi_nfw(r, lnM), b(lnM, z) xi_nl(r, z))
//     (OperatorMax1h2h_gl, pipelines/cosmology/operator_max1h2h_gl.hh),
//     with xi_nfw = rho_NFW / rho_m - 1 from haloModel/concentration and
//     rho_m = haloModel/rho_m_ref.  Needs xi_nl/{r, z, xi_nl}; does NOT
//     need the piecewise tables.  The profile is bin-independent, so
//     set_sample tabulates Sigma at every r_perp wall radius and its
//     aperture nodes once.
//
// Optional bsel (bsel_model = Costanzi26 or Sunayama23) applies, for a
// given (lnM, z), DeltaSigma_sel = B_sel * DeltaSigma_max - M_sel/(pi R^2).
//   piecewise_approx: BselDeltaSigmaOperator, which additionally requires
//     haloModel Sigma_bar_nfw, Sigma_bar_hh and delta_gap_0 for barSigma(<r).
//   gauss_legendre: the same quantity in its equivalent local form
//     DeltaSigma_sel = (2/R^2) int_0^R R' B Sigma dR' - B(R) Sigma(R),
//     i.e. B weights the aperture nodes — no extra tables.
//
// Optional miscentering (miscentering = T): the whole centred profile
// (max model, Bsel-weighted when on) is translated by a gamma-distributed
// offset, sigma_b = tau_mis R_lambda(lob_b), and mixed after the stack,
//   O = (1 - f_mis) O_cen + f_mis O_mis,
// with f_mis, tau_mis REQUIRED from [miscentering].  O_mis is computed by
// y3_cluster::Miscentering (systematics/miscentering/Miscentering.hh); this
// module only hands it the stack (W2d, z nodes, Bsel model) and the centred
// profile Sigma(r | lnM, z):
//   profile_method = piecewise_approx: Sigma_max from the haloModel Sigma
//     tables (Sigma_nfw, Sigma_hh as cubic splines in ln r; delta_gap_0/1)
//     from r_sigma[0], the core below it a uniform disk of mass
//     haloModel/m2d_core (r_sigma[0] ~ 1e-3: halo_model R_perp_core);
//   profile_method = gauss_legendre: the GL projection from 1e-4.
// mis_method picks the integral: gauss_legendre (default; fixed-GL nodes
// collapsed to one weight row per (richness slot, R)) or cuhre (adaptive
// full integral, the reference; piecewise_approx only, slow).
// miscentering = F leaves the output untouched.
//
// Options: bin_index x r_perp cartesian grid (bin slow / R fast),
// zt_low/zt_high/lnm_low/lnm_high (required), n_lnm (96), n_z (64),
// lob_centers (bsel/miscentering richness anchor), bsel (default F),
// bsel_model, profile_method (piecewise_approx), n_los (64),
// n_aperture (32), miscentering (default F), n_mis_offset (32),
// n_mis_theta (32), n_mis_aperture (32), n_r_mis (1024),
// mis_method (gauss_legendre | cuhre), mis_cuhre_epsrel (1e-6),
// mis_cuhre_maxeval (5e7).
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
#include "models/omega_z_des.hh"
#include "models/sel_function_t.hh"
#include "pipelines/cosmology/operator_max1h2h_gl.hh"
#include "pipelines/shared/lensing_helpers.hh"
#include "pipelines/systematics/miscentering/Miscentering.hh"
#include "pipelines/systematics/selection_boost/Bsel.hh"
#include "utils/datablock_reader.hh"
#include "utils/interp_1d.hh"
#include "utils/interp_2d.hh"
#include "utils/make_grid_points.hh"
#include "utils/make_interp_1d.hh"
#include "utils/make_interp_2d.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

class Shear1h2hMax {
public:
  using grid_t = y3_cluster::grid_t<2>;
  using grid_point_t = grid_t::value_type;
  static constexpr std::size_t n_outputs = 1;

  static bool read_bool_opt(cosmosis::DataBlock& cfg,
                            char const* sec,
                            char const* key,
                            bool default_val)
  {
    if (!cfg.has_val(sec, key))
      return default_val;
    try {
      return cfg.view<bool>(sec, key);
    } catch (...) {
      try {
        return cfg.view<int>(sec, key) != 0;
      } catch (...) {
        auto const str = cfg.view<std::string>(sec, key);
        return str == "T" || str == "true" || str == "True" || str == "1";
      }
    }
  }

  explicit Shear1h2hMax(cosmosis::DataBlock& cfg)
    : N_lnm_(cfg.has_val(module_label(), "n_lnm")
               ? cfg.view<int>(module_label(), "n_lnm") : 96)
    , N_z_(cfg.has_val(module_label(), "n_z")
             ? cfg.view<int>(module_label(), "n_z") : 64)
    , zt_lo_(cfg.view<double>(module_label(), "zt_low"))
    , zt_hi_(cfg.view<double>(module_label(), "zt_high"))
    , lnm_lo_(cfg.view<double>(module_label(), "lnm_low"))
    , lnm_hi_(cfg.view<double>(module_label(), "lnm_high"))
    , bsel_enabled_(read_bool_opt(cfg, module_label(), "bsel", false))
    , bsel_model_(bsel_enabled_
                    ? cfg.view<std::string>(module_label(), "bsel_model")
                    : "")
    , profile_method_(cfg.has_val(module_label(), "profile_method")
                        ? cfg.view<std::string>(module_label(), "profile_method")
                        : "piecewise_approx")
    , n_los_(cfg.has_val(module_label(), "n_los")
               ? cfg.view<int>(module_label(), "n_los") : 64)
    , n_aperture_(cfg.has_val(module_label(), "n_aperture")
                    ? cfg.view<int>(module_label(), "n_aperture") : 32)
    , mis_enabled_(read_bool_opt(cfg, module_label(), "miscentering", false))
  {
    y3_pipelines::gl_nodes(lnm_lo_, lnm_hi_, N_lnm_, lnm_x_, lnm_w_);
    y3_pipelines::gl_nodes(zt_lo_, zt_hi_, N_z_, z_x_, z_w_);
    lob_centers_ =
      y3_pipelines::read_lob_centers(cfg, module_label());
    if (lob_centers_.empty())
      throw std::runtime_error("Shear1h2hMax: lob_centers is empty");
    if (bsel_enabled_)
      y3_cluster::BselModels::parse_tag(bsel_model_);
    if (profile_method_ != "piecewise_approx" && profile_method_ != "gauss_legendre")
      throw std::invalid_argument("Shear1h2hMax: unknown profile_method");
    if (cfg.has_val(module_label(), "r_perp"))  // wall radii, tabulated per sample
      wall_radii_ = get_vector_double(cfg, module_label(), "r_perp");
    if (mis_enabled_) {
      if (wall_radii_.empty())
        throw std::runtime_error("Shear1h2hMax: miscentering requires r_perp");
      mis_.emplace(y3_cluster::Miscentering::Options::from_config(cfg, module_label()),
                   wall_radii_, lob_centers_);
      if (mis_->method() == y3_cluster::Miscentering::Method::cuhre &&
          profile_method_ == "gauss_legendre")
        throw std::invalid_argument(
          "Shear1h2hMax: mis_method = cuhre requires profile_method = piecewise_approx");
    }
  }

  void
  set_sample(cosmosis::DataBlock& s)
  {
    namespace w = y3_pipelines;

    bsel_op_.reset();
    if (bsel_enabled_)
      bsel_op_.emplace(
        y3_cluster::BselDeltaSigmaOperator::from_datablock(
          s, bsel_model_));

    y3_cluster::HMF_t const hmf(s);
    y3_cluster::DV_DO_DZ_t const dv(s);
    y3_cluster::OMEGA_Z_DES const omega(s);
    auto const sci = w::load_sigma_crit_inv(s);
    auto const bias = y3_cluster::make_Interp2D(s, "haloModel", "lnM", "z", "bias");

    // z-only factors (Sigma_crit_inv folded in, as the shear weight
    // requires) and the z-RESOLVED weight W2d(bin; lnM, z).
    std::vector<double> zfac(z_x_.size());
    for (std::size_t q = 0; q != z_x_.size(); ++q)
      zfac[q] = z_w_[q] * dv(z_x_[q]) * omega(z_x_[q]) * sci.clamp(z_x_[q]);

    // W2d(b; lnM_k, z_q) = w_k^lnM * zfac(z_q) * dn/dlnM(M_k, z_q)
    //                      * S_b(M_k, z_q),
    // with S_b the probability a halo of (M, z) lands in observed bin b.
    // R-independent, so built once per sample; evaluate() then reduces the
    // (lnM, z) integral to acc = sum_{k,q} W2d * DeltaSigma(R | lnM_k, z_q).
    // Flat layout: index (b * N_lnm + k) * N_z + q.  Not normalised here.
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

    if (profile_method_ == "gauss_legendre")
      set_sample_gauss_legendre(s, bias);
    else
      set_sample_piecewise(s, bias);

    if (mis_)
      set_sample_miscentering(s);
  }

  std::array<double, n_outputs>
  evaluate(grid_point_t const& pt) const
  {
    int const b = static_cast<int>(pt[0]);
    double const R = pt[1];
    if (b < 0 || static_cast<std::size_t>(b) >= n_bins_)
      throw std::out_of_range("Shear1h2hMax: bin_index out of range");
    double const* w2 = &w2d_[static_cast<std::size_t>(b) * N_lnm_ * N_z_];
    double const lob = lob_centers_[b % lob_centers_.size()];
    double const acc_cen = (profile_method_ == "gauss_legendre")
                             ? evaluate_gauss_legendre(w2, b, R, lob)
                             : evaluate_piecewise(w2, b, R, lob);
    if (!mis_)
      return {acc_cen};
    return {mis_->apply(acc_cen, mis_->evaluate(b, R))};
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
  // ---- piecewise_approx -------------------------------------------------

  void
  set_sample_piecewise(cosmosis::DataBlock& s, y3_cluster::Interp2D const& bias)
  {
    sigma_bar_nfw_.reset();
    sigma_bar_hh_.reset();
    if (bsel_enabled_) {
      sigma_bar_nfw_.emplace(y3_cluster::make_Interp2D(
        s, "haloModel", "r_sigma", "lnM", "Sigma_bar_nfw"));
      sigma_bar_hh_.emplace(make_sanitized_table(s, "Sigma_bar_hh"));
    }
    dsigma_nfw_.emplace(y3_cluster::make_Interp2D(
      s, "haloModel", "r_sigma", "lnM", "dSigma_nfw"));
    dsigma_hh_.emplace(make_sanitized_table(s, "dSigma_hh"));
    // Piecewise max-model terms: required, no max(1h, 2h) fallback.
    auto const r_trans = y3_cluster::make_Interp2D(
      s, "haloModel", "lnM", "z", "r_trans");
    auto const c0_trans = y3_cluster::make_Interp2D(
      s, "haloModel", "lnM", "z", "c0_trans");
    auto const delta_gap_1 = y3_cluster::make_Interp2D(
      s, "haloModel", "lnM", "z", "delta_gap_1");
    std::optional<y3_cluster::Interp2D> delta_gap_0;
    if (bsel_enabled_ || mis_enabled_)
      delta_gap_0.emplace(y3_cluster::make_Interp2D(
        s, "haloModel", "lnM", "z", "delta_gap_0"));
    // Miscentering: Sigma_max from the Sigma tables as natural cubic splines
    // in ln r through the r_sigma knots — one per lnM node (Sigma_1h) and
    // one per z node (Sigma_hh), the knot values interpolated in lnM / z
    // only — and the core disk below r_core = r_sigma[0] from
    // haloModel/m2d_core (Msun/h).
    spl_nfw_.clear();
    spl_hh_.clear();
    std::optional<y3_cluster::Interp2D> m2d_core;
    if (mis_enabled_) {
      auto const sigma_nfw = y3_cluster::make_Interp2D(
        s, "haloModel", "r_sigma", "lnM", "Sigma_nfw");
      auto const sigma_hh = make_sanitized_table(s, "Sigma_hh");
      m2d_core.emplace(y3_cluster::make_Interp2D(
        s, "haloModel", "lnM", "z", "m2d_core"));
      auto const r_sigma = s.view<std::vector<double>>("haloModel", "r_sigma");
      r_core_ = r_sigma.front();
      r_sigma_hi_ = r_sigma.back();
      std::vector<double> knots(r_sigma.size());
      for (std::size_t k = 0; k != N_lnm_; ++k) {
        for (std::size_t i = 0; i != r_sigma.size(); ++i)
          knots[i] = sigma_nfw.clamp(r_sigma[i], lnm_x_[k]);
        spl_nfw_.emplace_back(r_sigma, knots);
      }
      for (std::size_t q = 0; q != N_z_; ++q) {
        for (std::size_t i = 0; i != r_sigma.size(); ++i)
          knots[i] = sigma_hh.clamp(r_sigma[i], z_x_[q]);
        spl_hh_.emplace_back(r_sigma, knots);
      }
    }

    // (lnM, z)-only terms on the node grid — bin- and R-independent, so
    // hoisted here to keep evaluate() free of 2D interpolation.
    bias_kq_.assign(N_lnm_ * N_z_, 0.0);
    r_trans_kq_.assign(N_lnm_ * N_z_, 0.0);
    c0_trans_kq_.assign(N_lnm_ * N_z_, 0.0);
    delta_gap_1_kq_.assign(N_lnm_ * N_z_, 0.0);
    delta_gap_0_kq_.assign(N_lnm_ * N_z_, 0.0);
    for (std::size_t k = 0; k != N_lnm_; ++k)
      for (std::size_t q = 0; q != N_z_; ++q) {
        std::size_t const idx = k * N_z_ + q;
        bias_kq_[idx] = bias.clamp(lnm_x_[k], z_x_[q]);
        r_trans_kq_[idx] = r_trans.clamp(lnm_x_[k], z_x_[q]);
        c0_trans_kq_[idx] = c0_trans.clamp(lnm_x_[k], z_x_[q]);
        delta_gap_1_kq_[idx] = delta_gap_1.clamp(lnm_x_[k], z_x_[q]);
        if (delta_gap_0)
          delta_gap_0_kq_[idx] = delta_gap_0->clamp(lnm_x_[k], z_x_[q]);
      }
    sigma_core_kq_.clear();
    if (m2d_core) {
      constexpr double PI = 3.14159265358979323846;
      double const area = PI * r_core_ * r_core_ * 1.0e12;  // -> Msun h / pc^2
      sigma_core_kq_.resize(N_lnm_ * N_z_);
      for (std::size_t k = 0; k != N_lnm_; ++k)
        for (std::size_t q = 0; q != N_z_; ++q)
          sigma_core_kq_[k * N_z_ + q] = m2d_core->clamp(lnm_x_[k], z_x_[q]) / area;
    }

    build_bsel_wall();
  }

  // Bsel on the r_perp wall.  The quadrature nodes r_j of
  // BselDeltaSigmaOperator::apply depend only on (bin, R), and every
  // expensive factor at them depends on lnM OR z, never both:
  //   B_sel(R; z_q), dB_sel/dR(r_j; z_q), barSigma_1h(r_j | lnM_k),
  //   barSigma_hh(r_j | z_q).
  // Tabulating them once per sample leaves evaluate_piecewise() pure
  // arithmetic over (k, q, j): ~2e5 model/interpolation calls per sample
  // instead of ~1.5e7.  Same nodes and operation order as apply().
  void
  build_bsel_wall()
  {
    bsel_wall_.clear();
    if (bsel_op_ && !wall_radii_.empty()) {
      auto const& model = bsel_op_->model();
      std::size_t const n_wall = wall_radii_.size();
      bsel_wall_.resize(n_bins_ * n_wall);
      for (std::size_t b = 0; b != n_bins_; ++b) {
        int const bin = static_cast<int>(b);
        double const lob = lob_centers_[b % lob_centers_.size()];
        for (std::size_t i = 0; i != n_wall; ++i) {
          double const R = wall_radii_[i];
          auto& T = bsel_wall_[b * n_wall + i];
          bsel_op_->nodes(R, bin, T.r, T.c);
          std::size_t const n_j = T.r.size();
          T.b_R.resize(N_z_);
          T.d_bsel.resize(n_j * N_z_);
          T.sbar_hh.resize(n_j * N_z_);
          T.sbar_nfw.resize(n_j * N_lnm_);
          for (std::size_t q = 0; q != N_z_; ++q)
            T.b_R[q] = model(R, bin, lob, z_x_[q]);
          for (std::size_t j = 0; j != n_j; ++j) {
            for (std::size_t q = 0; q != N_z_; ++q) {
              T.d_bsel[j * N_z_ + q] = model.derivative(T.r[j], bin, lob, z_x_[q]);
              T.sbar_hh[j * N_z_ + q] = sigma_bar_hh_->clamp(T.r[j], z_x_[q]);
            }
            for (std::size_t k = 0; k != N_lnm_; ++k)
              T.sbar_nfw[j * N_lnm_ + k] = sigma_bar_nfw_->clamp(T.r[j], lnm_x_[k]);
          }
        }
      }
    }
  }

  double
  evaluate_piecewise(double const* w2, int b, double R, double lob) const
  {
    // Per-(bin, R) profile rows: the 1-halo term is z-free (one value
    // per mass node), the 2-halo table row is mass-free (one per z
    // node). Both are interpolated once here, so the (lnM, z) double
    // sum below touches no interpolator.
    std::vector<double> one(N_lnm_), two(N_z_);
    for (std::size_t k = 0; k != N_lnm_; ++k)
      one[k] = dsigma_nfw_->clamp(R, lnm_x_[k]);
    for (std::size_t q = 0; q != N_z_; ++q)
      two[q] = dsigma_hh_->clamp(R, z_x_[q]);

    // Wall point: Bsel factors tabulated in set_sample_piecewise.
    BselWall const* T = nullptr;
    if (!bsel_wall_.empty()) {
      auto const it = std::find(wall_radii_.begin(), wall_radii_.end(), R);
      if (it != wall_radii_.end())
        T = &bsel_wall_[static_cast<std::size_t>(b) * wall_radii_.size() +
                        static_cast<std::size_t>(it - wall_radii_.begin())];
    }

    double acc = 0.0;
    for (std::size_t k = 0; k != N_lnm_; ++k) {
      double const* wrow = w2 + k * N_z_;
      for (std::size_t q = 0; q != N_z_; ++q) {
        std::size_t const idx = k * N_z_ + q;
        double const d_max = excess_surface_density_node(
          R, one[k], two[q], idx);
        double d_val = d_max;
        if (T) {
          // apply() on tabulated factors:
          //   B(R) d_max - R sum_j c_j barSigma(<r_j) dB/dR(r_j).
          double m_sel_over_pi_r2 = 0.0;
          for (std::size_t j = 0; j != T->r.size(); ++j)
            m_sel_over_pi_r2 +=
              T->c[j] *
              mean_surface_density_value(T->r[j], T->sbar_nfw[j * N_lnm_ + k],
                                         T->sbar_hh[j * N_z_ + q], idx) *
              T->d_bsel[j * N_z_ + q];
          m_sel_over_pi_r2 *= R;
          d_val = T->b_R[q] * d_max - m_sel_over_pi_r2;
        } else if (bsel_op_) {
          // For a given (lnM, z): d_val = B_sel(R) * d_max - M_sel(R) / (pi R^2),
          // with barSigma(<r) from the same piecewise profile.
          d_val = bsel_op_->apply(
            d_max,
            [this, k, q, idx](double r) {
              return mean_surface_density_node(r, k, q, idx);
            },
            R, b, lob, z_x_[q]);
        }
        acc += wrow[q] * d_val;
      }
    }
    return acc;
  }

  // ---- gauss_legendre ---------------------------------------------------

  void
  set_sample_gauss_legendre(cosmosis::DataBlock& s,
                            y3_cluster::Interp2D const& bias)
  {
    // xi_nfw(r, lnM) = rho_NFW / rho_m - 1 with M = M200m, the same
    // convention halo_model uses for xi_1h (rho_s / rho_m = delta_c).
    double const rho_m = s.view<double>("haloModel", "rho_m_ref");
    auto const conc = y3_cluster::make_Interp1D(
      s, "haloModel", "lnM", "concentration");
    auto xi_nfw = [conc, rho_m](double r, double lnM) {
      constexpr double PI = 3.14159265358979323846;
      double const c = conc.clamp(lnM);
      double const r200 = std::cbrt(3.0 * std::exp(lnM) / (800.0 * PI * rho_m));
      double const x = r * c / r200;
      double const delta_c = (200.0 / 3.0) * c * c * c /
                             (std::log1p(c) - c / (1.0 + c));
      return delta_c / (x * (1.0 + x) * (1.0 + x)) - 1.0;
    };
    // Bicubic: bilinear kinks at every r knot stall the GL convergence.
    auto const xi_nl = y3_cluster::make_Interp2D(
      s, "xi_nl", "r", "z", "xi_nl", /*bicubic=*/true);
    double const r_max = s.view<std::vector<double>>("xi_nl", "r").back();

    gl_op_.emplace(
      xi_nfw,
      [bias](double lnM, double z) { return bias.clamp(lnM, z); },
      [xi_nl](double r, double z) { return xi_nl.clamp(r, z); },
      rho_m, r_max, n_los_, n_aperture_);
    gl_aperture_weights_ = gl_op_->aperture_weights();

    // Bin-independent: tabulate every wall radius once per sample.
    gl_sigma_.assign(wall_radii_.size(), {});
    for (std::size_t i = 0; i != wall_radii_.size(); ++i)
      gl_sigma_slots(wall_radii_[i], gl_sigma_[i]);
  }

  // Sigma on the (lnM, z) grid at R (slot 0) and at the aperture radii
  // R t_j^2 (slot j + 1): out[(slot * N_lnm + k) * N_z + q].
  void
  gl_sigma_slots(double R, std::vector<double>& out) const
  {
    std::size_t const n_kq = N_lnm_ * N_z_;
    auto const& t = gl_op_->aperture_nodes();
    out.assign((t.size() + 1) * n_kq, 0.0);
    std::vector<double> sigma;
    gl_op_->surface_density_grid(R, lnm_x_, z_x_, sigma);
    std::copy(sigma.begin(), sigma.end(), out.begin());
    for (std::size_t j = 0; j != t.size(); ++j) {
      gl_op_->surface_density_grid(R * t[j] * t[j], lnm_x_, z_x_, sigma);
      std::copy(sigma.begin(), sigma.end(), out.begin() + (j + 1) * n_kq);
    }
  }

  // DeltaSigma = sum_j a_j B(R t_j^2) Sigma(R t_j^2) - B(R) Sigma(R), with
  // a_j = 4 w_j t_j^3 and B = 1 without bsel.  With bsel this is exactly
  // B DeltaSigma - M_sel / (pi R^2) (integration by parts).
  double
  evaluate_gauss_legendre(double const* w2, int b, double R, double lob) const
  {
    std::vector<double> local;
    double const* sig = nullptr;
    auto const it = std::find(wall_radii_.begin(), wall_radii_.end(), R);
    if (it != wall_radii_.end()) {
      sig = gl_sigma_[static_cast<std::size_t>(it - wall_radii_.begin())].data();
    } else {
      gl_sigma_slots(R, local);
      sig = local.data();
    }

    auto const& t = gl_op_->aperture_nodes();
    std::size_t const n_slot = t.size() + 1;
    std::vector<double> coef(n_slot * N_z_);
    for (std::size_t q = 0; q != N_z_; ++q) {
      auto B = [&](double r) {
        return bsel_op_ ? bsel_op_->model()(r, b, lob, z_x_[q]) : 1.0;
      };
      coef[q] = -B(R);
      for (std::size_t j = 0; j != t.size(); ++j)
        coef[(j + 1) * N_z_ + q] =
          gl_aperture_weights_[j] * B(R * t[j] * t[j]);
    }

    std::size_t const n_kq = N_lnm_ * N_z_;
    double acc = 0.0;
    for (std::size_t slot = 0; slot != n_slot; ++slot) {
      double const* srow = sig + slot * n_kq;
      double const* crow = &coef[slot * N_z_];
      for (std::size_t k = 0; k != N_lnm_; ++k)
        for (std::size_t q = 0; q != N_z_; ++q)
          acc += w2[k * N_z_ + q] * crow[q] * srow[k * N_z_ + q];
    }
    return acc;
  }

  // ---- miscentering -----------------------------------------------------
  //
  // y3_cluster::Miscentering (systematics/miscentering/Miscentering.hh)
  // does the integral and the stack; this module only describes its stack
  // (W2d, z nodes, Bsel model) and its centred profile Sigma(r | lnM_k, z_q):
  //   piecewise_approx: Sigma_max from the haloModel Sigma splines and gap
  //     terms from r_sigma[0], with the core below it the uniform disk
  //     m2d_core / (pi r_core^2);
  //   gauss_legendre: the GL projection (gl_op_) from MIS_R_LO, no core.
  void
  set_sample_miscentering(cosmosis::DataBlock& s)
  {
    using y3_cluster::Miscentering;
    bool const piecewise = (profile_method_ != "gauss_legendre");
    Miscentering::Stack const stack{
      n_bins_, N_lnm_, N_z_, w2d_.data(), &z_x_,
      bsel_op_ ? &bsel_op_->model() : nullptr};
    Miscentering::Profile profile;
    if (piecewise) {
      profile.r_lo = r_core_;
      profile.r_max = r_sigma_hi_;
      profile.core = sigma_core_kq_;
      profile.grid = [this](double r, std::vector<double>& out) {
        // Splines once per lnM node and once per z node, then the max model.
        std::vector<double> one(N_lnm_), two(N_z_);
        for (std::size_t k = 0; k != N_lnm_; ++k) one[k] = spl_nfw_[k](r);
        for (std::size_t q = 0; q != N_z_; ++q) two[q] = spl_hh_[q](r);
        for (std::size_t k = 0; k != N_lnm_; ++k)
          for (std::size_t q = 0; q != N_z_; ++q)
            out[k * N_z_ + q] = surface_density_value(r, one[k], two[q], k * N_z_ + q);
      };
      profile.at = [this](double r, std::size_t k, std::size_t q) {
        return surface_density_node(r, k, q, k * N_z_ + q);
      };
    } else {
      profile.r_lo = MIS_R_LO;
      profile.r_max = std::numeric_limits<double>::infinity();
      profile.grid = [this](double r, std::vector<double>& out) {
        gl_op_->surface_density_grid(r, lnm_x_, z_x_, out);
      };
    }
    mis_->set_sample(s, stack, profile);
  }

  // Piecewise Sigma_max(r) for a given (lnM_k, z_q), below the table the
  // uniform core disk:
  //   r <  r_core  : m2d_core / (pi r_core^2)
  //   r <  r_trans : Sigma_1h(r) + Delta_gap^0 + 2 Delta_gap^1 (1 - (r/r_trans)^2)
  //   r >= r_trans : b Sigma_hh(r)
  // Only built when miscentering is enabled.
  double surface_density_node(double r, std::size_t k, std::size_t q,
                              std::size_t idx) const
  {
    if (r < r_core_)
      return sigma_core_kq_[idx];
    return surface_density_value(r, spl_nfw_[k](r), spl_hh_[q](r), idx);
  }

  // Same for r >= r_core, given Sigma_1h(r | lnM_k) and Sigma_hh(r | z_q).
  double surface_density_value(double r, double sig_nfw, double sig_hh,
                               std::size_t idx) const
  {
    double const rt = r_trans_kq_[idx];
    if (r >= rt)
      return bias_kq_[idx] * sig_hh;
    double const r_ratio = r / rt;
    return sig_nfw + delta_gap_0_kq_[idx] +
           2.0 * delta_gap_1_kq_[idx] * (1.0 - r_ratio * r_ratio);
  }

  static y3_cluster::Interp2D
  make_sanitized_table(cosmosis::DataBlock& s, char const* key)
  {
    using doubles = std::vector<double>;
    auto const r = s.view<doubles>("haloModel", "r_sigma");
    auto const z = s.view<doubles>("haloModel", "z");
    auto const& nd =
      s.view<cosmosis::ndarray<double>>("haloModel", key);
    std::vector<double> vals(nd.begin(), nd.end());
    if (vals.size() != r.size() * z.size())
      throw std::runtime_error(std::string("Shear1h2hMax: ") + key + " extents mismatch");
    for (auto& v : vals)
      if (!std::isfinite(v)) v = 0.0;
    return y3_cluster::Interp2D(r, z, vals);
  }

  // Piecewise DeltaSigma_max(R) for a given (lnM_k, z_q):
  //   R <  r_trans : DeltaSigma_1h(R) + Delta_gap^1 (R / r_trans)^2
  //   R >= r_trans : b DeltaSigma_hh(R) + C_0 / R^2
  // one_k = DeltaSigma_1h(R | lnM_k), two_q = DeltaSigma_hh(R | z_q).
  double excess_surface_density_node(double R, double one_k, double two_q,
                                     std::size_t idx) const
  {
    double const rt = r_trans_kq_[idx];
    if (R >= rt)
      return bias_kq_[idx] * two_q + (c0_trans_kq_[idx] / (R * R)) * 1.0e-12;
    double const r_ratio = R / rt;
    return one_k + delta_gap_1_kq_[idx] * r_ratio * r_ratio;
  }

  // barSigma(<r) = Sigma(r) + DeltaSigma(r) of the same piecewise profile,
  // for a given (lnM_k, z_q); the mean_sigma_fn input to
  // BselDeltaSigmaOperator::apply.  Only built when bsel is enabled.
  //   r <  r_trans : barSigma_1h(r) + Delta_gap^0 + 2 Delta_gap^1
  //                  - Delta_gap^1 (r / r_trans)^2
  //   r >= r_trans : b barSigma_hh(r) + C_0 / r^2
  double mean_surface_density_node(double r, std::size_t k, std::size_t q,
                                   std::size_t idx) const
  {
    return mean_surface_density_value(r, sigma_bar_nfw_->clamp(r, lnm_x_[k]),
                                      sigma_bar_hh_->clamp(r, z_x_[q]), idx);
  }

  // Same, given barSigma_1h(r | lnM_k) and barSigma_hh(r | z_q).
  double mean_surface_density_value(double r, double sbar_nfw, double sbar_hh,
                                    std::size_t idx) const
  {
    double const rt = r_trans_kq_[idx];
    if (r >= rt)
      return bias_kq_[idx] * sbar_hh + (c0_trans_kq_[idx] / (r * r)) * 1.0e-12;
    double const gap1 = delta_gap_1_kq_[idx];
    double const r_ratio = r / rt;
    return sbar_nfw + delta_gap_0_kq_[idx] + 2.0 * gap1 - gap1 * r_ratio * r_ratio;
  }

  std::size_t N_lnm_, N_z_;
  double zt_lo_, zt_hi_, lnm_lo_, lnm_hi_;
  bool bsel_enabled_;
  std::string bsel_model_;
  std::string profile_method_;  // piecewise_approx | gauss_legendre
  std::size_t n_los_, n_aperture_;
  bool mis_enabled_;
  std::vector<double> lnm_x_, lnm_w_, z_x_, z_w_, lob_centers_;

  std::optional<y3_cluster::Interp2D> dsigma_nfw_, dsigma_hh_;
  std::optional<y3_cluster::Interp2D> sigma_bar_nfw_, sigma_bar_hh_;
  std::vector<y3_cluster::CubicSplineLnR> spl_nfw_, spl_hh_;  // miscentering
  std::optional<y3_cluster::BselDeltaSigmaOperator> bsel_op_;
  // Bsel factors at the apply() nodes r_j of one (bin, wall R).
  struct BselWall {
    std::vector<double> r, c;           // nodes, weights (BselDeltaSigmaOperator::nodes)
    std::vector<double> b_R;            // B_sel(R; z_q)
    std::vector<double> d_bsel;         // dB/dR(r_j; z_q),        (j, q)
    std::vector<double> sbar_hh;        // barSigma_hh(r_j | z_q), (j, q)
    std::vector<double> sbar_nfw;       // barSigma_1h(r_j | lnM_k), (j, k)
  };
  std::vector<BselWall> bsel_wall_;     // (bin, wall R)
  std::size_t n_bins_{0};
  std::vector<double> w2d_, bias_kq_;
  std::vector<double> r_trans_kq_, c0_trans_kq_, delta_gap_1_kq_, delta_gap_0_kq_;

  std::vector<double> wall_radii_, gl_aperture_weights_;
  std::vector<std::vector<double>> gl_sigma_;
  std::optional<y3_cluster::OperatorMax1h2h_gl> gl_op_;

  static constexpr double MIS_R_LO = 1.0e-4;  // h^-1 Mpc, GL profile grid floor
  std::optional<y3_cluster::Miscentering> mis_;
  double r_core_{0.0}, r_sigma_hi_{0.0};  // haloModel/r_sigma ends
  std::vector<double> sigma_core_kq_;     // m2d_core / (pi r_core^2)
};

#endif
