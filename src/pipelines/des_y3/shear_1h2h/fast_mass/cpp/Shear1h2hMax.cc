// Traditional 1h+2h shear via the max model — C++ backend.
//
//   Phi_max(R, lnM, z | bin) = max( DSigma_cl(R, lnM | bin),
//                                   b(lnM, z) * DSigma_hh(R, z) )
//
// the Y1-era SIG_MAX/KAPPA_MAX/GAMMA_MAX composition on the modern
// haloModel tables (same observable as the Python
// ../python/shear1h2h_max.py), stacked with the selection weight
//
//   O_ij(R) = int dz int dlnM  n dV/dOmegadz Omega Sigma_crit^-1
//             S_ij(lnM, z) Phi_max(R, lnM, z).
//
// Structure note: the two-halo term is z-dependent, so — unlike
// Shear1hFastMass, which reuses SelGLCore's z-contracted weight — the
// redshift integral must stay inside the mass integral. This driver
// therefore builds the z-RESOLVED weight W2d(bin; lnM, z) on the same
// fixed GL nodes (same term composition as SelGLCore, just without the
// z sum) and contracts (lnM, z) per (bin, R).
//
// Every table is read through the project's interpolation primitives
// (Interp2D / Interp1D with clamped queries): haloModel dSigma_nfw,
// bias and dSigma_hh, average_sigma_crit_inv, the selection tensor via
// SelFunction_t, and the miscentred NFW look-up via NFW_DSIGMA_MIS.
//
// dSigma_hh carries NaN over ~60% of its (R, z) table by construction
// (see docs/known_issues/dsigma_hh_debug_flag.md); the values are sanitized to 0
// BEFORE being handed to Interp2D, which is exact for a max model
// (max(1h, 0) = 1h where the 2h term is undefined) and keeps NaN out
// of the interpolator's stencil. Requires compute_lensing_2h = T.
//
// Options: bin_index x r_perp cartesian grid (bin slow / R fast),
// zt_low/zt_high/lnm_low/lnm_high (required), n_lnm (96), n_z (64),
// lob_centers, include_miscentering (default T).
// Output: shear1h2h_max/vals — same section as the Python backend
// (interchangeable; never run both in one pipeline).
// Status: reference implementation of the traditional-shear arm.
#include "cosmosis/datablock/datablock.hh"
#include "cosmosis/datablock/ndarray.hh"

#include "models/dv_do_dz_t.hh"
#include "models/hmf_t.hh"
#include "models/n_operator_sel_gl_t.hh"
#include "models/nfw_dsigma_mis.hh"
#include "models/omega_z_des.hh"
#include "models/p_operator_cuhre_t.hh"
#include "models/sel_function_t.hh"
#include "modules/num_counts_sel/lensing_weights.hh"
#include "utils/datablock_reader.hh"
#include "utils/interp_1d.hh"
#include "utils/interp_2d.hh"
#include "utils/make_grid_points.hh"
#include "utils/make_interp_2d.hh"
#include "utils/module_macros.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>
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
    , dsigma_mis_(4.0, 2.77533742639e+11, y3_cluster::GAMMA)
  {
    y3_cluster::p_op_detail::gl_nodes(lnm_lo_, lnm_hi_, N_lnm_, lnm_x_, lnm_w_);
    y3_cluster::p_op_detail::gl_nodes(zt_lo_, zt_hi_, N_z_, z_x_, z_w_);
    lob_centers_ =
      y3_cluster_sel_weights::mis_detail::read_lob_centers(cfg,
                                                           module_label());
    if (lob_centers_.empty())
      throw std::runtime_error("Shear1h2hMax: lob_centers is empty");
  }

  void
  set_sample(cosmosis::DataBlock& s)
  {
    namespace w = y3_cluster_sel_weights;

    y3_cluster::HMF_t const hmf(s);
    y3_cluster::DV_DO_DZ_t const dv(s);
    y3_cluster::OMEGA_Z_DES const omega(s);
    auto const sci = w::load_sigma_crit_inv(s);

    dsigma_nfw_.emplace(y3_cluster::make_Interp2D(
      s, "haloModel", "r_sigma", "lnM", "dSigma_nfw"));
    bias_.emplace(
      y3_cluster::make_Interp2D(s, "haloModel", "lnM", "z", "bias"));
    dsigma_hh_.emplace(make_sanitized_hh(s));

    f_mis_ = include_mis_
               ? w::mis_detail::read_mis_param(
                   s, "f_mis", w::mis_detail::F_MIS_DEFAULT)
               : 0.0;
    double const tau_mis = w::mis_detail::read_mis_param(
      s, "tau_mis", w::mis_detail::TAU_MIS_DEFAULT);
    dsigma_mis_.set_rho_mult(
      s.view<double>("cosmological_parameters", "omega_M"));

    // z-only factors (Sigma_crit_inv folded in, as the shear weight
    // requires) and the z-RESOLVED weight W2d(bin; lnM, z).
    std::vector<double> zfac(z_x_.size());
    for (std::size_t q = 0; q != z_x_.size(); ++q)
      zfac[q] = z_w_[q] * dv(z_x_[q]) * omega(z_x_[q]) * sci.clamp(z_x_[q]);

    int const n_bins = y3_cluster::nosel_detail::n_bins_from_block(s);
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
      r_mis_[b] = tau_mis * w::mis_detail::R_lambda(
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
    for (std::size_t k = 0; k != N_lnm_; ++k) {
      double const d_cen = dsigma_nfw_->clamp(R, lnm_x_[k]);
      double const d_mis = dsigma_mis_(R, r_mis_[b], lnm_x_[k]);
      one[k] = (1.0 - f_mis_) * d_cen + f_mis_ * d_mis;
    }
    for (std::size_t q = 0; q != N_z_; ++q)
      two[q] = dsigma_hh_->clamp(R, z_x_[q]);

    double acc = 0.0;
    double const* w2 = &w2d_[static_cast<std::size_t>(b) * N_lnm_ * N_z_];
    for (std::size_t k = 0; k != N_lnm_; ++k) {
      double const* wrow = w2 + k * N_z_;
      double const* brow = &bias_kq_[k * N_z_];
      double const one_k = one[k];
      for (std::size_t q = 0; q != N_z_; ++q)
        acc += wrow[q] * std::max(one_k, brow[q] * two[q]);
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
  // replaced by 0 first (docs/known_issues/dsigma_hh_debug_flag.md). The datablock
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
  double zt_lo_, zt_hi_, lnm_lo_, lnm_hi_;
  bool include_mis_;
  y3_cluster::NFW_DSIGMA_MIS dsigma_mis_;
  std::vector<double> lnm_x_, lnm_w_, z_x_, z_w_, lob_centers_;

  std::optional<y3_cluster::Interp2D> dsigma_nfw_, bias_, dsigma_hh_;
  double f_mis_{y3_cluster_sel_weights::mis_detail::F_MIS_DEFAULT};
  std::size_t n_bins_{0};
  std::vector<double> w2d_, bias_kq_, r_mis_;
};

DEFINE_COSMOSIS_SCALAR_EVALUATOR_MODULE(Shear1h2hMax)
