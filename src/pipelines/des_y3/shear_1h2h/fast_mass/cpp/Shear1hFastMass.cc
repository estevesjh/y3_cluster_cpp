// Miscentred one-halo shear via the fast_mass strategy — C++ backend.
//
// The exact-z-contraction fast path stated as a des_y3 module: the
// same algorithm as the production Shear1hMisSel.so (`method = exact`,
// Shear1hMisSelGL in the immutable n_operator_sel_gl_t.hh) — SelGLCore
// builds W_ij(lnM) with Sigma_crit_inv folded in, then the 1-D GL mass
// sum over the production miscentred mixture — re-expressed here with
// its own configuration label and the namespace output section
// (shear1h_fast_mass/vals, shared with the Python backend; the two are
// interchangeable and must never run in one pipeline: DataBlock
// sections do not overwrite). Every composed piece is an existing
// immutable model: SelGLCore, the lensing_weights mis helpers, the
// haloModel dSigma_nfw Interp2D and NFW_DSIGMA_MIS.
//
// Options: bin_index x r_perp cartesian grid (bin slow / R fast),
// lob_centers (default 25 37.5 52.5 130), required zt_low/zt_high/
// lnm_low/lnm_high, n_lnm (96), n_z (64).
// Status: reference re-expression; production remains Shear1hMisSel.so.
#include "models/n_operator_sel_gl_t.hh"
#include "utils/module_macros.hh"

#include <array>
#include <optional>
#include <stdexcept>
#include <vector>

class Shear1hFastMassCpp {
public:
  using grid_t = y3_cluster::grid_t<2>;
  using grid_point_t = grid_t::value_type;
  static constexpr std::size_t n_outputs = 1;

  explicit Shear1hFastMassCpp(cosmosis::DataBlock& cfg)
    : core_(cfg, module_label())
    , dsigma_mis_(4.0, 2.77533742639e+11, y3_cluster::GAMMA)
  {
    lob_centers_ =
      y3_cluster_sel_weights::mis_detail::read_lob_centers(cfg,
                                                           module_label());
    if (lob_centers_.empty())
      throw std::runtime_error("Shear1hFastMassCpp: lob_centers is empty");
  }

  void
  set_sample(cosmosis::DataBlock& s)
  {
    namespace w = y3_cluster_sel_weights;
    core_.build_weights(s, /*include_sci=*/true);
    dsigma_nfw_.emplace(y3_cluster::make_Interp2D(
      s, "haloModel", "r_sigma", "lnM", "dSigma_nfw"));
    f_mis_ = w::mis_detail::read_mis_param(s, "f_mis",
                                           w::mis_detail::F_MIS_DEFAULT);
    double const tau_mis = w::mis_detail::read_mis_param(
      s, "tau_mis", w::mis_detail::TAU_MIS_DEFAULT);
    dsigma_mis_.set_rho_mult(
      s.view<double>("cosmological_parameters", "omega_M"));
    r_mis_.assign(core_.n_bins(), 0.0);
    for (std::size_t b = 0; b != core_.n_bins(); ++b)
      r_mis_[b] = tau_mis * w::mis_detail::R_lambda(
                              lob_centers_[b % lob_centers_.size()]);
  }

  std::array<double, n_outputs>
  evaluate(grid_point_t const& pt) const
  {
    int const b = static_cast<int>(pt[0]);
    double const R = pt[1];
    if (b < 0 || static_cast<std::size_t>(b) >= core_.n_bins())
      throw std::out_of_range("Shear1hFastMassCpp: bin_index");
    auto const& wb = core_.weights(b);
    auto const& xs = core_.lnm_x();
    auto const& ws = core_.lnm_w();
    double acc = 0.0;
    for (std::size_t k = 0; k != xs.size(); ++k) {
      double const d_cen = dsigma_nfw_->clamp(R, xs[k]);
      double const d_mis = dsigma_mis_(R, r_mis_[b], xs[k]);
      acc += ws[k] * wb[k] *
             ((1.0 - f_mis_) * d_cen + f_mis_ * d_mis);
    }
    return {acc};
  }

  static char const* module_label() { return "Shear1hFastMass"; }

  static std::array<char const*, n_outputs>
  output_sections()
  {
    return {"shear1h_fast_mass"};
  }

  static grid_t
  make_grid_points(cosmosis::DataBlock& cfg)
  {
    return y3_cluster::make_grid_points_cartesian_product(
      cfg, module_label(), "bin_index", "r_perp");
  }

private:
  y3_cluster::nosel_gl_detail::SelGLCore core_;
  std::vector<double> lob_centers_;
  std::optional<y3_cluster::Interp2D> dsigma_nfw_;
  y3_cluster::NFW_DSIGMA_MIS dsigma_mis_;
  double f_mis_{y3_cluster_sel_weights::mis_detail::F_MIS_DEFAULT};
  std::vector<double> r_mis_;
};

DEFINE_COSMOSIS_SCALAR_EVALUATOR_MODULE(Shear1hFastMassCpp)
