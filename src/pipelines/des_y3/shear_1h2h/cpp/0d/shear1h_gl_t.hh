// Miscentred one-halo shear via the S_ij-tabulated fixed-GL path (formerly fast_mass) — C++ backend.
//
// The exact-z-contraction fast path stated as a des_y3 module: the
// same algorithm as the production Shear1hMisSel.so (`method = exact`,
// Shear1hMisSelGL in the immutable n_operator_sel_gl_t.hh) — SelGlWeights
// builds W_ij(lnM) with Sigma_crit_inv folded in, then the 1-D GL mass
// sum over the production miscentred mixture d_tot — re-expressed here
// with its own configuration label and the namespace output section
// (shear1h_gl/vals, shared with the Python backend; the two are
// interchangeable and must never run in one pipeline: DataBlock
// sections do not overwrite). Every composed piece is an existing
// immutable model or pipeline-local helper: SelGlWeights, the
// shared/lensing_helpers.hh functions, the haloModel dSigma_nfw
// Interp2D and NFW_DSIGMA_MIS.
//
// f_mis and tau_mis are REQUIRED datablock values (miscentering/f_mis,
// miscentering/tau_mis): set_sample throws if the section is missing —
// no silent fallback to the Y3 fiducial defaults.
//
// Options: bin_index x r_perp cartesian grid (bin slow / R fast),
// lob_centers (default 25 37.5 52.5 130), required zt_low/zt_high/
// lnm_low/lnm_high, n_lnm (96), n_z (64).
// Status: reference re-expression; production remains Shear1hMisSel.so.
#ifndef Y3_CLUSTER_CPP_SHEAR1H_GL_T_HH
#define Y3_CLUSTER_CPP_SHEAR1H_GL_T_HH

#include "pipelines/shared/sel_gl_weights.hh"
#include "models/nfw_dsigma_mis.hh"
#include "pipelines/shared/lensing_helpers.hh"
#include "utils/interp_2d.hh"
#include "utils/make_grid_points.hh"
#include "utils/make_interp_2d.hh"

#include <array>
#include <optional>
#include <stdexcept>
#include <vector>

class Shear1hGl {
public:
  using grid_t = y3_cluster::grid_t<2>;
  using grid_point_t = grid_t::value_type;
  static constexpr std::size_t n_outputs = 1;

  explicit Shear1hGl(cosmosis::DataBlock& cfg)
    : core_(cfg, module_label())
    , dsigma_mis_(y3_cluster::CONC, y3_cluster::RHOC, y3_cluster::GAMMA)
  {
    lob_centers_ =
      y3_pipelines::read_lob_centers(cfg, module_label());
    if (lob_centers_.empty())
      throw std::runtime_error("Shear1hGl: lob_centers is empty");
  }

  void
  set_sample(cosmosis::DataBlock& s)
  {
    namespace w = y3_pipelines;
    core_.build_weights(s, /*include_sci=*/true);
    dsigma_nfw_.emplace(y3_cluster::make_Interp2D(
      s, "haloModel", "r_sigma", "lnM", "dSigma_nfw"));
    // Required: no fallback to the fiducial defaults — a pipeline that
    // has not published the miscentering section must fail loudly.
    f_mis_ = s.view<double>("miscentering", "f_mis");
    double const tau_mis = s.view<double>("miscentering", "tau_mis");
    dsigma_mis_.set_rho_mult(
      s.view<double>("cosmological_parameters", "omega_M"));
    r_mis_.assign(core_.n_bins(), 0.0);
    for (std::size_t b = 0; b != core_.n_bins(); ++b)
      r_mis_[b] = tau_mis * w::R_lambda(
                              lob_centers_[b % lob_centers_.size()]);
  }

  std::array<double, n_outputs>
  evaluate(grid_point_t const& pt) const
  {
    int const b = static_cast<int>(pt[0]);
    double const R = pt[1];
    if (b < 0 || static_cast<std::size_t>(b) >= core_.n_bins())
      throw std::out_of_range("Shear1hGl: bin_index");
    auto const& wb = core_.weights(b);
    auto const& xs = core_.lnm_x();
    auto const& ws = core_.lnm_w();
    double acc = 0.0;
    for (std::size_t k = 0; k != xs.size(); ++k) {
      double const d_cen = dsigma_nfw_->clamp(R, xs[k]);
      double const d_mis = dsigma_mis_(R, r_mis_[b], xs[k]);
      double const d_tot = (1.0 - f_mis_) * d_cen + f_mis_ * d_mis;
      acc += ws[k] * wb[k] * d_tot;
    }
    return {acc};
  }

  static char const* module_label() { return "Shear1hGl"; }

  static std::array<char const*, n_outputs>
  output_sections()
  {
    return {"shear1h_gl"};
  }

  static grid_t
  make_grid_points(cosmosis::DataBlock& cfg)
  {
    return y3_cluster::make_grid_points_cartesian_product(
      cfg, module_label(), "bin_index", "r_perp");
  }

private:
  y3_pipelines::SelGlWeights core_;
  std::vector<double> lob_centers_;
  std::optional<y3_cluster::Interp2D> dsigma_nfw_;
  // TODO(#14): drop the hardcoded c=4 — once claude/issue-4-dsigma-hh-2h-term
  // merges, call dsigma_mis_.set_concentration_table(
  //   make_Interp1D(s, "haloModel", "lnM", "concentration"))
  // in set_sample so the miscentred term uses the same per-mass Child18
  // concentration as the centred dSigma_nfw table.
  y3_cluster::NFW_DSIGMA_MIS dsigma_mis_;
  double f_mis_{0.0};
  std::vector<double> r_mis_;
};

#endif
