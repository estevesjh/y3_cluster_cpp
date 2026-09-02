// Projection shear -- full (theta_mis, z, lnM)-resolved reference, C++.
//
// CPU cuhre sibling of the CUDA/PAGANI diagnostic DSigmaPrj3dGpu (see
// ../../cuda/3d/dsigma_prj_3d_gpu_t.cuh), same fully-coupled adaptive
// integral, cubacpp::Cuhre instead of PAGANI:
//
//   DSigma_prj(R|lob,zob) = int dz int dtheta_mis  dV/dz/dOmega(z)
//       2*pi*sin(theta_mis) int dlnM  n(M,z) *
//       1[Delta_chi(theta_mis,z) >= R_excl] *
//       (1 + b(M,z) * b_sel(lob,zob,theta_mis) * xi_NL(Delta_chi, zbar)) *
//       DSigma_mis(R, R_mis | M)
//
// matching docs/matter_sigma_prj.tex's Sigma_prj definition (strict
// reference) with the halo-exclusion term made concrete, per
// review/08312026/shear_prj_3d_cpp_plan.md:
//
//   * exclusion is a genuine 3D ball: Delta_chi(theta_mis,z) is the
//     exact 3D comoving separation (law of cosines on chi(z), chi(zob),
//     theta_mis); inside R_excl = R_lambda(lob)*(1+zob) the WHOLE
//     bracket is zero (not just the clustered term, as DSigmaPrj3dGpu
//     does today).
//   * no w_pz factor -- the photo-z kernel belongs only inside the
//     P[X] operator that derives b_sel(theta_mis) itself
//     (RichnessSelection/docs/richness_selection.tex), never in
//     Sigma_prj directly.
//   * R_mis = theta_mis * chi(z) -- the projected halo's own comoving
//     distance. Derivation: R_mis is physically theta_mis * D_A(z)
//     (the projected halo's own angular-diameter distance), converted
//     to the comoving frame the NFW tables live in via *(1+z) at that
//     same z; D_A(z)*(1+z) = chi(z) exactly, so the two steps collapse
//     to theta_mis * chi(z), no division by (1+z) anywhere. Not
//     D_A(zob)/chi(zob) as DSigmaPrj3dGpu computes today (2026-08-31
//     owner decision; cur_R_ itself is left untouched, matching
//     Shear1h3d's default q=1 -- only its opt-in
//     `one_halo_physical_density` flag applies a (1+z) rescale).
//   * xi_NL's second argument is zbar = (z + zob)/2, not zob alone.
//
// theta_mis integrates in log (Jacobian theta_mis*dln(theta_mis)):
// DSigma_mis(R|M,R_mis) peaks sharply at R_mis ~ R, ~1e-5 wide in
// linear theta_mis for the smallest radii -- log-theta_mis makes that
// feature O(1) wide, same reason DSigmaPrj3dGpu uses it.
//
// b_sel(lob,zob,theta_mis) is read from the b_sel_marginalised wall
// via the exact-match y3_cluster::sp_detail::BSelBins reader (also
// gives lob for this row, so no separate lob_centers config is
// needed). DSigma_mis is the SINGLE-kernel miscentering table
// (y3_cluster::NFW_DSIGMA_MIS, "single" kernel) -- its R/R_mis
// arguments and r_s are documented [cMpc/h]; the physical D_A(z)
// convention above was confirmed against DSigmaPrj3dGpu/matter_sigma_prj.tex
// (2026-08-31 owner decision).
//
// Output: dsigmaprj3d/{vals, errors, probs, status, nregions}.
// Status: new reference backend, validate against DSigmaPrj3dGpu
// (expect agreement away from R_excl / z~zob only, by design -- see
// the plan doc's "Validation after it builds" section).
#ifndef Y3_CLUSTER_CPP_DSIGMA_PRJ_3D_T_HH
#define Y3_CLUSTER_CPP_DSIGMA_PRJ_3D_T_HH

#include "utils/datablock_reader.hh"
#include "utils/make_grid_points.hh"
#include "utils/make_integration_volumes.hh"
#include "utils/make_interp_1d.hh"
#include "utils/make_interp_2d.hh"

#include "cosmosis/datablock/datablock.hh"
#include "cubacpp/integration_volume.hh"

#include "models/bsel_bins_t.hh"
#include "models/dv_do_dz_t.hh"
#include "models/hmf_t.hh"
#include "models/nfw_dsigma_mis.hh"
#include "utils/interp_1d.hh"
#include "utils/interp_2d.hh"

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>

class DSigmaPrj3d {
public:
  using grid_t = y3_cluster::grid_t<4>;
  using grid_point_t = grid_t::value_type;

private:
  using volume_t = cubacpp::IntegrationVolume<3>;

  bool use_halo_model_conc_{false};   // issue #14

  std::optional<y3_cluster::HMF_t> hmf_;
  std::optional<y3_cluster::DV_DO_DZ_t> dv_do_dz_;
  std::optional<y3_cluster::Interp1D> chi_;      // d_c(z), Mpc (x h0 in use)
  std::optional<y3_cluster::Interp2D> bias_;     // b(lnM, z)
  std::optional<y3_cluster::Interp2D> xi_nl_;    // xi_NL(r, z)
  std::optional<y3_cluster::NFW_DSIGMA_MIS> dsigma_mis_;   // 'single'
  std::optional<y3_cluster::sp_detail::BSelBins> bsel_;
  double h0_{0.7};

  // Per-wall-point scalars (set in set_grid_point).
  double zob_{0}, chi_o_{0}, r_excl_{0}, cur_R_{0};
  double b_small_{0}, b_large_{0}, k_sig_{0}, theta0_{0};

public:
  explicit DSigmaPrj3d(cosmosis::DataBlock& cfg)
  {
    // issue #14 (opt-in): honor use_halo_model_conc (per-mass c(lnM) into
    // the miscentered NFW); default keeps the fixed-c production path.
    use_halo_model_conc_ =
        cfg.has_val(module_label(), "use_halo_model_conc") &&
        cfg.view<bool>(module_label(), "use_halo_model_conc");
  }

  void
  set_sample(cosmosis::DataBlock& s)
  {
    hmf_.emplace(s);
    dv_do_dz_.emplace(s);
    chi_.emplace(y3_cluster::make_Interp1D(s, "distances", "z", "d_c"));
    bias_.emplace(y3_cluster::make_Interp2D(s, "haloModel", "lnM", "z", "bias"));
    xi_nl_.emplace(y3_cluster::make_Interp2D(s, "xi_nl", "r", "z", "xi_nl"));
    dsigma_mis_.emplace(y3_cluster::CONC, y3_cluster::RHOC, y3_cluster::SINGLE);
    // UNIFIED rho_m convention (2026-08-24): boundary AND amplitude on
    // haloModel/rho_m_ref; the old in-integrand omega_m multiply is gone.
    dsigma_mis_->set_rho_ref(s.view<double>("haloModel", "rho_m_ref"));
    if (use_halo_model_conc_)
      dsigma_mis_->set_concentration_table(
        y3_cluster::make_Interp1D(s, "haloModel", "lnM", "concentration"));
    h0_ = s.view<double>("cosmological_parameters", "h0");
    bsel_.emplace(s);
  }

  void
  set_grid_point(grid_point_t const& pt)
  {
    int const lob_bin = static_cast<int>(pt[0]);
    zob_ = 0.5 * (pt[1] + pt[2]);
    cur_R_ = pt[3];
    chi_o_ = chi_->clamp(zob_) * h0_;

    auto const bv = bsel_->at(lob_bin, zob_);
    b_small_ = bv.b_small;
    b_large_ = bv.b_large;

    double const R_lambda = std::pow(bv.lob / 100.0, 0.2);   // cMpc/h
    r_excl_ = R_lambda * (1.0 + zob_);
    double const theta_lam = R_lambda * (1.0 + zob_) / chi_o_;
    k_sig_ = 2.5 / theta_lam;
    theta0_ = 0.5 * theta_lam;
  }

  double
  operator()(double lntheta_mis, double z, double lnM) const
  {
    double const theta_mis = std::exp(lntheta_mis);

    double const chi_z = chi_->clamp(z) * h0_;         // chi(z), cMpc/h
    double const dchi = std::sqrt(std::max(
      chi_z * chi_z + chi_o_ * chi_o_ -
        2.0 * chi_z * chi_o_ * std::cos(theta_mis), 0.0));
    // R_mis = theta_mis * D_A(z) * (1+z) [physical -> comoving, own z]
    //       = theta_mis * chi(z)/(1+z) * (1+z) = theta_mis * chi_z.
    double const R_mis = theta_mis * chi_z;
    double const zbar = 0.5 * (z + zob_);

    // Clustered excess ONLY (drops the "1 +" random/mean-field term) --
    // see sigma_prj_3d_t.hh for the rationale (2026-08-31 owner
    // decision, deviates from DSigma_prj's strict [1 + b*bsel*xi_NL]
    // definition; compare against the mock's own excess, not its raw
    // SIGMA_PRJ_of_R).
    double bracket = 0.0;
    if (dchi >= r_excl_) {   // outside the 3D exclusion ball
      double const bsel =
        b_small_ + (b_large_ - b_small_) /
                     (1.0 + std::exp(-k_sig_ * (theta_mis - theta0_)));
      bracket = bias_->clamp(lnM, z) * bsel * xi_nl_->clamp(dchi, zbar);
    }
    // dchi < r_excl_: bracket stays 0 -- whole integrand vanishes there,
    // both channels.

    double const dsmis = (*dsigma_mis_)(cur_R_, R_mis, lnM);
    return theta_mis * 2.0 * M_PI * std::sin(theta_mis) * (*dv_do_dz_)(z) *
           (*hmf_)(lnM, z) * bracket * dsmis;   // no w_pz factor
  }

  static char const* module_label() { return "DSigmaPrj3d"; }

  static std::vector<volume_t>
  make_integration_volumes(cosmosis::DataBlock& cfg)
  {
    return y3_cluster::make_integration_volumes_wall_of_numbers(
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
