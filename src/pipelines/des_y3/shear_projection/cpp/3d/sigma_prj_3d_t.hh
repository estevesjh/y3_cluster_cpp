// Projection shear -- full (theta_mis, z, lnM)-resolved reference, C++.
//
// Sibling of dsigma_prj_3d_t.hh::DSigmaPrj3d: identical integrand, the
// only change is the miscentering kernel -- Sigma_mis (surface
// density, y3_cluster::NFW_SIGMA_MIS) instead of DSigma_mis (excess
// surface density, y3_cluster::NFW_DSIGMA_MIS). See that header and
// review/08312026/shear_prj_3d_cpp_plan.md for the integrand formula,
// the three physics corrections (3D-ball exclusion, no w_pz, R_mis via
// the projected halo's own D_A(z)), and configuration keys -- all
// apply here unchanged:
//
//   Sigma_prj(R|lob,zob) = int dz int dtheta_mis  dV/dz/dOmega(z)
//       2*pi*sin(theta_mis) int dlnM  n(M,z) *
//       1[Delta_chi(theta_mis,z) >= R_excl] *
//       (1 + b(M,z) * b_sel(lob,zob,theta_mis) * xi_NL(Delta_chi, zbar)) *
//       Sigma_mis(R, R_mis | M)
//
// Sigma_mis is the SINGLE-kernel table (y3_cluster::NFW_SIGMA_MIS,
// NFW_SIG_SINGLE) -- same [cMpc/h] R/R_mis convention and Msun/h/pc^2
// output as NFW_DSIGMA_MIS (both documented in their respective
// headers). R_mis = theta_mis * chi(z): physically theta_mis * D_A(z)
// (the projected halo's own angular-diameter distance), converted to
// the comoving frame via *(1+z) at that same z; D_A(z)*(1+z)=chi(z)
// exactly, so the two steps collapse with no division anywhere
// (2026-08-31 owner decision; cur_R_ is left untouched, matching
// Shear1h3d's default q=1).
//
// Output: sigmaprj3d/{vals, errors, probs, status, nregions}.
// Status: new reference backend.
#ifndef Y3_CLUSTER_CPP_SIGMA_PRJ_3D_T_HH
#define Y3_CLUSTER_CPP_SIGMA_PRJ_3D_T_HH

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
#include "models/nfw_dsigma_mis.hh"   // y3_cluster::CONC / RHOC (shared, not redefined here)
#include "models/nfw_sigma_mis.hh"
#include "utils/interp_1d.hh"
#include "utils/interp_2d.hh"

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>

class SigmaPrj3d {
public:
  using grid_t = y3_cluster::grid_t<4>;
  using grid_point_t = grid_t::value_type;

private:
  using volume_t = cubacpp::IntegrationVolume<3>;

  bool use_halo_model_conc_{false};   // issue #14
  // Diagnostic (2026-08-31): set Sigma_mis=1 instead of calling the NFW
  // table, so the run computes the SAME integral's denominator --
  // int dz dtheta_mis dlnM n(M,z) [bracket] with no mass-dependent
  // profile amplitude. Dividing the normal (drop_sigma_mis_=false) run
  // by this one isolates <bias*bsel*xi_NL>, decoupled from Sigma_mis(M)'s
  // own mass-scaling, which otherwise compounds the mass-range
  // sensitivity already present in n(M)*b(M).
  bool drop_sigma_mis_{false};

  std::optional<y3_cluster::HMF_t> hmf_;
  std::optional<y3_cluster::DV_DO_DZ_t> dv_do_dz_;
  std::optional<y3_cluster::Interp1D> chi_;      // d_c(z), Mpc (x h0 in use)
  std::optional<y3_cluster::Interp2D> bias_;     // b(lnM, z)
  std::optional<y3_cluster::Interp2D> xi_nl_;    // xi_NL(r, z)
  std::optional<y3_cluster::NFW_SIGMA_MIS> sigma_mis_;   // 'single'
  std::optional<y3_cluster::sp_detail::BSelBins> bsel_;
  double h0_{0.7};

  // Per-wall-point scalars (set in set_grid_point).
  double zob_{0}, chi_o_{0}, r_excl_{0}, cur_R_{0};
  double b_small_{0}, b_large_{0}, k_sig_{0}, theta0_{0};

public:
  explicit SigmaPrj3d(cosmosis::DataBlock& cfg)
  {
    // issue #14 (opt-in): honor use_halo_model_conc (per-mass c(lnM) into
    // the miscentered NFW); default keeps the fixed-c production path.
    use_halo_model_conc_ =
        cfg.has_val(module_label(), "use_halo_model_conc") &&
        cfg.view<bool>(module_label(), "use_halo_model_conc");
    drop_sigma_mis_ =
        cfg.has_val(module_label(), "drop_sigma_mis") &&
        cfg.view<bool>(module_label(), "drop_sigma_mis");
  }

  void
  set_sample(cosmosis::DataBlock& s)
  {
    hmf_.emplace(s);
    dv_do_dz_.emplace(s);
    chi_.emplace(y3_cluster::make_Interp1D(s, "distances", "z", "d_c"));
    bias_.emplace(y3_cluster::make_Interp2D(s, "haloModel", "lnM", "z", "bias"));
    xi_nl_.emplace(y3_cluster::make_Interp2D(s, "xi_nl", "r", "z", "xi_nl"));
    sigma_mis_.emplace(y3_cluster::CONC, y3_cluster::RHOC, y3_cluster::NFW_SIG_SINGLE);
    // UNIFIED rho_m convention (2026-08-24): boundary AND amplitude on
    // haloModel/rho_m_ref; the old in-integrand omega_m multiply is gone.
    sigma_mis_->set_rho_ref(s.view<double>("haloModel", "rho_m_ref"));
    if (use_halo_model_conc_)
      sigma_mis_->set_concentration_table(
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

    // Clustered excess ONLY (drops the "1 +" random/mean-field term):
    // the mean-field piece is a fixed cylinder-geometry constant
    // (Omega_m*rho_crit*depth, ~96% of SIGMA_PRJ_of_R in the
    // SelectionBias mock -- MOCK_RECIPE.md Sec.9's own uniform-universe
    // check), sensitive to aperture/HMF-range choices that don't match
    // the mock's exact cylinder; the clustering excess is the actual
    // astrophysics (bias x xi_NL) this model is meant to validate.
    // 2026-08-31 owner decision -- deviates from Sigma_prj/DSigma_prj's
    // strict [1 + b*bsel*xi_NL] definition (matter_sigma_prj.tex,
    // richness_selection.tex); compare against the mock's OWN excess
    // (its SIGMA_PRJ_of_R minus that same uniform-universe prediction),
    // not its raw SIGMA_PRJ_of_R.
    double bracket = 0.0;
    if (dchi >= r_excl_) {   // outside the 3D exclusion ball
      double const bsel =
        b_small_ + (b_large_ - b_small_) /
                     (1.0 + std::exp(-k_sig_ * (theta_mis - theta0_)));
      bracket = bias_->clamp(lnM, z) * bsel * xi_nl_->clamp(dchi, zbar);
    }
    // dchi < r_excl_: bracket stays 0 -- whole integrand vanishes there.

    double const smis = drop_sigma_mis_ ? 1.0 : (*sigma_mis_)(cur_R_, R_mis, lnM);
    return theta_mis * 2.0 * M_PI * std::sin(theta_mis) * (*dv_do_dz_)(z) *
           (*hmf_)(lnM, z) * bracket * smis;   // no w_pz factor
  }

  static char const* module_label() { return "SigmaPrj3d"; }

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
