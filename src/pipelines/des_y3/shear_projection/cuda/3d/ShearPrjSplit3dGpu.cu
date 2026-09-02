// ShearPrjSplit3dGpu.cu -- three-region PAGANI split of Arwa Qadi's GPU
// projection integral (proposal: "split Arwa's GPU projection integral
// into three PAGANI regions", J. Esteves 2026-08-28).
//
// Her integrand -- y3_cuda::SigmaPrjGPU<DSigmaTotalWeight>, defined in
// shear_prj_module/sigma_prj_gpu_t.cuh, exactly the type
// ShearPrjEvaluator_t.cu drives with ONE adaptive (theta,z,lnM) box per
// wall row -- is used here UNCHANGED. Not one line of that class is
// touched. This file only supplies a different DRIVER: instead of one
// PAGANI call per row over the full z in [zt_low,zt_high], it makes
// THREE calls per row (foreground / core / background), split exactly
// at the two exclusion-ring boundaries (where her own theta_excl_at_z
// hits zero), and sums the three results:
//     I = I_fg + I_core + I_bg
//     err = sqrt(err_fg^2 + err_core^2 + err_bg^2)
// The rationale: halo exclusion carves a ~1e-3-wide feature (in z) into
// the middle of a ~0.7-wide integration box. PAGANI must locate that
// feature by bisection before it can resolve it, and every subdivision
// spent locating it is spent at full 3-D cost. Putting the boundaries
// where we already know them removes the search, not the physics.
//
// This driver follows the SAME hand-rolled setup/execute/cleanup shape
// as bSelMargGPU.cu (src/pipelines/systematics/selection_bias/cuda/3d/),
// which is the precedent for a multi-integrate-then-combine CUDA module
// in this codebase -- DEFINE_COSMOSIS_CUDA_INTEGRATION_MODULE gives
// exactly one integral per grid point and cannot express this.
//
// Output: shear_prj_split/{vals,errors} = the summed (fg+core+bg) value
// and quadrature-summed error, directly comparable to her shear_prj/vals
// (same integrand, same wall) and to our dsigma_prj_gl/vals (see the
// proposal's Comparison A). Per-region {vals,errors}_{fg,core,bg} are
// also published for diagnosis (which region dominates cost/error; see
// the proposal's Sec. 4 validation plan).

#include "cosmosis/datablock/datablock.hh"
#include "cosmosis/datablock/ndarray.hh"

#include "utils/datablock_reader.hh"
#include "utils/gpu_integrator.cuh"
#include "utils/make_grid_points.hh"
#include "utils/mpi_support.hh"

#include "modules/gpu_prj_costanzi2026/shear_prj_module/sigma_prj_gpu_t.cuh"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using cosmosis::DataBlock;

namespace {

  char const* const LABEL = "shear_prj_split";

  struct ShearPrjSplitConfig {
    double theta_low = 0.0, theta_high = 0.0;
    double lnm_low = 0.0, lnm_high = 0.0;
    // Outer fg/bg edges. Proposal Sec 3.2: default to her [shear_prj]
    // zt_low/zt_high, but read as OUR OWN ini keys (own section,
    // "shear_prj_split") so they can be widened/narrowed later without a
    // rebuild -- e.g. to the photo-z support (+/-sigma_z) instead of the
    // full [0.1,0.8] box, per the proposal's open question 2 (not decided
    // here; left at her bounds as the stated default).
    double z_lo = 0.0, z_hi = 0.0;
    y3_cluster::grid_t<4> grid_points;  // (lambda_bin, zo_low, zo_high, R)
    // Proposal Sec 3.3: ONE shared eps_rel/eps_abs/max_eval/algorithm for
    // all three regions in this first version -- per-region overrides are
    // explicitly left open, not built here. Reading each straight from a
    // single named key (rather than e.g. an array of ints) keeps a later
    // "per-region budget" change a small diff: just add
    // eps_rel_core/eps_rel_fg/... keys and branch in read_config.
    double eps_rel = 0.0, eps_abs = 0.0, max_eval = 0.0;
    std::string algorithm;
    int device_id = 0;
  };

  ShearPrjSplitConfig
  read_config(DataBlock& cfg)
  {
    ShearPrjSplitConfig c;

    c.theta_low  = cfg.view<double>(LABEL, "theta_low");
    c.theta_high = cfg.view<double>(LABEL, "theta_high");
    c.z_lo       = cfg.view<double>(LABEL, "zt_low");
    c.z_hi       = cfg.view<double>(LABEL, "zt_high");
    c.lnm_low    = cfg.view<double>(LABEL, "lnm_low");
    c.lnm_high   = cfg.view<double>(LABEL, "lnm_high");

    auto lambda_bin_vec = get_vector_double(cfg, LABEL, "lambda_bin");
    auto zo_low_vec     = get_vector_double(cfg, LABEL, "zo_low");
    auto zo_high_vec    = get_vector_double(cfg, LABEL, "zo_high");
    auto radii_vec      = get_vector_double(cfg, LABEL, "radii");

    c.grid_points.set_names({"lambda_bin", "zo_low", "zo_high", "radii"});
    for (std::size_t i = 0; i < radii_vec.size(); ++i) {
      c.grid_points.push_back(
        {lambda_bin_vec[i], zo_low_vec[i], zo_high_vec[i], radii_vec[i]});
    }

    c.eps_rel   = cfg.view<double>(LABEL, "eps_rel");
    c.eps_abs   = cfg.view<double>(LABEL, "eps_abs");
    c.max_eval  = cfg.view<double>(LABEL, "max_eval");
    c.algorithm = cfg.view<std::string>(LABEL, "algorithm");

    auto info = y3_cluster::get_mpi_info();
    c.device_id = info.local_rank;
    return c;
  }

  // Plain host linear interpolation on a std::vector -- same pattern as
  // bSelMargGPU.cu's lerp_host (file-local there, so copied rather than
  // shared). No device memory involved, safe to call from execute().
  double
  lerp_host(std::vector<double> const& x, std::vector<double> const& y,
            double xq)
  {
    int const n = static_cast<int>(x.size());
    if (n == 0) return 0.0;
    if (n == 1) return y[0];
    if (xq <= x[0]) return y[0];
    if (xq >= x[n - 1]) return y[n - 1];
    int i = 0;
    for (int k = 1; k < n; ++k) {
      if (x[k] > xq) { i = k - 1; break; }
      i = k - 1;
    }
    double const f = (xq - x[i]) / (x[i + 1] - x[i] + 1e-30);
    return y[i] + f * (y[i + 1] - y[i]);
  }

  // Host-side copy of distances/{z,d_c}. Her SigmaPrjGPU keeps its own
  // device-resident quad::Interp1D over the SAME datablock section as a
  // private member (chi_, no accessor) -- this is a separate, host-only
  // read of the identical section, not a competing model or a change to
  // her class.
  struct ChiTable {
    std::vector<double> z, d_c;
    double
    operator()(double zq) const
    {
      return lerp_host(z, d_c, zq);
    }
  };

  ChiTable
  read_chi_table(DataBlock& sample)
  {
    ChiTable t;
    t.z   = get_vector_double(sample, "distances", "z");
    t.d_c = get_vector_double(sample, "distances", "d_c");
    return t;
  }

  // Bisect for the z where chi(z) = chi(z_ob) + sign*R_excl, i.e. exactly
  // where her theta_excl_at_z(chi(z), chi(z_ob), R_excl) crosses zero
  // (sigma_prj_gpu_t.cuh) -- the exclusion-ring boundary. chi_of(z) =
  // chi_table(z) * h0 matches her chi_.clamp(z)*h0_ convention exactly.
  // chi(z) is monotonic increasing over any physically sane [z_lo,z_hi],
  // so plain bisection suffices.
  //
  // Degenerate case (proposal Sec 3.4): if the target chi is outside the
  // achievable range on [z_lo,z_hi] -- e.g. R_excl so large, or z_ob so
  // close to an edge, that z_ob +/- delta_z_excl falls outside the box --
  // there is no interior root; clamp to the edge. The caller's
  // integrate_region then sees a zero-width or inverted sub-interval and
  // returns (0,0) for it rather than integrating, so the total still
  // equals a plain single-region integral over whatever of [z_lo,z_hi]
  // remains -- it degrades, it does not throw or misbehave.
  double
  solve_chi_boundary(ChiTable const& chi_table, double h0, double chi_o,
                     double R_excl, double sign, double z_lo, double z_hi)
  {
    double const target = chi_o + sign * R_excl;
    auto const chi_of = [&](double z) { return chi_table(z) * h0; };

    double lo = z_lo, hi = z_hi;
    double const f_lo = chi_of(lo) - target;
    double f_hi = chi_of(hi) - target;
    if (f_lo >= 0.0) return z_lo;
    if (f_hi <= 0.0) return z_hi;

    for (int iter = 0; iter < 60; ++iter) {
      double const mid = 0.5 * (lo + hi);
      double const f_mid = chi_of(mid) - target;
      if (f_mid == 0.0) return mid;
      if ((f_mid > 0.0) == (f_hi > 0.0)) {
        hi = mid;
        f_hi = f_mid;
      } else {
        lo = mid;
      }
    }
    return 0.5 * (lo + hi);
  }

  // One PAGANI call over [theta_low,theta_high] x [z_a,z_b] x
  // [lnm_low,lnm_high]. Degenerate (z_b <= z_a) sub-intervals -- the
  // proposal's Sec 3.4 requirement -- contribute exactly (0,0), never an
  // integrator call on an inverted or zero-width volume.
  template <class Integrand>
  numint::integration_result
  integrate_region(y3_cuda::MultiDimensionalIntegrator<3>& integrator,
                   Integrand& integrand, double theta_low, double theta_high,
                   double z_a, double z_b, double lnm_low, double lnm_high,
                   double eps_abs, double eps_rel)
  {
    numint::integration_result zero{};
    zero.estimate = 0.0;
    zero.errorest = 0.0;
    if (!(z_b > z_a)) return zero;

    quad::Volume<double, 3> vol;
    vol.lows[0] = theta_low;
    vol.highs[0] = theta_high;
    vol.lows[1] = z_a;
    vol.highs[1] = z_b;
    vol.lows[2] = lnm_low;
    vol.highs[2] = lnm_high;
    return integrator.integrate(integrand, eps_abs, eps_rel, vol);
  }

} // namespace

class ShearPrjSplit3dGpu {
public:
  explicit ShearPrjSplit3dGpu(DataBlock& cfg) : cfg_(read_config(cfg))
  {
    cudaSetDevice(cfg_.device_id);
  }

  void
  execute(DataBlock& sample)
  {
    cudaSetDevice(cfg_.device_id);

    std::size_t const ngrid = cfg_.grid_points.size();

    std::vector<double> vals_fg(ngrid), errs_fg(ngrid);
    std::vector<double> vals_core(ngrid), errs_core(ngrid);
    std::vector<double> vals_bg(ngrid), errs_bg(ngrid);
    std::vector<double> vals(ngrid), errs(ngrid);

    y3_cuda::MultiDimensionalIntegrator<3> integrator(cfg_.algorithm);
    integrator.set_maxeval(cfg_.max_eval);

    // Same integrand type her single-box driver uses
    // (ShearPrjEvaluator_t.cu): SigmaPrjGPU<DSigmaTotalWeight>, i.e.
    // shear_prj/vals = (1 + b*b_sel*xi_NL) * DSigma_mis, unchanged.
    using Integrand = y3_cuda::SigmaPrjGPU<y3_cuda::DSigmaTotalWeight>;
    Integrand integrand(sample);
    integrand.set_sample(sample);

    ChiTable const chi_table = read_chi_table(sample);
    double const h0 = sample.view<double>("cosmological_parameters", "h0");

    for (std::size_t ig = 0; ig != ngrid; ++ig) {
      auto const& gp = cfg_.grid_points.points[ig];

      Integrand::grid_point_t pt;
      pt[0] = gp[0];
      pt[1] = gp[1];
      pt[2] = gp[2];
      pt[3] = gp[3];
      integrand.set_grid_point(pt);

      int const lob_bin = static_cast<int>(gp[0]);
      double const zo_low = gp[1];
      double const zo_high = gp[2];
      double const zob = 0.5 * (zo_low + zo_high);

      // Same R_excl(lob,zob) her operator() computes internally -- her
      // own free functions, not reimplemented.
      double const lob_center =
        y3_cuda::sigma_prj_gpu_detail::lob_center_from_bin(lob_bin);
      double const R_excl =
        y3_cuda::sigma_prj_gpu_detail::R_lambda(lob_center) * (1.0 + zob);

      double const chi_o = chi_table(zob) * h0;

      double const z_core_lo = solve_chi_boundary(
        chi_table, h0, chi_o, R_excl, -1.0, cfg_.z_lo, zob);
      double const z_core_hi = solve_chi_boundary(
        chi_table, h0, chi_o, R_excl, +1.0, zob, cfg_.z_hi);

      auto const res_fg = integrate_region(
        integrator, integrand, cfg_.theta_low, cfg_.theta_high, cfg_.z_lo,
        z_core_lo, cfg_.lnm_low, cfg_.lnm_high, cfg_.eps_abs, cfg_.eps_rel);
      auto const res_core = integrate_region(
        integrator, integrand, cfg_.theta_low, cfg_.theta_high, z_core_lo,
        z_core_hi, cfg_.lnm_low, cfg_.lnm_high, cfg_.eps_abs, cfg_.eps_rel);
      auto const res_bg = integrate_region(
        integrator, integrand, cfg_.theta_low, cfg_.theta_high, z_core_hi,
        cfg_.z_hi, cfg_.lnm_low, cfg_.lnm_high, cfg_.eps_abs, cfg_.eps_rel);

      vals_fg[ig] = res_fg.estimate;
      errs_fg[ig] = res_fg.errorest;
      vals_core[ig] = res_core.estimate;
      errs_core[ig] = res_core.errorest;
      vals_bg[ig] = res_bg.estimate;
      errs_bg[ig] = res_bg.errorest;

      vals[ig] = res_fg.estimate + res_core.estimate + res_bg.estimate;
      errs[ig] = std::sqrt(res_fg.errorest * res_fg.errorest +
                           res_core.errorest * res_core.errorest +
                           res_bg.errorest * res_bg.errorest);
    }

    using cosmosis::ndarray;
    sample.put_val(LABEL, "vals", ndarray<double>(vals, {ngrid}));
    sample.put_val(LABEL, "errors", ndarray<double>(errs, {ngrid}));
    sample.put_val(LABEL, "vals_fg", ndarray<double>(vals_fg, {ngrid}));
    sample.put_val(LABEL, "errors_fg", ndarray<double>(errs_fg, {ngrid}));
    sample.put_val(LABEL, "vals_core", ndarray<double>(vals_core, {ngrid}));
    sample.put_val(LABEL, "errors_core", ndarray<double>(errs_core, {ngrid}));
    sample.put_val(LABEL, "vals_bg", ndarray<double>(vals_bg, {ngrid}));
    sample.put_val(LABEL, "errors_bg", ndarray<double>(errs_bg, {ngrid}));
  }

private:
  ShearPrjSplitConfig cfg_;
};

// CosmoSIS interface
extern "C" {

void*
setup(DataBlock* cfg)
{
  return new ShearPrjSplit3dGpu(*cfg);
}

DATABLOCK_STATUS
execute(DataBlock* sample, void* module)
{
  auto mod = static_cast<ShearPrjSplit3dGpu*>(module);
  mod->execute(*sample);
  return DBS_SUCCESS;
}

int
cleanup(void* module)
{
  delete static_cast<ShearPrjSplit3dGpu*>(module);
  return 0;
}

} // extern "C"
