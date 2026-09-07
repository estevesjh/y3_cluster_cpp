// bselmarggpu.cu -- Unified GPU module for Costanzi-2026 selection bias
// Computes P1, I1, J, b_eff and the marginalized asymptotes B_small, B_large.
// 
//
// Outputs to datablock:
//   b_sel_marg_P1/vals, errors, zo_low, zo_high, lambda_bin, b_eff
//   b_sel_marg_I1/vals, errors
//   b_sel_marg_J/vals,  errors
//   b_sel_marginalised/b_small, b_large, lob, zob  (for gamma_prj.cuh)

#include "cosmosis/datablock/datablock.hh"
#include "cosmosis/datablock/ndarray.hh"

#include "utils/make_grid_points.hh"
#include "utils/datablock_reader.hh"
#include "utils/gpu_integrator.cuh"
#include "utils/mpi_support.hh"

#include "models/p_operator_gpu_t.cuh"
#include "models/mor_shifted_poisson_t.cuh"
// BROKEN: do not use models/emg_des_t.cuh's EMG_DES_t from host code.
// Its members are quad::Interp1D, which store *device* pointers allocated
// via cuda_malloc()/cudaMemcpy (see common/cuda/Interp1D.cuh). EMG_DES_t is
// safe to call from inside a CUDA kernel (that's how numberCountsFull_t.cu
// and Shear1hMisSel.cu use it), but compute_B_marginalised() below runs on
// the host -- constructing EMG_DES_t there and calling .cdf() dereferences
// a device pointer on the CPU, which segfaults. See emg_cdf_host() further
// down for a plain-host reimplementation used instead.
// #include "models/emg_des_t.cuh"
#include "models/b_sel.cuh"  // all fucntions for computing b_sel asymptotes

#include <algorithm>
#include <iostream>
#include <optional>
#include <vector>
#include <cmath>

using cosmosis::DataBlock;

namespace {

  // Module configuration
  struct BSelMargGPUConfig {
    quad::Volume<double, 3> volume;
    y3_cluster::grid_t<3> grid_points;
    double eps_rel;
    double eps_abs;
    double max_eval;
    std::string algorithm;
    int device_id;
  };

  BSelMargGPUConfig
  read_config(DataBlock& cfg)
  {
    char const* label = "bselmarggpu";
    BSelMargGPUConfig c;

    // Integration bounds
    c.volume.lows[0]  = cfg.view<double>(label, "lt_low");
    c.volume.highs[0] = cfg.view<double>(label, "lt_high");
    c.volume.lows[1]  = cfg.view<double>(label, "zt_low");
    c.volume.highs[1] = cfg.view<double>(label, "zt_high");
    c.volume.lows[2]  = cfg.view<double>(label, "lnm_low");
    c.volume.highs[2] = cfg.view<double>(label, "lnm_high");

    // Grid points
    auto zo_low_vec  = get_vector_double(cfg, label, "zo_low");
    auto zo_high_vec = get_vector_double(cfg, label, "zo_high");
    auto bin_vec     = get_vector_double(cfg, label, "lambda_bin");

    c.grid_points.set_names({"zo_low", "zo_high", "lambda_bin"});
    for (size_t i = 0; i < zo_low_vec.size(); ++i) {
      c.grid_points.push_back({zo_low_vec[i], zo_high_vec[i], bin_vec[i]});
    }

    // Integration settings
    c.eps_rel   = cfg.view<double>(label, "eps_rel");
    c.eps_abs   = cfg.view<double>(label, "eps_abs");
    c.max_eval  = cfg.view<double>(label, "max_eval");
    c.algorithm = cfg.view<std::string>(label, "algorithm");

    // CUDA device
    auto info = y3_cluster::get_mpi_info();
    c.device_id = info.local_rank;

    return c;
  }

  // DES Y3 richness bin centres
  inline double lob_center(int bin)
  {
    double const centres[4] = {25.0, 37.5, 52.5, 130.0};
    return (bin >= 0 && bin < 4) ? centres[bin] : 25.0;
  }

  // DES Y3 richness bin edges, matching lob_center() above and the
  // lob_bin_low/lob_bin_high arrays used by numberCountsFull_t/Shear1hMisSel
  // (see shearTot_pipeline.ini): [20,30], [30,45], [45,60], [60,200].
  // Needed by compute_B_marginalised() to evaluate P(lob|ltr,z) over the
  // actual bin, not just at its centre.
  inline void lob_bin_edges(int bin, double& lo, double& hi)
  {
    double const edges_lo[4] = {20.0, 30.0, 45.0,  60.0};
    double const edges_hi[4] = {30.0, 45.0, 60.0, 200.0};
    int const b = (bin >= 0 && bin < 4) ? bin : 0;
    lo = edges_lo[b];
    hi = edges_hi[b];
  }

  // Compute b_eff via mass-weighted integral:
  // b_eff = int dM n(M,z) P(ltr=lob|M,z) M b(M,z) / int dM n(M,z) P(ltr=lob|M,z) M
  // Uses trapezoidal integration in ln(M)
  double compute_b_eff(
      double lob,
      double zob,
      std::vector<double> const& M_grid,
      std::vector<double> const& n_M,    // dn/dlnM at zob
      std::vector<double> const& b_M,    // bias at zob
      y3_cuda::MOR_SHIFTED_POISSON_t const& mor)
  {
    int const NM = M_grid.size();
    double num = 0.0, den = 0.0;

    for (int i = 0; i < NM - 1; ++i) {
      double const M = M_grid[i];
      double const lnM = std::log(M);
      double const dlnM = std::log(M_grid[i+1]) - std::log(M_grid[i]);

      // P(ltr=lob | M, z) - evaluate MOR at ltr=lob
      double const p_mor = mor(lob, lnM, zob);

      // Weight: n(M) * P(lob|M,z) * M  (n_M is dn/dlnM, so n*M*dlnM = dn/dlnM*dlnM)
      double const wt = n_M[i] * p_mor;

      // Trapezoidal integration in lnM
      num += wt * b_M[i] * dlnM;
      den += wt * dlnM;
    }

    return (den > 0.0) ? (num / den) : 1.0;
  }

  // Interpolate 2D array (z, M) to get 1D slice at fixed z
  void interp_to_z(
      std::vector<double> const& vals,  // shape (Nz, NM) flattened
      std::vector<double> const& z_axis,
      std::vector<double> const& m_axis,
      double z_target,
      std::vector<double> const& M_out,
      std::vector<double>& result)
  {
    int const Nz = z_axis.size();
    int const NM_in = m_axis.size();
    int const NM_out = M_out.size();
    result.resize(NM_out);

    // Find z bracket
    int iz = 0;
    for (int i = 1; i < Nz; ++i) {
      if (z_axis[i] > z_target) { iz = i - 1; break; }
      iz = i - 1;
    }
    iz = std::max(0, std::min(iz, Nz - 2));
    double const fz = (z_target - z_axis[iz]) / (z_axis[iz+1] - z_axis[iz] + 1e-30);

    // Interpolate each M point
    for (int k = 0; k < NM_out; ++k) {
      double const logM_t = std::log(M_out[k]);

      // Find M bracket (in log space)
      int im = 0;
      for (int i = 1; i < NM_in; ++i) {
        if (std::log(m_axis[i]) > logM_t) { im = i - 1; break; }
        im = i - 1;
      }
      im = std::max(0, std::min(im, NM_in - 2));
      double const fm = (logM_t - std::log(m_axis[im])) /
                        (std::log(m_axis[im+1]) - std::log(m_axis[im]) + 1e-30);

      // Bilinear interpolation
      double const v00 = vals[iz * NM_in + im];
      double const v01 = vals[iz * NM_in + im + 1];
      double const v10 = vals[(iz+1) * NM_in + im];
      double const v11 = vals[(iz+1) * NM_in + im + 1];

      result[k] = (1-fz) * ((1-fm)*v00 + fm*v01) + fz * ((1-fm)*v10 + fm*v11);
    }
  }

  // Host-side copy of the "plob_ltr_params" datablock section (written by
  // y3_buzzard/prj_params.py): the 8 EMG coefficients on a shared z grid.
  // Plain std::vector<double>, unlike EMG_DES_t's quad::Interp1D members,
  // so it's safe to read from host code -- see the comment on the removed
  // `#include "models/emg_des_t.cuh"` above for why that matters.
  struct PlobLtrParams {
    std::vector<double> z, a_mu, b_mu, a_sig, b_sig, a_tau, b_tau, a_fprj, b_fprj;
  };

  PlobLtrParams
  read_plob_ltr_params(DataBlock& sample)
  {
    PlobLtrParams p;
    p.z      = get_vector_double(sample, "plob_ltr_params", "z");
    p.a_mu   = get_vector_double(sample, "plob_ltr_params", "a_mu");
    p.b_mu   = get_vector_double(sample, "plob_ltr_params", "b_mu");
    p.a_sig  = get_vector_double(sample, "plob_ltr_params", "a_sig");
    p.b_sig  = get_vector_double(sample, "plob_ltr_params", "b_sig");
    p.a_tau  = get_vector_double(sample, "plob_ltr_params", "a_tau");
    p.b_tau  = get_vector_double(sample, "plob_ltr_params", "b_tau");
    p.a_fprj = get_vector_double(sample, "plob_ltr_params", "a_fprj");
    p.b_fprj = get_vector_double(sample, "plob_ltr_params", "b_fprj");
    return p;
  }

  // Plain linear interpolation on a host std::vector (x assumed sorted
  // ascending), clamped at the ends -- same behaviour as quad::Interp1D's
  // clamp()+eval(), just without touching device memory.
  double
  lerp_host(std::vector<double> const& x, std::vector<double> const& y, double xq)
  {
    int const n = static_cast<int>(x.size());
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

  // Host reimplementation of EMG_DES_t::cdf() (src/models/emg_des_t.cuh),
  // using lerp_host() in place of quad::Interp1D::clamp(). Uses the direct
  // exp(A)*Phi(...) tail form rather than emg_des_t.cuh's erfcx-stabilised
  // version: that stabilisation guards against exp(A) overflow when ltr can
  // range far above lob (as in numberCountsFull_t's K_i), but here ltr is
  // always <= lob by construction (see compute_B_marginalised's ltr loop),
  // which keeps A bounded and the direct form numerically safe.
  double
  emg_cdf_host(PlobLtrParams const& p, double lob, double ltr, double z)
  {
    double const ltr_safe = std::max(ltr, 0.5);

    double const a_mu_z   = lerp_host(p.z, p.a_mu,   z);
    double const b_mu_z   = lerp_host(p.z, p.b_mu,   z);
    double const a_sig_z  = lerp_host(p.z, p.a_sig,  z);
    double const b_sig_z  = lerp_host(p.z, p.b_sig,  z);
    double const a_tau_z  = lerp_host(p.z, p.a_tau,  z);
    double const b_tau_z  = lerp_host(p.z, p.b_tau,  z);
    double const a_fprj_z = lerp_host(p.z, p.a_fprj, z);
    double const b_fprj_z = lerp_host(p.z, p.b_fprj, z);

    double const mu = a_mu_z + b_mu_z * ltr_safe;
    double sigma = b_sig_z * std::pow(ltr_safe, a_sig_z);
    double tau   = b_tau_z / std::pow(ltr_safe, a_tau_z);
    double const denom = std::pow(1.0 + std::exp(-ltr_safe), a_fprj_z);
    double fprj = b_fprj_z / std::max(denom, 1e-300);
    sigma = std::max(sigma, 1e-8);
    tau   = std::max(tau,   1e-8);
    fprj  = std::max(0.0, std::min(1.0, fprj));

    double const z_std = (lob - mu) / sigma;
    double const gaussian_cdf = 0.5 * (1.0 + std::erf(z_std / std::sqrt(2.0)));

    double A = -tau * (lob - mu) + 0.5 * tau * tau * sigma * sigma;
    A = std::max(-700.0, std::min(700.0, A));
    double const tail =
        std::exp(A) * 0.5 * (1.0 + std::erf((z_std - tau * sigma) / std::sqrt(2.0)));

    double const result = gaussian_cdf - fprj * tail;
    return std::max(0.0, std::min(1.0, result));
  }

  // Compute B_small, B_large via marginalization over ltr
  // B_small = <b_zero>_{P(ltr|lob,zob)}
  // B_large = <b_infty>_{P(ltr|lob,zob)}
  //
  // Uses b_sel_asymptotes() from b_sel.cuh for each ltr value,
  // weighted by P(ltr|lob,zob).
  //
  // FIX (see notebook diagnosis, arXiv:2604.05833 comparison, 2026-07-07):
  // the previous implementation (kept below, commented out) weighted each
  // ltr sample by MOR(ltr | M=1e14, z) -- a single hardcoded representative
  // mass, independent of lob and never actually conditioned on lob at all
  // (beyond capping the ltr range at ltr<=lob). Since that MOR(ltr|M=1e14)
  // profile peaks near ltr~14 for every richness bin, higher-lob bins ended
  // up averaging almost entirely over delta_prj = (lob-ltr)/Delta_RND - 1
  // values far above 5 -- well outside the range the Costanzi-2026 linear
  // b_infty(delta_prj) = b_eff*(1+0.13*delta_prj) relation was calibrated
  // and tested on (arXiv:2604.05833 Figs. 6-7 only go up to delta_prj~5).
  // That is why B_large/b_eff came out at 1.7-3.2 instead of the paper's
  // ~0.8-2.0 -- for the lob=130 bin, *100%* of the marginalisation weight
  // fell at delta_prj>5.
  //
  // The fix builds the actual P(ltr|lob,zob) posterior instead:
  //   P(ltr|z)     = int dM n(M,z) MOR(ltr|M,z)             (mass-marginalised prior)
  //   P(lob|ltr,z) = EMG_CDF(lob_bin_high) - EMG_CDF(lob_bin_low)   (richness-projection kernel)
  //   P(ltr|lob,z) ~ P(ltr|z) * P(lob|ltr,z)
  // using the same M_grid/n_M tables already built for compute_b_eff(), and
  // the EMG_DES_t model (src/models/emg_des_t.cuh) that numberCountsFull_t
  // and Shear1hMisSel already use for the same P(lob|ltr,z) kernel.
  void compute_B_marginalised(
      double lob,
      double lob_bin_low,
      double lob_bin_high,
      double zob,
      double P1,
      double I1,
      double J,
      double b_eff,
      y3_cuda::MOR_SHIFTED_POISSON_t const& mor,
      PlobLtrParams const& plob,
      std::vector<double> const& M_grid,  // NM points, geometric 1e13..1e16
      std::vector<double> const& n_M,     // dn/dlnM at zob on M_grid
      double& B_small,
      double& B_large)
  {
    // Integration over ltr from 0 to lob
    // Use 50 points for the integral
    int const N_ltr = 50;
    double const ltr_min = 0.1;  // Avoid ltr=0 singularity
    double const ltr_max = lob;
    double const dltr = (ltr_max - ltr_min) / (N_ltr - 1);

    int const NM = M_grid.size();

    double sum_b_zero = 0.0;
    double sum_b_infty = 0.0;
    double sum_weight = 0.0;

    // ------------------------------------------------------------------
    // OLD (WRONG): single hardcoded representative mass, lob-independent.
    // double const lnM_eff = std::log(1e14);  // Representative mass
    // ------------------------------------------------------------------

    for (int i = 0; i < N_ltr; ++i) {
      double const ltr = ltr_min + i * dltr;

      // Compute b_zero, b_infty for this specific ltr using b_sel.cuh
      double b_zero, b_infty;
      y3_cuda::b_sel_asymptotes(lob, ltr, P1, I1, J, b_eff, b_zero, b_infty);

      // ----------------------------------------------------------------
      // OLD (WRONG): weight by MOR at one fixed mass -- not conditioned
      // on lob at all, so the ltr average never tracked lob.
      //
      // double const weight = mor(ltr, lnM_eff, zob);
      // ----------------------------------------------------------------

      // P(ltr|z): marginalise MOR(ltr|M,z) over the halo mass function,
      // trapezoidal in ln(M) (same pattern as compute_b_eff() above).
      double p_ltr_given_z = 0.0;
      for (int im = 0; im < NM - 1; ++im) {
        double const lnM_lo = std::log(M_grid[im]);
        double const lnM_hi = std::log(M_grid[im + 1]);
        double const dlnM = lnM_hi - lnM_lo;
        double const mor_lo = mor(ltr, lnM_lo, zob);
        double const mor_hi = mor(ltr, lnM_hi, zob);
        p_ltr_given_z += 0.5 * (n_M[im] * mor_lo + n_M[im + 1] * mor_hi) * dlnM;
      }

      // P(lob|ltr,z): probability that a halo with true richness ltr is
      // observed within this lob bin, from the same EMG kernel used as
      // K_i in numberCountsFull_t.cu / Shear1hMisSel.cu (host-side
      // reimplementation -- see emg_cdf_host() above for why).
      double const p_lob_given_ltr =
          emg_cdf_host(plob, lob_bin_high, ltr, zob) -
          emg_cdf_host(plob, lob_bin_low, ltr, zob);

      double const weight = p_ltr_given_z * std::max(0.0, p_lob_given_ltr);

      sum_b_zero += b_zero * weight;
      sum_b_infty += b_infty * weight;
      sum_weight += weight;
    }

    // Normalize
    if (sum_weight > 1e-30) {
      B_small = sum_b_zero / sum_weight;
      B_large = sum_b_infty / sum_weight;
    } else {
      // Fallback: no selection bias correction
      B_small = b_eff;
      B_large = b_eff;
    }
  }

}  // anonymous namespace


class BSelMargGPU {
public:
  explicit BSelMargGPU(DataBlock& cfg)
    : cfg_(read_config(cfg))
  {
    cudaSetDevice(cfg_.device_id);
  }

  void execute(DataBlock& sample)
  {
    cudaSetDevice(cfg_.device_id);

    size_t const ngrid = cfg_.grid_points.size();

    // Storage for results
    std::vector<double> P1_vals(ngrid), P1_errs(ngrid);
    std::vector<double> I1_vals(ngrid), I1_errs(ngrid);
    std::vector<double> J_vals(ngrid),  J_errs(ngrid);
    std::vector<double> b_eff_vals(ngrid);
    std::vector<double> B_small_vals(ngrid), B_large_vals(ngrid);
    std::vector<double> lob_vals(ngrid), zob_vals(ngrid);

    // Create integrator
    y3_cuda::MultiDimensionalIntegrator<3> integrator(cfg_.algorithm);
    integrator.set_maxeval(cfg_.max_eval);

    // Create integrands for each operator type
    // They share the same underlying models but use different weight functions
    y3_cuda::P_operator_gpu<y3_cuda::BSelMargP1GPUWeight> p1_integrand(sample);
    y3_cuda::P_operator_gpu<y3_cuda::BSelMargI1GPUWeight> i1_integrand(sample);
    y3_cuda::P_operator_gpu<y3_cuda::BSelMargJGPUWeight>  j_integrand(sample);

    // Initialize models from sample (done once, shared via sample)
    p1_integrand.set_sample(sample);
    i1_integrand.set_sample(sample);
    j_integrand.set_sample(sample);

    // -----------------------------------------------------------------
    // Setup for b_eff computation (CPU-side mass-weighted integral)
    // -----------------------------------------------------------------
    y3_cuda::MOR_SHIFTED_POISSON_t mor(sample);

    // P(lob|ltr,z) kernel for compute_B_marginalised(); reads
    // "plob_ltr_params", written by y3_buzzard/prj_params.py earlier in
    // the pipeline (see shearTot_pipeline.ini module order).
    // BROKEN -- segfaults, see the comment on the removed emg_des_t.cuh
    // include above (Interp1D holds device pointers, not host-safe):
    // y3_cuda::EMG_DES_t emg(sample);
    PlobLtrParams plob = read_plob_ltr_params(sample);

    // Read HMF and bias tables
    auto mf_m = get_vector_double(sample, "mass_function", "m_h");
    auto mf_z = get_vector_double(sample, "mass_function", "z");
    // dndlnmh is a 2D ndarray (Nz, NM) - read as ndarray and flatten to vector
    auto mf_ndarray = sample.view<cosmosis::ndarray<double>>("mass_function", "dndlnmh");
    std::vector<double> mf_vals(mf_ndarray.begin(), mf_ndarray.end());

    auto hm_m = get_vector_double(sample, "haloModel", "m_h");
    auto hm_z = get_vector_double(sample, "haloModel", "z");
    // bias is also a 2D ndarray (Nz, NM)
    auto bias_ndarray = sample.view<cosmosis::ndarray<double>>("haloModel", "bias");
    std::vector<double> hm_bias(bias_ndarray.begin(), bias_ndarray.end());

    // Build M grid for b_eff integration (100 points, 1e13 to 1e16)
    int const NM_beff = 100;
    double const M_min = 1e13;
    double const M_max = 1e16;
    std::vector<double> M_grid(NM_beff);
    for (int i = 0; i < NM_beff; ++i) {
      M_grid[i] = M_min * std::pow(M_max / M_min, double(i) / (NM_beff - 1));
    }

    // Integrate for each grid point
    for (size_t ig = 0; ig < ngrid; ++ig) {
      auto const& gp = cfg_.grid_points.points[ig];
      double const zob = 0.5 * (gp[0] + gp[1]);
      int const lob_bin = static_cast<int>(gp[2]);
      double const lob = lob_center(lob_bin);

      // P1
      p1_integrand.set_grid_point(gp);
      auto res_p1 = integrator.integrate(p1_integrand, cfg_.eps_abs, cfg_.eps_rel, cfg_.volume);
      P1_vals[ig] = res_p1.estimate;
      P1_errs[ig] = res_p1.errorest;

      // I1
      i1_integrand.set_grid_point(gp);
      auto res_i1 = integrator.integrate(i1_integrand, cfg_.eps_abs, cfg_.eps_rel, cfg_.volume);
      I1_vals[ig] = res_i1.estimate;
      I1_errs[ig] = res_i1.errorest;

      // J = I2 - I1 (computed directly)
      j_integrand.set_grid_point(gp);
      auto res_j = integrator.integrate(j_integrand, cfg_.eps_abs, cfg_.eps_rel, cfg_.volume);
      J_vals[ig] = res_j.estimate;
      J_errs[ig] = res_j.errorest;

      // Compute b_eff via mass-weighted integral
      std::vector<double> n_M_at_z, b_M_at_z;
      interp_to_z(mf_vals, mf_z, mf_m, zob, M_grid, n_M_at_z);
      interp_to_z(hm_bias, hm_z, hm_m, zob, M_grid, b_M_at_z);
      b_eff_vals[ig] = compute_b_eff(lob, zob, M_grid, n_M_at_z, b_M_at_z, mor);

      // Compute B_small, B_large via marginalization over ltr
      // Uses b_sel_asymptotes() from b_sel.cuh
      double lob_bin_lo, lob_bin_hi;
      lob_bin_edges(lob_bin, lob_bin_lo, lob_bin_hi);
      double B_small, B_large;
      // OLD (WRONG) call, kept for reference:
      // compute_B_marginalised(lob, zob, P1_vals[ig], I1_vals[ig], J_vals[ig],
      //                        b_eff_vals[ig], mor, B_small, B_large);
      compute_B_marginalised(lob, lob_bin_lo, lob_bin_hi, zob,
                             P1_vals[ig], I1_vals[ig], J_vals[ig],
                             b_eff_vals[ig], mor, plob, M_grid, n_M_at_z,
                             B_small, B_large);
      B_small_vals[ig] = B_small;
      B_large_vals[ig] = B_large;
      lob_vals[ig] = lob;
      zob_vals[ig] = zob;
    }

    // Write results to datablock - use same section names as CPU version
    // so bsel.py can consume the output unchanged
    using cosmosis::ndarray;

    // P1 section
    ndarray<double> P1_arr(P1_vals, {ngrid});
    ndarray<double> P1_err_arr(P1_errs, {ngrid});
    sample.put_val("b_sel_marg_P1", "vals", P1_arr);
    sample.put_val("b_sel_marg_P1", "errors", P1_err_arr);

    // I1 section
    ndarray<double> I1_arr(I1_vals, {ngrid});
    ndarray<double> I1_err_arr(I1_errs, {ngrid});
    sample.put_val("b_sel_marg_I1", "vals", I1_arr);
    sample.put_val("b_sel_marg_I1", "errors", I1_err_arr);

    // J section (= I2 - I1, computed directly)
    ndarray<double> J_arr(J_vals, {ngrid});
    ndarray<double> J_err_arr(J_errs, {ngrid});
    sample.put_val("b_sel_marg_J", "vals", J_arr);
    sample.put_val("b_sel_marg_J", "errors", J_err_arr);

    // Write grid points to P1 section (bsel.py reads from there)
    std::vector<double> zo_low_out(ngrid), zo_high_out(ngrid), bin_out(ngrid);
    for (size_t i = 0; i < ngrid; ++i) {
      zo_low_out[i]  = cfg_.grid_points.points[i][0];
      zo_high_out[i] = cfg_.grid_points.points[i][1];
      bin_out[i]     = cfg_.grid_points.points[i][2];
    }
    ndarray<double> zo_low_arr(zo_low_out, {ngrid});
    ndarray<double> zo_high_arr(zo_high_out, {ngrid});
    ndarray<double> bin_arr(bin_out, {ngrid});
    sample.put_val("b_sel_marg_P1", "zo_low", zo_low_arr);
    sample.put_val("b_sel_marg_P1", "zo_high", zo_high_arr);
    sample.put_val("b_sel_marg_P1", "lambda_bin", bin_arr);

    // b_eff (mass-weighted halo bias)
    ndarray<double> b_eff_arr(b_eff_vals, {ngrid});
    sample.put_val("b_sel_marg_P1", "b_eff", b_eff_arr);

    // Marginalized b_sel asymptotes for gamma_prj.cuh
    // Section: b_sel_marginalised (matches what sigma_prj expects)
    ndarray<double> B_small_arr(B_small_vals, {ngrid});
    ndarray<double> B_large_arr(B_large_vals, {ngrid});
    ndarray<double> lob_arr(lob_vals, {ngrid});
    ndarray<double> zob_arr(zob_vals, {ngrid});
    sample.put_val("b_sel_marginalised", "b_small", B_small_arr);
    sample.put_val("b_sel_marginalised", "b_large", B_large_arr);
    sample.put_val("b_sel_marginalised", "lob", lob_arr);
    sample.put_val("b_sel_marginalised", "zob", zob_arr);
  }

private:
  BSelMargGPUConfig cfg_;
};


// CosmoSIS interface
extern "C" {

void* setup(DataBlock* cfg)
{
  return new BSelMargGPU(*cfg);
}

DATABLOCK_STATUS execute(DataBlock* sample, void* module)
{
  auto mod = static_cast<BSelMargGPU*>(module);
  mod->execute(*sample);
  return DBS_SUCCESS;
}

int cleanup(void* module)
{
  delete static_cast<BSelMargGPU*>(module);
  return 0;
}

}  // extern "C"
