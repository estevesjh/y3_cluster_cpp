// bselmarggpu.cu -- Unified GPU module for Costanzi-2026 selection bias
//
// Computes P1, I1, J, b_eff and the marginalized asymptotes B_small, B_large.
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
#include "models/b_sel.cuh"  // For b_sel_asymptotes()

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

  // Compute b_eff via mass-weighted integral:
  //   b_eff = int dM n(M,z) P(ltr=lob|M,z) M b(M,z) / int dM n(M,z) P(ltr=lob|M,z) M
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

  // Compute B_small, B_large via marginalization over ltr
  // B_small = <b_zero>_{P(ltr|lob,zob)}
  // B_large = <b_infty>_{P(ltr|lob,zob)}
  //
  // Uses b_sel_asymptotes() from b_sel.cuh for each ltr value,
  // weighted by MOR probability.
  void compute_B_marginalised(
      double lob,
      double zob,
      double P1,
      double I1,
      double J,
      double b_eff,
      y3_cuda::MOR_SHIFTED_POISSON_t const& mor,
      double& B_small,
      double& B_large)
  {
    // Integration over ltr from 0 to lob
    // Use 50 points for the integral
    int const N_ltr = 50;
    double const ltr_min = 0.1;  // Avoid ltr=0 singularity
    double const ltr_max = lob;
    double const dltr = (ltr_max - ltr_min) / (N_ltr - 1);

    double sum_b_zero = 0.0;
    double sum_b_infty = 0.0;
    double sum_weight = 0.0;

    // Use a representative lnM for MOR evaluation
    // (MOR is evaluated at the mean mass for the richness bin)
    double const lnM_eff = std::log(1e14);  // Representative mass

    for (int i = 0; i < N_ltr; ++i) {
      double const ltr = ltr_min + i * dltr;

      // Compute b_zero, b_infty for this specific ltr using b_sel.cuh
      double b_zero, b_infty;
      y3_cuda::b_sel_asymptotes(lob, ltr, P1, I1, J, b_eff, b_zero, b_infty);

      // Weight by MOR probability P(ltr | M, z)
      // MOR(ltr, lnM, z) gives P(ltr | M, z)
      double const weight = mor(ltr, lnM_eff, zob);

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

    // Read HMF and bias tables
    auto mf_m = get_vector_double(sample, "mass_function", "m_h");
    auto mf_z = get_vector_double(sample, "mass_function", "z");
    auto mf_vals = get_vector_double(sample, "mass_function", "dndlnmh");

    auto hm_m = get_vector_double(sample, "haloModel", "m_h");
    auto hm_z = get_vector_double(sample, "haloModel", "z");
    auto hm_bias = get_vector_double(sample, "haloModel", "bias");

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
      double B_small, B_large;
      compute_B_marginalised(lob, zob, P1_vals[ig], I1_vals[ig], J_vals[ig],
                             b_eff_vals[ig], mor, B_small, B_large);
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
