// bSelTest.cu - CosmoSIS module to compute and output b_sel(theta | lob, ltr, zob)
//
// This module reads P1, I1, J from the datablock (from P_operator modules),
// computes b_eff via mass integration, and outputs b_sel(theta) on a theta
// grid for plotting.
//
// Datablock reads:
//   b_sel_marg_P1/vals, b_sel_marg_I1/vals, b_sel_marg_J/vals
//   mass_function/*, haloModel/bias, cluster_mor/*, distances/*
//
// Datablock writes:
//   b_sel_test/{theta, theta_over_theta_lob, b_sel, b_zero, b_infty, b_eff}
//
// Grid: (lob, ltr, zob) triplets specified in ini file

#include "cosmosis/datablock/datablock.hh"
#include "cosmosis/datablock/datablock_status.h"

#include "models/b_sel.cuh"
#include "models/mor_shifted_poisson_t.cuh"
#include "utils/datablock_reader.hh"

#include <cmath>
#include <iostream>
#include <vector>

using cosmosis::DataBlock;

namespace {

// Module configuration
struct Config {
  std::vector<double> lob;    // observed richness values
  std::vector<double> ltr;    // true richness values
  std::vector<double> zob;    // observed redshift values
  int n_theta;
  double theta_lo;
  double theta_hi;
  bool verbose;
};

// Compute b_eff via mass integration:
//   b_eff = int dM n(M,z) P(ltr=lob|M,z) M b(M,z) / int dM n(M,z) P(ltr=lob|M,z) M
double compute_b_eff(
    double lob_val, double zob_val,
    std::vector<double> const& M_grid,
    std::vector<double> const& n_M,   // dn/dM at zob
    std::vector<double> const& b_M,   // bias at zob
    y3_cuda::MOR_SHIFTED_POISSON_t const& mor)
{
  int const NM = M_grid.size();
  double num = 0.0, den = 0.0;

  for (int i = 0; i < NM - 1; ++i) {
    double const M = M_grid[i];
    double const lnM = std::log(M);
    double const dlnM = std::log(M_grid[i+1]) - std::log(M_grid[i]);

    // P(ltr=lob | M, z) - use lob as ltr for b_eff computation
    double const p_mor = mor(lob_val, lnM, zob_val);

    // Weight: n(M) * P(lob|M,z) * M
    double const wt = n_M[i] * p_mor * M;

    // Trapezoidal integration in lnM
    num += wt * b_M[i] * dlnM;
    den += wt * dlnM;
  }

  return (den > 0.0) ? (num / den) : 1.0;
}

// Interpolate 2D array (z, M) to get 1D array at fixed z
void interp_to_z(
    std::vector<double> const& vals,  // shape (Nz, NM)
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

    // Find M bracket
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

} // anonymous namespace


extern "C" void* setup(DataBlock* options) {
  auto* cfg = new Config;

  cfg->lob = get_vector_double(*options, "bSelTest", "lob");
  cfg->ltr = get_vector_double(*options, "bSelTest", "ltr");
  cfg->zob = get_vector_double(*options, "bSelTest", "zob");

  cfg->n_theta = options->view<int>("bSelTest", "n_theta");
  cfg->theta_lo = options->view<double>("bSelTest", "theta_lo");
  cfg->theta_hi = options->view<double>("bSelTest", "theta_hi");

  cfg->verbose = options->has_val("bSelTest", "verbose")
                 ? options->view<bool>("bSelTest", "verbose") : false;

  if (cfg->verbose) {
    std::cout << "[bSelTest] Setup: " << cfg->lob.size() << " lob values, "
              << cfg->ltr.size() << " ltr values, "
              << cfg->zob.size() << " zob values, "
              << cfg->n_theta << " theta points" << std::endl;
  }

  return cfg;
}


extern "C" int execute(DataBlock* block, void* config) {
  auto* cfg = static_cast<Config*>(config);

  // Read P1, I1, J from datablock
  auto P1_vec = block->view<std::vector<double>>("b_sel_marg_P1", "vals");
  auto I1_vec = block->view<std::vector<double>>("b_sel_marg_I1", "vals");
  auto J_vec  = block->view<std::vector<double>>("b_sel_marg_J", "vals");

  // Read grid info to find which P1/I1/J corresponds to which (zob, lob_bin)
  auto zo_low_vec  = get_vector_double(*block, "b_sel_marg_P1", "zo_low");
  auto zo_high_vec = get_vector_double(*block, "b_sel_marg_P1", "zo_high");
  auto lam_bin_vec = get_vector_double(*block, "b_sel_marg_P1", "lambda_bin");

  // Read cosmology
  double const h0 = block->view<double>("cosmological_parameters", "h0");
  double const omm = block->view<double>("cosmological_parameters", "omega_m");
  double omn = 0.0;
  if (block->has_val("cosmological_parameters", "omega_nu")) {
    omn = block->view<double>("cosmological_parameters", "omega_nu");
  }

  // Read distances for chi(z)
  auto z_dist = block->view<std::vector<double>>("distances", "z");
  auto chi_dist = block->view<std::vector<double>>("distances", "d_c");

  // Read MOR parameters
  y3_cuda::MOR_SHIFTED_POISSON_t mor(*block);

  // Read HMF and bias for b_eff computation
  auto mf_m = block->view<std::vector<double>>("mass_function", "m_h");
  auto mf_z = block->view<std::vector<double>>("mass_function", "z");
  auto mf_vals_nd = block->view<std::vector<double>>("mass_function", "dndlnmh");

  auto hm_m = block->view<std::vector<double>>("haloModel", "m_h");
  auto hm_z = block->view<std::vector<double>>("haloModel", "z");
  auto hm_bias_nd = block->view<std::vector<double>>("haloModel", "bias");

  // Rescale HMF mass axis by (omm - omn) to match CPU convention
  std::vector<double> mf_m_scaled(mf_m.size());
  for (size_t i = 0; i < mf_m.size(); ++i) {
    mf_m_scaled[i] = mf_m[i] * (omm - omn);
  }

  // Build M grid for b_eff integration
  int const NM_beff = 100;
  double const M_min = 1e13;
  double const M_max = 1e15;
  std::vector<double> M_grid(NM_beff);
  for (int i = 0; i < NM_beff; ++i) {
    M_grid[i] = M_min * std::pow(M_max / M_min, double(i) / (NM_beff - 1));
  }

  // Build theta grid
  std::vector<double> theta_arr(cfg->n_theta);
  double const log_lo = std::log(cfg->theta_lo);
  double const log_hi = std::log(cfg->theta_hi);
  for (int i = 0; i < cfg->n_theta; ++i) {
    theta_arr[i] = std::exp(log_lo + i * (log_hi - log_lo) / (cfg->n_theta - 1));
  }

  // Output arrays
  int const N_out = cfg->lob.size();
  std::vector<double> out_theta_over_theta_lob(N_out * cfg->n_theta);
  std::vector<double> out_b_sel(N_out * cfg->n_theta);
  std::vector<double> out_b_zero(N_out);
  std::vector<double> out_b_infty(N_out);
  std::vector<double> out_b_eff(N_out);

  // For each (lob, ltr, zob) triplet
  for (int idx = 0; idx < N_out; ++idx) {
    double const lob = cfg->lob[idx];
    double const ltr = cfg->ltr[idx];
    double const zob = cfg->zob[idx];

    // Find P1, I1, J for this (zob, lob_bin)
    // For now, use simple lookup - assume grid matches
    double P1 = 0.0, I1 = 0.0, J = 0.0;
    int const lob_bin = (lob < 30) ? 0 : (lob < 45) ? 1 : (lob < 60) ? 2 : 3;
    for (size_t g = 0; g < zo_low_vec.size(); ++g) {
      double const zo_mid = 0.5 * (zo_low_vec[g] + zo_high_vec[g]);
      int const bin_g = static_cast<int>(lam_bin_vec[g]);
      if (bin_g == lob_bin && std::abs(zo_mid - zob) < 0.05) {
        P1 = P1_vec[g];
        I1 = I1_vec[g];
        J = J_vec[g];
        break;
      }
    }

    // Interpolate chi(zob)
    double chi_o = 0.0;
    for (size_t i = 1; i < z_dist.size(); ++i) {
      if (z_dist[i] > zob) {
        double const f = (zob - z_dist[i-1]) / (z_dist[i] - z_dist[i-1]);
        chi_o = (chi_dist[i-1] + f * (chi_dist[i] - chi_dist[i-1])) * h0;
        break;
      }
    }

    // Compute b_eff via mass integration
    std::vector<double> n_M_at_z, b_M_at_z;
    interp_to_z(mf_vals_nd, mf_z, mf_m_scaled, zob, M_grid, n_M_at_z);
    interp_to_z(hm_bias_nd, hm_z, hm_m, zob, M_grid, b_M_at_z);

    // Convert dndlnM to dndM
    for (int i = 0; i < NM_beff; ++i) {
      n_M_at_z[i] /= M_grid[i];
    }

    double const b_eff = compute_b_eff(lob, zob, M_grid, n_M_at_z, b_M_at_z, mor);
    out_b_eff[idx] = b_eff;

    // Compute b_zero and b_infty
    double b_zero, b_infty;
    y3_cuda::b_sel_asymptotes(lob, ltr, P1, I1, J, b_eff, b_zero, b_infty);
    out_b_zero[idx] = b_zero;
    out_b_infty[idx] = b_infty;

    // Compute theta_lob
    double const theta_lob_val = y3_cuda::theta_lob(lob, zob, chi_o);

    // Compute b_sel(theta) for each theta
    for (int it = 0; it < cfg->n_theta; ++it) {
      double const theta = theta_arr[it];
      out_theta_over_theta_lob[idx * cfg->n_theta + it] = theta / theta_lob_val;
      out_b_sel[idx * cfg->n_theta + it] =
          y3_cuda::b_sel_from_asymptotes(theta, b_zero, b_infty, theta_lob_val);
    }

    if (cfg->verbose) {
      std::cout << "[bSelTest] lob=" << lob << " ltr=" << ltr << " zob=" << zob
                << " P1=" << P1 << " I1=" << I1 << " J=" << J
                << " b_eff=" << b_eff << " b_zero=" << b_zero
                << " b_infty=" << b_infty << std::endl;
    }
  }

  // Write outputs
  block->put_val("b_sel_test", "theta", theta_arr);
  block->put_val("b_sel_test", "lob", cfg->lob);
  block->put_val("b_sel_test", "ltr", cfg->ltr);
  block->put_val("b_sel_test", "zob", cfg->zob);
  block->put_val("b_sel_test", "theta_over_theta_lob", out_theta_over_theta_lob);
  block->put_val("b_sel_test", "b_sel", out_b_sel);
  block->put_val("b_sel_test", "b_zero", out_b_zero);
  block->put_val("b_sel_test", "b_infty", out_b_infty);
  block->put_val("b_sel_test", "b_eff", out_b_eff);

  return 0;
}


extern "C" int cleanup(void* config) {
  delete static_cast<Config*>(config);
  return 0;
}
