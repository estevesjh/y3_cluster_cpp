// GPU 1h+2h max-model shear — full (lambda_true, lnM, z) fixed-GL reference
//
// CPU, before the first kernel:
//   - GL nodes and weights for lnM, z and lambda_true
//   - bin edges from the ini
//   - models and tables from the datablock
//   - copy inputs to the GPU
//
// GPU (explicit_sel_weights kernel): richness selection at every (lnM, z) node
//   sel[b,k,q] = sum_i w_i P_HOD(lt_i | M_k, z_q)
//                * [F(lam_max | lt_i, z_q) - F(lam_min | lt_i, z_q)]
//   - lambda_true window per (lnM, z)
//   - P_HOD from MOR_HOD_t
//   - F from EMG_DES_t::cdf
//
// CPU, between the two kernels:
//   - copy sel back from the GPU
//   - weight
//     W2d[b,k,q] = lnm_w_k z_w_q n(M_k,z_q) dV/dOmegadz(z_q) Omega(z_q)
//                  * Sigma_crit^-1(z_q) S_j(z_q) sel[b,k,q]
//   - table values at the (R, lnM, z) nodes
//   - Bsel factors, when bsel = T (build_bsel_wall)
//   - copy inputs to the GPU
//
// GPU (max_model_contract kernel): centred shear for every (bin, R)
//   O_cen[b,R] = sum_kq W2d[b,k,q] DSigma_tot(R, lnM_k, z_q)
//   - max model, piecewise around r_trans:
//       R <  r_trans: DSigma_1h + Delta_gap^1 (R / r_trans)^2
//       R >= r_trans: b DSigma_hh + (C_0 / R^2) 1e-12
//     falls back to max(1h, 2h) when the r_trans tables are absent
//   - bsel = T, same model as the CPU Shear1h2hMax:
//       DSigma_tot = B_sel(R) DSigma_max
//                    - R sum_j c_j barSigma(<r_j) dB_sel/dR(r_j)
//     on the BselDeltaSigmaOperator nodes r_j
//
// CPU, after the kernels:
//   - copy O_cen back from the GPU
//   - miscentering = T, same model as the CPU Shear1h2hMax (apply_miscentering):
//       O = (1 - f_mis) O_cen + f_mis O_mis
//     the whole centred max-model profile is translated by a gamma-distributed
//     offset and mixed after the stack; O_mis comes from y3_cluster::Miscentering
//     and is also Bsel-weighted when bsel = T
//
// Needs to run models:
//   - bsel_model (Costanzi26 | Sunayama23) when bsel = T
//   - f_mis and tau_mis from [miscentering] when miscentering = T
//   - the r_trans tables, for both bsel and miscentering
//   - haloModel Sigma_bar_nfw, Sigma_bar_hh and delta_gap_0 for bsel
//   - haloModel Sigma_nfw, Sigma_hh, m2d_core and delta_gap_0 for miscentering
//     plus the halo_model core segment (R_perp_core)
//
// Note: Bsel is split: its factors are tabulated on the CPU and summed on the GPU. Miscentering runs only on the CPU.
// Not implemented: one_halo_physical_density (throws).
// Output: shear1h2h_max_explicit_gl_gpu/vals.

#ifndef Y3_CLUSTER_CPP_SHEAR1H2H_MAX_EXPLICIT_GL_GPU_CUH
#define Y3_CLUSTER_CPP_SHEAR1H2H_MAX_EXPLICIT_GL_GPU_CUH

#include "cosmosis/datablock/datablock.hh"
#include "cosmosis/datablock/ndarray.hh"

#include "models/dv_do_dz_t.hh"
#include "models/emg_des_t.cuh"
#include "models/hmf_t.hh"
#include "models/mor_hod_t.cuh"
#include "pipelines/shared/sel_gl_weights.hh" // for gl_nodes only
#include "models/omega_z_des.hh"
#include "pipelines/shared/lensing_helpers.hh"
#include "pipelines/systematics/miscentering/Miscentering.hh"
#include "utils/datablock_reader.hh"
#include "utils/interp_1d.hh"
#include "utils/interp_2d.hh"
#include "utils/make_grid_points.hh"
#include "utils/make_interp_2d.hh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace y3_cuda_des_y3 {

  inline void
  cuda_check(cudaError_t rc, char const* what)
  {
    if (rc != cudaSuccess)
      throw std::runtime_error(std::string("Shear1h2hMaxExplicitGlGpu: ") + what +
                               ": " + cudaGetErrorString(rc));
  }

  // Richness selection at every fixed-GL node
  // One thread per (lnM, z) node. For each node: the lambda_true window
  // [mu_eff - L sig_eff, mu_eff + L sig_eff] (cut at 0), n_q GL nodes in
  // it, and for every bin
  // sel[b,k,q] = sum_i  w_i P_HOD(lt_i | M, z) [CDF(lam_max) - CDF(lam_min)]
  // P_HOD is MOR_HOD_t, the EMG CDF is EMG_DES_t.
  // Identical to the kernel in NumCountsExplicitGlGpu.cuh.
  static constexpr int MAX_BINS = 32;

  inline __global__ void
  explicit_sel_weights(y3_cuda::MOR_HOD_t mor, y3_cuda::EMG_DES_t emg,
                       int n_bins, int n_lnm, int n_z, int n_q, double l_lam,
                       double const* lnm_x, double const* z_x,
                       double const* gl_t, double const* gl_w,
                       double const* lam_min, double const* lam_max,
                       double* sel)
  {
    int const tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_lnm * n_z) return;
    int const q = tid % n_z;
    int const k = tid / n_z;
    double const lnM = lnm_x[k];
    double const z = z_x[q];

    // lambda_true window from the HOD mean and variance at (M, z).
    double const M = exp(lnM);
    double const Mmin = pow(10.0, mor.log10_Mmin());
    double const dM1 = pow(10.0, mor.log10_M1()) - Mmin;
    double const lcentral = (M >= Mmin) ? 1.0 : 0.0;
    double const dM = fmax(M - Mmin, 0.0);
    double mu_sat = 0.0;
    if (dM1 > 0.0 && dM > 0.0)
      mu_sat = pow(dM / dM1, mor.alpha()) *
               pow((1.0 + z) / (1.0 + mor.z_pivot()), mor.epsilon());
    double const delta = (mor.sigma_lambda() * mu_sat) *
                         (mor.sigma_lambda() * mu_sat);
    double const nu = mu_sat + delta;
    double const mu_eff = lcentral + mu_sat;
    double const sig_eff = nu > 0.0 ? sqrt(nu) : 0.0;
    double const a_lo = fmax(mu_eff - l_lam * sig_eff, 0.0);
    double const b_hi = mu_eff + l_lam * sig_eff;

    double acc[MAX_BINS];
    for (int b = 0; b != n_bins; ++b) acc[b] = 0.0;

    if (b_hi > a_lo) {   // degenerate window -> all weights stay 0
      double const half = 0.5 * (b_hi - a_lo);
      double const mid = 0.5 * (a_lo + b_hi);
      for (int i = 0; i != n_q; ++i) {
        double const lt = mid + half * gl_t[i];
        double const p = mor(lt, lnM, z);
        if (p == 0.0) continue;
        double const wp = half * gl_w[i] * p;
        for (int b = 0; b != n_bins; ++b)
          acc[b] += wp * (emg.cdf(lam_max[b], lt, z) -
                          emg.cdf(lam_min[b], lt, z));
      }
    }
    for (int b = 0; b != n_bins; ++b)
      sel[(static_cast<std::size_t>(b) * n_lnm + k) * n_z + q] = acc[b];
  }

  //   R <  r_trans:  DeltaSigma_1h + Delta_gap^1 (R / r_trans)^2
  //   R >= r_trans:  b DeltaSigma_hh + (C_0 / R^2) 1e-12
  // has_piecewise = 0 (tables absent) falls back to max(1h, 2h).
  inline __global__ void
  max_model_contract(int n_bins, int n_R, int n_lnm, int n_z,
                     double const* unique_R,
                     double const* dsigma_nfw_Rk, double const* two_Rq,
                     double const* bias_kq, double const* w2d,
                     int has_piecewise, double const* rtrans_kq,
                     double const* c0_kq, double const* gap1_kq,
                     int has_bsel, int nj_max, double const* gap0_kq,
                     int const* bs_nj, double const* bs_r, double const* bs_c,
                     double const* bs_b, double const* bs_db,
                     double const* bs_sbar_hh, double const* bs_sbar_nfw,
                     double* acc)
  {
    int const tid = blockIdx.x * blockDim.x + threadIdx.x;
    int const total = n_bins * n_R * n_lnm;
    if (tid >= total) return;
    int const k = tid % n_lnm;
    int const ridx = (tid / n_lnm) % n_R;
    int const b = tid / (n_lnm * n_R);

    double const R = unique_R[ridx];
    double const one_k = dsigma_nfw_Rk[ridx * n_lnm + k];

    std::size_t const kq0 = static_cast<std::size_t>(k) * n_z;
    double const* wrow = w2d + (static_cast<std::size_t>(b) * n_lnm + k) * n_z;
    double const* brow = bias_kq + kq0;
    double const* trow = two_Rq + static_cast<std::size_t>(ridx) * n_z;
    // This (bin, R)'s Bsel rows; see build_bsel_wall for the layouts.
    std::size_t const wall = static_cast<std::size_t>(b) * n_R + ridx;
    std::size_t const j0 = wall * nj_max;
    double partial = 0.0;
    for (int q = 0; q != n_z; ++q) {
      double const d_2h = brow[q] * trow[q];
      double d_tot;
      if (has_piecewise) {
        double const rt = rtrans_kq[kq0 + q];
        if (R < rt) {
          double const r_ratio = R / rt;
          d_tot = one_k + gap1_kq[kq0 + q] * r_ratio * r_ratio;
        } else {
          d_tot = d_2h + (c0_kq[kq0 + q] / (R * R)) * 1.0e-12;
        }
      } else {
        d_tot = fmax(one_k, d_2h);
      }
      if (has_bsel) {
        // B(R) d_tot - R sum_j c_j barSigma(<r_j) dB/dR(r_j), with the
        // piecewise barSigma:
        //   r <  r_trans : barSigma_1h(r) + Delta_gap^0 + 2 Delta_gap^1
        //                  - Delta_gap^1 (r / r_trans)^2
        //   r >= r_trans : b barSigma_hh(r) + C_0 / r^2
        double const rt = rtrans_kq[kq0 + q];
        double const gap1 = gap1_kq[kq0 + q];
        double m_sel_over_pi_r2 = 0.0;
        for (int j = 0; j != bs_nj[wall]; ++j) {
          std::size_t const jz = (j0 + j) * n_z + q;
          double const r = bs_r[j0 + j];
          double sbar;
          if (r >= rt) {
            sbar = brow[q] * bs_sbar_hh[jz] +
                   (c0_kq[kq0 + q] / (r * r)) * 1.0e-12;
          } else {
            double const r_ratio = r / rt;
            sbar = bs_sbar_nfw[(j0 + j) * n_lnm + k] + gap0_kq[kq0 + q] +
                   2.0 * gap1 - gap1 * r_ratio * r_ratio;
          }
          m_sel_over_pi_r2 += bs_c[j0 + j] * sbar * bs_db[jz];
        }
        d_tot = bs_b[wall * n_z + q] * d_tot - R * m_sel_over_pi_r2;
      }
      partial += wrow[q] * d_tot;
    }

    atomicAdd(&acc[b * n_R + ridx], partial);
  }

}  // namespace y3_cuda_des_y3

class Shear1h2hMaxExplicitGlGpu {
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

  explicit Shear1h2hMaxExplicitGlGpu(cosmosis::DataBlock& cfg)
    : N_lnm_(cfg.has_val(module_label(), "n_lnm")
               ? cfg.view<int>(module_label(), "n_lnm") : 96)
    , N_z_(cfg.has_val(module_label(), "n_z")
             ? cfg.view<int>(module_label(), "n_z") : 64)
    , N_q_(cfg.has_val(module_label(), "n_q")
             ? cfg.view<int>(module_label(), "n_q") : 32)
    , l_lam_(cfg.has_val(module_label(), "l_lam")
               ? cfg.view<double>(module_label(), "l_lam") : 6.0)
    , zt_lo_(cfg.view<double>(module_label(), "zt_low"))
    , zt_hi_(cfg.view<double>(module_label(), "zt_high"))
    , lnm_lo_(cfg.view<double>(module_label(), "lnm_low"))
    , lnm_hi_(cfg.view<double>(module_label(), "lnm_high"))
    , mis_enabled_(read_bool_opt(cfg, module_label(), "miscentering", false))
    , bsel_enabled_(read_bool_opt(cfg, module_label(), "bsel", false))
    , bsel_model_(bsel_enabled_
                    ? cfg.view<std::string>(module_label(), "bsel_model")
                    : "")
  {
    if (read_bool_opt(cfg, module_label(), "include_miscentering", false))
      throw std::runtime_error(
        "Shear1h2hMaxExplicitGlGpu: include_miscentering (1-halo-only term) "
        "was removed; use miscentering = T");
    y3_pipelines::gl_nodes(lnm_lo_, lnm_hi_, N_lnm_, lnm_x_, lnm_w_);
    double const excl_lo = cfg.has_val(module_label(), "zt_excl_low")
                             ? cfg.view<double>(module_label(), "zt_excl_low")
                             : 0.0;
    double const excl_hi = cfg.has_val(module_label(), "zt_excl_high")
                             ? cfg.view<double>(module_label(), "zt_excl_high")
                             : -1.0;
    z_nodes(zt_lo_, zt_hi_, excl_lo, excl_hi, N_z_, z_x_, z_w_);
    // lambda_true nodes on [-1, 1]; mapped to each (lnM, z) window in the
    // kernel (numpy leggauss twin).
    y3_pipelines::gl_nodes(-1.0, 1.0, N_q_, gl_t_, gl_w_);
    lob_centers_ =
      y3_pipelines::read_lob_centers(cfg, module_label());
    if (lob_centers_.empty())
      throw std::runtime_error("Shear1h2hMaxExplicitGlGpu: lob_centers is empty");

    // Bin definitions come from the config, not from sel_function.
    lam_min_ = get_vector_double(cfg, module_label(), "lam_min");
    lam_max_ = get_vector_double(cfg, module_label(), "lam_max");
    zob_min_ = get_vector_double(cfg, module_label(), "zob_min");
    zob_max_ = get_vector_double(cfg, module_label(), "zob_max");
    sigma_z_ = get_vector_double(cfg, module_label(), "sigma_z");
    n_bins_ = lam_min_.size();
    if (n_bins_ == 0 ||
        n_bins_ > static_cast<std::size_t>(y3_cuda_des_y3::MAX_BINS) ||
        lam_max_.size() != n_bins_ || zob_min_.size() != n_bins_ ||
        zob_max_.size() != n_bins_ || sigma_z_.size() != n_bins_)
      throw std::runtime_error("Shear1h2hMaxExplicitGlGpu: bad bin arrays");

    if (bsel_enabled_)
      y3_cluster::BselModels::parse_tag(bsel_model_);

    r_perp_ = get_vector_double(cfg, module_label(), "r_perp");
    if (r_perp_.empty())
      throw std::runtime_error("Shear1h2hMaxExplicitGlGpu: r_perp is empty");
    if (mis_enabled_)
      mis_.emplace(y3_cluster::Miscentering::Options::from_config(cfg, module_label()),
                   r_perp_, lob_centers_);
  }

  void
  set_sample(cosmosis::DataBlock& s)
  {
    namespace w = y3_pipelines;
    using y3_cuda_des_y3::cuda_check;

    y3_cluster::HMF_t const hmf(s);
    y3_cluster::DV_DO_DZ_t const dv(s);
    y3_cluster::OMEGA_Z_DES const omega(s);
    auto const sci = w::load_sigma_crit_inv(s);
    y3_cuda::MOR_HOD_t const mor(s);
    y3_cuda::EMG_DES_t const emg(s);

    std::optional<y3_cluster::Interp2D> dsigma_nfw, bias, dsigma_hh;
    dsigma_nfw.emplace(y3_cluster::make_Interp2D(
      s, "haloModel", "r_sigma", "lnM", "dSigma_nfw"));
    bias.emplace(
      y3_cluster::make_Interp2D(s, "haloModel", "lnM", "z", "bias"));
    dsigma_hh.emplace(make_sanitized_table(s, "dSigma_hh"));

    // The piecewise max-model tables (published by halo_model when
    // compute_lensing_1h = compute_lensing_2h = T).
    bool const has_piecewise = s.has_val("haloModel", "r_trans");
    std::optional<y3_cluster::Interp2D> r_trans, c0_trans, delta_gap_1;
    if (has_piecewise) {
      r_trans.emplace(
        y3_cluster::make_Interp2D(s, "haloModel", "lnM", "z", "r_trans"));
      c0_trans.emplace(
        y3_cluster::make_Interp2D(s, "haloModel", "lnM", "z", "c0_trans"));
      delta_gap_1.emplace(
        y3_cluster::make_Interp2D(s, "haloModel", "lnM", "z", "delta_gap_1"));
    }

    if ((mis_ || bsel_enabled_) && !has_piecewise)
      throw std::runtime_error(
        "Shear1h2hMaxExplicitGlGpu: miscentering and bsel need the piecewise "
        "max-model tables (halo_model compute_max1h2h = T)");
    bsel_op_.reset();
    if (bsel_enabled_)
      bsel_op_.emplace(
        y3_cluster::BselDeltaSigmaOperator::from_datablock(s, bsel_model_));


        
    {
      int phys = 0;
      if (s.has_val("haloModel", "one_halo_physical_density"))
        s.get_val("haloModel", "one_halo_physical_density", phys);
      if (phys != 0)
        throw std::runtime_error(
          "Shear1h2hMaxExplicitGlGpu: one_halo_physical_density is not implemented "
          "on the GPU max backend; use the CPU Shear1h2hMax");
    }

    auto dalloc = [](auto const& host, double*& dev) {
      cuda_check(cudaMalloc(&dev, host.size() * sizeof(double)), "cudaMalloc");
      cuda_check(cudaMemcpy(dev, host.data(), host.size() * sizeof(double),
                            cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    };

    // ---- Selection at every node (GPU), one thread per (lnM, z).
    std::size_t const n_sel = n_bins_ * N_lnm_ * N_z_;
    std::vector<double> sel(n_sel, 0.0);
    {
      double *d_lnm = nullptr, *d_z = nullptr, *d_t = nullptr, *d_w = nullptr,
             *d_lmin = nullptr, *d_lmax = nullptr, *d_sel = nullptr;
      dalloc(lnm_x_, d_lnm);
      dalloc(z_x_, d_z);
      dalloc(gl_t_, d_t);
      dalloc(gl_w_, d_w);
      dalloc(lam_min_, d_lmin);
      dalloc(lam_max_, d_lmax);
      cuda_check(cudaMalloc(&d_sel, n_sel * sizeof(double)), "cudaMalloc sel");

      int const total = int(N_lnm_ * N_z_);
      int const threads = 128;
      int const blocks = (total + threads - 1) / threads;
      y3_cuda_des_y3::explicit_sel_weights<<<blocks, threads>>>(
        mor, emg, int(n_bins_), int(N_lnm_), int(N_z_), int(N_q_), l_lam_,
        d_lnm, d_z, d_t, d_w, d_lmin, d_lmax, d_sel);
      cuda_check(cudaGetLastError(), "selection kernel launch");
      cuda_check(cudaDeviceSynchronize(), "selection kernel sync");
      cuda_check(cudaMemcpy(sel.data(), d_sel, n_sel * sizeof(double),
                            cudaMemcpyDeviceToHost), "cudaMemcpy sel D2H");
      for (auto* p : {d_lnm, d_z, d_t, d_w, d_lmin, d_lmax, d_sel})
        cudaFree(p);
    }

    // z-only factors (Sigma_crit_inv folded in) and the z-RESOLVED weight
    //   W2d[b,k,q] = lnm_w_k zfac_q n(M_k,z_q) S_j(z_q) sel[b,k,q]
    std::vector<double> zfac(N_z_);
    for (std::size_t q = 0; q != N_z_; ++q)
      zfac[q] = z_w_[q] * dv(z_x_[q]) * omega(z_x_[q]) * sci.clamp(z_x_[q]);

    std::vector<double> w2d(n_sel, 0.0);
    for (std::size_t b = 0; b != n_bins_; ++b)
      for (std::size_t q = 0; q != N_z_; ++q) {
        double const s_j =
          zkernel_sj(z_x_[q], zob_min_[b], zob_max_[b], sigma_z_[b]);
        for (std::size_t k = 0; k != N_lnm_; ++k) {
          std::size_t const i = (b * N_lnm_ + k) * N_z_ + q;
          w2d[i] = lnm_w_[k] * zfac[q] * hmf(lnm_x_[k], z_x_[q]) * s_j *
                   sel[i];
        }
      }

    std::vector<double> bias_kq(N_lnm_ * N_z_, 0.0);
    std::vector<double> rtrans_kq(N_lnm_ * N_z_, 0.0);
    std::vector<double> c0_kq(N_lnm_ * N_z_, 0.0);
    std::vector<double> gap1_kq(N_lnm_ * N_z_, 0.0);
    std::vector<double> gap0_kq(N_lnm_ * N_z_, 0.0);
    std::optional<y3_cluster::Interp2D> delta_gap_0;
    if (mis_ || bsel_enabled_)
      delta_gap_0.emplace(
        y3_cluster::make_Interp2D(s, "haloModel", "lnM", "z", "delta_gap_0"));
    for (std::size_t k = 0; k != N_lnm_; ++k)
      for (std::size_t q = 0; q != N_z_; ++q) {
        std::size_t const i = k * N_z_ + q;
        bias_kq[i] = bias->clamp(lnm_x_[k], z_x_[q]);
        if (has_piecewise) {
          rtrans_kq[i] = r_trans->clamp(lnm_x_[k], z_x_[q]);
          c0_kq[i] = c0_trans->clamp(lnm_x_[k], z_x_[q]);
          gap1_kq[i] = delta_gap_1->clamp(lnm_x_[k], z_x_[q]);
        }
        if (delta_gap_0)
          gap0_kq[i] = delta_gap_0->clamp(lnm_x_[k], z_x_[q]);
      }

    std::size_t const n_R = r_perp_.size();
    std::vector<double> dsigma_nfw_Rk(n_R * N_lnm_, 0.0);
    std::vector<double> two_Rq(n_R * N_z_, 0.0);
    for (std::size_t ridx = 0; ridx != n_R; ++ridx) {
      double const R = r_perp_[ridx];
      for (std::size_t k = 0; k != N_lnm_; ++k)
        dsigma_nfw_Rk[ridx * N_lnm_ + k] = dsigma_nfw->clamp(R, lnm_x_[k]);
      for (std::size_t q = 0; q != N_z_; ++q)
        two_Rq[ridx * N_z_ + q] = dsigma_hh->clamp(R, z_x_[q]);
    }

    BselWall bs;
    if (bsel_op_)
      build_bsel_wall(s, bs);

    // ---- Profile and contraction (GPU), bin x R x lnM threads.
    double *d_R = nullptr, *d_nfwRk = nullptr, *d_twoRq = nullptr,
           *d_biaskq = nullptr, *d_w2d = nullptr, *d_rt = nullptr,
           *d_c0 = nullptr, *d_gap1 = nullptr, *d_acc = nullptr,
           *d_gap0 = nullptr, *d_bs_r = nullptr, *d_bs_c = nullptr,
           *d_bs_b = nullptr, *d_bs_db = nullptr, *d_bs_shh = nullptr,
           *d_bs_snfw = nullptr;
    int* d_bs_nj = nullptr;
    dalloc(r_perp_, d_R);
    dalloc(dsigma_nfw_Rk, d_nfwRk);
    dalloc(two_Rq, d_twoRq);
    dalloc(bias_kq, d_biaskq);
    dalloc(w2d, d_w2d);
    dalloc(rtrans_kq, d_rt);
    dalloc(c0_kq, d_c0);
    dalloc(gap1_kq, d_gap1);
    if (bsel_op_) {
      dalloc(gap0_kq, d_gap0);
      dalloc(bs.r, d_bs_r);
      dalloc(bs.c, d_bs_c);
      dalloc(bs.b, d_bs_b);
      dalloc(bs.d_bsel, d_bs_db);
      dalloc(bs.sbar_hh, d_bs_shh);
      dalloc(bs.sbar_nfw, d_bs_snfw);
      cuda_check(cudaMalloc(&d_bs_nj, bs.nj.size() * sizeof(int)), "cudaMalloc nj");
      cuda_check(cudaMemcpy(d_bs_nj, bs.nj.data(), bs.nj.size() * sizeof(int),
                            cudaMemcpyHostToDevice), "cudaMemcpy nj");
    }
    std::size_t const n_acc = n_bins_ * n_R;
    cuda_check(cudaMalloc(&d_acc, n_acc * sizeof(double)), "cudaMalloc acc");
    cuda_check(cudaMemset(d_acc, 0, n_acc * sizeof(double)), "cudaMemset acc");

    int const total = int(n_bins_ * n_R * N_lnm_);
    int const threads = 256;
    int const blocks = (total + threads - 1) / threads;
    y3_cuda_des_y3::max_model_contract<<<blocks, threads>>>(
      int(n_bins_), int(n_R), int(N_lnm_),
      int(N_z_), d_R, d_nfwRk, d_twoRq, d_biaskq,
      d_w2d, has_piecewise ? 1 : 0, d_rt, d_c0, d_gap1,
      bsel_op_ ? 1 : 0, int(bs.nj_max), d_gap0, d_bs_nj, d_bs_r, d_bs_c,
      d_bs_b, d_bs_db, d_bs_shh, d_bs_snfw, d_acc);
    cuda_check(cudaGetLastError(), "kernel launch");
    cuda_check(cudaDeviceSynchronize(), "kernel sync");

    results_.assign(n_acc, 0.0);
    cuda_check(cudaMemcpy(results_.data(), d_acc, n_acc * sizeof(double),
                          cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    for (auto* p : {d_R, d_nfwRk, d_twoRq, d_biaskq,
                    d_w2d, d_rt, d_c0, d_gap1, d_acc, d_gap0, d_bs_r, d_bs_c,
                    d_bs_b, d_bs_db, d_bs_shh, d_bs_snfw})
      cudaFree(p);
    cudaFree(d_bs_nj);

    n_R_ = n_R;

    if (mis_)
      apply_miscentering(s, w2d, bias_kq, rtrans_kq, gap0_kq, gap1_kq);
  }

  std::array<double, n_outputs>
  evaluate(grid_point_t const& pt) const
  {
    int const b = static_cast<int>(pt[0]);
    double const R = pt[1];
    if (b < 0 || static_cast<std::size_t>(b) >= n_bins_)
      throw std::out_of_range("Shear1h2hMaxExplicitGlGpu: bin_index out of range");
    for (std::size_t ridx = 0; ridx != r_perp_.size(); ++ridx) {
      if (std::abs(r_perp_[ridx] - R) < 1e-9)
        return {results_[static_cast<std::size_t>(b) * n_R_ + ridx]};
    }
    throw std::out_of_range("Shear1h2hMaxExplicitGlGpu: r_perp not in configured grid");
  }

  static char const* module_label() { return "Shear1h2hMaxExplicitGlGpu"; }

  static std::array<char const*, n_outputs>
  output_sections()
  {
    return {"shear1h2h_max_explicit_gl_gpu"};
  }

  static grid_t
  make_grid_points(cosmosis::DataBlock& cfg)
  {
    return y3_cluster::make_grid_points_cartesian_product(
      cfg, module_label(), "bin_index", "r_perp");
  }

private:
  // Bsel factors on the (bin, R) wall, for the kernel. The operator's nodes
  // depend only on (bin, R), and each factor on lnM OR z, never both. The
  // node count is 2 n_gl at most (Sunayama23 splits at its break radius).
  // With wall = b * n_R + ridx and j0 = wall * nj_max:
  //   nj[wall], r[j0 + j], c[j0 + j],
  //   b[wall * n_z + q]                 B_sel(R; z_q)
  //   d_bsel[(j0 + j) * n_z + q]        dB/dR(r_j; z_q)
  //   sbar_hh[(j0 + j) * n_z + q]       barSigma_hh(r_j | z_q)
  //   sbar_nfw[(j0 + j) * n_lnm + k]    barSigma_1h(r_j | lnM_k)
  struct BselWall {
    std::size_t nj_max{0};
    std::vector<int> nj;
    std::vector<double> r, c, b, d_bsel, sbar_hh, sbar_nfw;
  };

  void
  build_bsel_wall(cosmosis::DataBlock& s, BselWall& T) const
  {
    auto const sigma_bar_nfw = y3_cluster::make_Interp2D(
      s, "haloModel", "r_sigma", "lnM", "Sigma_bar_nfw");
    auto const sigma_bar_hh = make_sanitized_table(s, "Sigma_bar_hh");
    auto const& model = bsel_op_->model();
    std::size_t const n_R = r_perp_.size(), n_wall = n_bins_ * n_R;
    T.nj_max = 2 * bsel_op_->n_gl();
    T.nj.assign(n_wall, 0);
    T.r.assign(n_wall * T.nj_max, 0.0);
    T.c.assign(n_wall * T.nj_max, 0.0);
    T.b.assign(n_wall * N_z_, 0.0);
    T.d_bsel.assign(n_wall * T.nj_max * N_z_, 0.0);
    T.sbar_hh.assign(n_wall * T.nj_max * N_z_, 0.0);
    T.sbar_nfw.assign(n_wall * T.nj_max * N_lnm_, 0.0);
    std::vector<double> r, c;
    for (std::size_t b = 0; b != n_bins_; ++b) {
      int const bin = static_cast<int>(b);
      double const lob = lob_centers_[b % lob_centers_.size()];
      for (std::size_t i = 0; i != n_R; ++i) {
        double const R = r_perp_[i];
        std::size_t const wall = b * n_R + i, j0 = wall * T.nj_max;
        bsel_op_->nodes(R, bin, r, c);
        T.nj[wall] = static_cast<int>(r.size());
        for (std::size_t q = 0; q != N_z_; ++q)
          T.b[wall * N_z_ + q] = model(R, bin, lob, z_x_[q]);
        for (std::size_t j = 0; j != r.size(); ++j) {
          T.r[j0 + j] = r[j];
          T.c[j0 + j] = c[j];
          for (std::size_t q = 0; q != N_z_; ++q) {
            std::size_t const jz = (j0 + j) * N_z_ + q;
            T.d_bsel[jz] = model.derivative(r[j], bin, lob, z_x_[q]);
            T.sbar_hh[jz] = sigma_bar_hh.clamp(r[j], z_x_[q]);
          }
          for (std::size_t k = 0; k != N_lnm_; ++k)
            T.sbar_nfw[(j0 + j) * N_lnm_ + k] =
              sigma_bar_nfw.clamp(r[j], lnm_x_[k]);
        }
      }
    }
  }

  // Mixes O_mis into results_ (host side, same model as the CPU Shear1h2hMax).
  // Sigma_max(r | lnM_k, z_q) comes from the Sigma tables as natural cubic
  // splines in ln r, one per lnM node (Sigma_1h) and one per z node
  // (Sigma_hh), with a uniform disk of mass haloModel/m2d_core below
  // r_core = r_sigma[0]:
  //   r <  r_core  : m2d_core / (pi r_core^2)
  //   r <  r_trans : Sigma_1h(r) + Delta_gap^0 + 2 Delta_gap^1 (1 - (r/r_trans)^2)
  //   r >= r_trans : b Sigma_hh(r)
  // The Stack and Profile below refer to locals. That is safe: they are only
  // called inside set_sample, and on-wall evaluate() reads stored results.
  void
  apply_miscentering(cosmosis::DataBlock& s, std::vector<double> const& w2d,
                     std::vector<double> const& bias_kq,
                     std::vector<double> const& rtrans_kq,
                     std::vector<double> const& gap0_kq,
                     std::vector<double> const& gap1_kq)
  {
    using y3_cluster::CubicSplineLnR;
    using y3_cluster::Miscentering;
    auto const sigma_nfw = y3_cluster::make_Interp2D(
      s, "haloModel", "r_sigma", "lnM", "Sigma_nfw");
    auto const sigma_hh = make_sanitized_table(s, "Sigma_hh");
    auto const m2d_core = y3_cluster::make_Interp2D(
      s, "haloModel", "lnM", "z", "m2d_core");
    auto const r_sigma = s.view<std::vector<double>>("haloModel", "r_sigma");
    double const r_core = r_sigma.front();

    std::vector<CubicSplineLnR> spl_nfw, spl_hh;
    std::vector<double> knots(r_sigma.size());
    for (std::size_t k = 0; k != N_lnm_; ++k) {
      for (std::size_t i = 0; i != r_sigma.size(); ++i)
        knots[i] = sigma_nfw.clamp(r_sigma[i], lnm_x_[k]);
      spl_nfw.emplace_back(r_sigma, knots);
    }
    for (std::size_t q = 0; q != N_z_; ++q) {
      for (std::size_t i = 0; i != r_sigma.size(); ++i)
        knots[i] = sigma_hh.clamp(r_sigma[i], z_x_[q]);
      spl_hh.emplace_back(r_sigma, knots);
    }

    constexpr double PI = 3.14159265358979323846;
    double const area = PI * r_core * r_core * 1.0e12;  // -> Msun h / pc^2
    std::vector<double> core_kq(N_lnm_ * N_z_);
    for (std::size_t k = 0; k != N_lnm_; ++k)
      for (std::size_t q = 0; q != N_z_; ++q)
        core_kq[k * N_z_ + q] = m2d_core.clamp(lnm_x_[k], z_x_[q]) / area;

    auto const sigma_max = [&](double r, double sig_nfw, double sig_hh,
                               std::size_t idx) {
      double const rt = rtrans_kq[idx];
      if (r >= rt)
        return bias_kq[idx] * sig_hh;
      double const r_ratio = r / rt;
      return sig_nfw + gap0_kq[idx] +
             2.0 * gap1_kq[idx] * (1.0 - r_ratio * r_ratio);
    };

    Miscentering::Stack const stack{
      n_bins_, N_lnm_, N_z_, w2d.data(), &z_x_,
      bsel_op_ ? &bsel_op_->model() : nullptr};
    Miscentering::Profile profile;
    profile.r_lo = r_core;
    profile.r_max = r_sigma.back();
    profile.core = core_kq;
    profile.grid = [&](double r, std::vector<double>& out) {
      std::vector<double> one(N_lnm_), two(N_z_);
      for (std::size_t k = 0; k != N_lnm_; ++k) one[k] = spl_nfw[k](r);
      for (std::size_t q = 0; q != N_z_; ++q) two[q] = spl_hh[q](r);
      for (std::size_t k = 0; k != N_lnm_; ++k)
        for (std::size_t q = 0; q != N_z_; ++q)
          out[k * N_z_ + q] = sigma_max(r, one[k], two[q], k * N_z_ + q);
    };
    profile.at = [&](double r, std::size_t k, std::size_t q) {
      std::size_t const idx = k * N_z_ + q;
      if (r < r_core)
        return core_kq[idx];
      return sigma_max(r, spl_nfw[k](r), spl_hh[q](r), idx);
    };
    mis_->set_sample(s, stack, profile);

    for (std::size_t b = 0; b != n_bins_; ++b)
      for (std::size_t ridx = 0; ridx != n_R_; ++ridx) {
        double& o = results_[b * n_R_ + ridx];
        o = mis_->apply(o, mis_->evaluate(int(b), r_perp_[ridx]));
      }
  }

  // True-z grid: one GL panel on [lo, hi], or two panels around the
  // excluded true-z range [excl_lo, excl_hi] (Buzzard mocks drop halos
  // there; same keys as sel_function zt_excl_low/zt_excl_high). The n
  // nodes are shared between the panels in proportion to their lengths.
  // Inactive unless excl_hi > excl_lo inside [lo, hi].
  static void
  z_nodes(double lo, double hi, double excl_lo, double excl_hi,
          std::size_t n, std::vector<double>& x, std::vector<double>& w)
  {
    excl_lo = std::max(excl_lo, lo);
    excl_hi = std::min(excl_hi, hi);
    if (!(excl_hi > excl_lo)) {
      y3_pipelines::gl_nodes(lo, hi, n, x, w);
      return;
    }
    double const l1 = excl_lo - lo;
    double const l2 = hi - excl_hi;
    if (l1 <= 0.0 && l2 <= 0.0)
      throw std::runtime_error(
        "Shear1h2hMaxExplicitGlGpu: zt_excl range covers [zt_low, zt_high]");
    if (l1 <= 0.0) {
      y3_pipelines::gl_nodes(excl_hi, hi, n, x, w);
      return;
    }
    if (l2 <= 0.0) {
      y3_pipelines::gl_nodes(lo, excl_lo, n, x, w);
      return;
    }
    if (n < 2)
      throw std::runtime_error(
        "Shear1h2hMaxExplicitGlGpu: n_z must be >= 2 with a zt_excl range");
    long n1 = std::lround(double(n) * l1 / (l1 + l2));
    n1 = std::min(std::max(n1, 1L), long(n) - 1);
    std::vector<double> x2, w2;
    y3_pipelines::gl_nodes(lo, excl_lo, std::size_t(n1), x, w);
    y3_pipelines::gl_nodes(excl_hi, hi, n - std::size_t(n1), x2, w2);
    x.insert(x.end(), x2.begin(), x2.end());
    w.insert(w.end(), w2.begin(), w2.end());
  }

  // Observed-redshift kernel S_j: Gaussian CDF difference over the z_ob
  // bin (Python sel_function._S_j).
  static double
  zkernel_sj(double zt, double zob_min, double zob_max, double sigma_z)
  {
    constexpr double SQRT2_INV = 0.7071067811865475;
    return 0.5 * (std::erf((zob_max - zt) / sigma_z * SQRT2_INV) -
                  std::erf((zob_min - zt) / sigma_z * SQRT2_INV));
  }

  // A haloModel (r_sigma, z) table through Interp2D, with the producer's NaNs
  // replaced by 0 first (defensive guard; the producer defect in
  // docs/known_issues/dsigma_hh_debug_flag.md is fixed), same convention
  // as the CPU backend.
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
      throw std::runtime_error(std::string("Shear1h2hMaxExplicitGlGpu: ") + key +
                               " extents mismatch");
    for (auto& v : vals)
      if (!std::isfinite(v)) v = 0.0;
    return y3_cluster::Interp2D(r, z, vals);
  }

  std::size_t N_lnm_, N_z_, N_q_;
  double l_lam_;
  double zt_lo_, zt_hi_, lnm_lo_, lnm_hi_;
  bool mis_enabled_;
  bool bsel_enabled_;
  std::string bsel_model_;
  std::optional<y3_cluster::Miscentering> mis_;
  std::optional<y3_cluster::BselDeltaSigmaOperator> bsel_op_;
  std::vector<double> lnm_x_, lnm_w_, z_x_, z_w_, gl_t_, gl_w_;
  std::vector<double> lob_centers_, r_perp_;
  std::vector<double> lam_min_, lam_max_, zob_min_, zob_max_, sigma_z_;

  std::size_t n_bins_{0};
  std::size_t n_R_{0};
  std::vector<double> results_;
};

#endif
