// GPU Cluster number counts — full (lambda_true, lnM, z) fixed-GL reference,
// 
// GPU (explicit_sel_weights kernel): richness selection at every (lnM, z) node
//   sel[b,k,q] = sum_i w_i P_HOD(lt_i | M_k, z_q)
//                * [F(lam_max | lt_i, z_q) - F(lam_min | lt_i, z_q)]
//   - lambda_true window per (lnM, z)
//   - P_HOD from MOR_HOD_t
//   - F from EMG_DES_t::cdf
//
// CPU, before the kernel:
//   - GL nodes and weights for lnM, z and lambda_true
//   - bin edges from the ini
//   - models from the datablock
//   - copy inputs to the GPU
//
// CPU, after the kernel:
//   - copy sel back from the GPU
//   - n(M,z), dV/dOmegadz, Omega(z) and S_j(z)
//   - final sum
//     N_b = sum_kq w_k w_q n(M_k,z_q) dV/dOmegadz(z_q) Omega(z_q) S_j(z_q)
//           * sel[b,k,q]


// Note: PAGANI runs the whole integral on the GPU. This module runs only the selection on the GPU and sums over (lnM, z) on the CPU.
// TO BE INVESTIGATED: if moving info from the GPU to the CPU is a bottleneck, consider doing the integral on CPU instead (NumCountsExplicitGlCpu). 
// Output: numcounts_explicit_gl_gpu/vals, one entry per bin.

#ifndef Y3_CLUSTER_CPP_NUMCOUNTS_EXPLICIT_GL_GPU_CUH
#define Y3_CLUSTER_CPP_NUMCOUNTS_EXPLICIT_GL_GPU_CUH

#include "cosmosis/datablock/datablock.hh"

#include "models/dv_do_dz_t.hh"
#include "models/emg_des_t.cuh"
#include "models/hmf_t.hh"
#include "models/mor_hod_t.cuh"
#include "models/omega_z_des.hh"
#include "pipelines/shared/sel_gl_weights.hh" // for gl_nodes only 
#include "utils/datablock_reader.hh"
#include "utils/make_grid_points.hh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>







namespace y3_cuda_des_y3 {

  inline void
  cuda_check(cudaError_t rc, char const* what)
  {
    if (rc != cudaSuccess)
      throw std::runtime_error(std::string("NumCountsExplicitGlGpu: ") + what +
                               ": " + cudaGetErrorString(rc));
  }

  // Richness selection at every fixed-GL node
  // One thread per (lnM, z) node. For each node: the lambda_true window
  // [mu_eff - L sig_eff, mu_eff + L sig_eff] (cut at 0), n_q GL nodes in
  // it, and for every bin
  // sel[b,k,q] = sum_i  w_i P_HOD(lt_i | M, z) [CDF(lam_max) - CDF(lam_min)]
  // P_HOD is MOR_HOD_t, the EMG CDF is EMG_DES_t.
  // Identical to the kernel in Shear1h2hMaxExplicitGlGpu.cuh.

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

}  // namespace y3_cuda_des_y3

class NumCountsExplicitGlGpu {
public:
  using grid_t = y3_cluster::grid_t<1>;
  using grid_point_t = grid_t::value_type;
  static constexpr std::size_t n_outputs = 1;

  explicit NumCountsExplicitGlGpu(cosmosis::DataBlock& cfg)
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
  {
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
      throw std::runtime_error("NumCountsExplicitGlGpu: bad bin arrays");
  }

  void
  set_sample(cosmosis::DataBlock& s)
  {
    using y3_cuda_des_y3::cuda_check;

    y3_cluster::HMF_t const hmf(s);
    y3_cluster::DV_DO_DZ_t const dv(s);
    y3_cluster::OMEGA_Z_DES const omega(s);
    y3_cuda::MOR_HOD_t const mor(s);
    y3_cuda::EMG_DES_t const emg(s);

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

    // ---- (lnM, z) sum on the host, per bin:
    //   N_b = sum_kq lnm_w_k zfac_q n(M_k,z_q) S_j(z_q) sel[b,k,q]
    // (no Sigma_crit_inv: counts carry no lensing weight).
    std::vector<double> zfac(N_z_);
    for (std::size_t q = 0; q != N_z_; ++q)
      zfac[q] = z_w_[q] * dv(z_x_[q]) * omega(z_x_[q]);

    results_.assign(n_bins_, 0.0);
    for (std::size_t b = 0; b != n_bins_; ++b)
      for (std::size_t q = 0; q != N_z_; ++q) {
        double const s_j =
          zkernel_sj(z_x_[q], zob_min_[b], zob_max_[b], sigma_z_[b]);
        for (std::size_t k = 0; k != N_lnm_; ++k) {
          std::size_t const i = (b * N_lnm_ + k) * N_z_ + q;
          results_[b] += lnm_w_[k] * zfac[q] * hmf(lnm_x_[k], z_x_[q]) *
                         s_j * sel[i];
        }
      }
  }

  std::array<double, n_outputs>
  evaluate(grid_point_t const& pt) const
  {
    int const b = static_cast<int>(pt[0]);
    if (b < 0 || static_cast<std::size_t>(b) >= n_bins_)
      throw std::out_of_range("NumCountsExplicitGlGpu: bin_index out of range");
    return {results_[static_cast<std::size_t>(b)]};
  }

  static char const* module_label() { return "NumCountsExplicitGlGpu"; }

  static std::array<char const*, n_outputs>
  output_sections()
  {
    return {"numcounts_explicit_gl_gpu"};
  }

  static grid_t
  make_grid_points(cosmosis::DataBlock& cfg)
  {
    return y3_cluster::make_grid_points_wall_of_numbers(cfg, module_label(),
                                                        "bin_index");
  }

private:
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
        "NumCountsExplicitGlGpu: zt_excl range covers [zt_low, zt_high]");
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
        "NumCountsExplicitGlGpu: n_z must be >= 2 with a zt_excl range");
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

  std::size_t N_lnm_, N_z_, N_q_;
  double l_lam_;
  double zt_lo_, zt_hi_, lnm_lo_, lnm_hi_;
  std::vector<double> lnm_x_, lnm_w_, z_x_, z_w_, gl_t_, gl_w_;
  std::vector<double> lam_min_, lam_max_, zob_min_, zob_max_, sigma_z_;

  std::size_t n_bins_{0};
  std::vector<double> results_;
};

#endif
