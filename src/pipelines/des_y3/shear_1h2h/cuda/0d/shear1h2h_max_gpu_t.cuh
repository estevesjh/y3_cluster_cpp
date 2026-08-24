// Traditional 1h+2h shear via the max model — GPU adaptation.
//
// A port of the des_y3 fast_mass C++ backend (../cpp/shear1h2h_max_t.hh):
// same observable,
//
//   d_tot(R, lnM, z | bin) = max( DSigma_cl(R, lnM | bin),
//                                 b(lnM, z) * DSigma_hh(R, z) )
//   O_ij(R) = int dz int dlnM  n dV/dOmegadz Omega Sigma_crit^-1
//             S_ij(lnM, z) d_tot(R, lnM, z),
//
// with DSigma_cl the target-cluster miscentred mixture (gamma-kernel
// NFW, f_mis/tau_mis). The one structural change from the CPU class is
// WHERE the dominant cost runs: DSigma_cl's miscentred piece
// (y3_cluster::NFW_DSIGMA_MIS, several transcendentals + a table lookup
// per (bin, R, lnM) node -- 12 x 10 x 96 = 11,520 evaluations on the
// production grid, the measured cost driver of the CPU class) is a
// single CUDA kernel over y3_cuda::NFW_DSIGMA_MIS (device-resident
// quad::Interp2D table, passed by value exactly as the other des_y3 GPU
// modules pass their integrands, with set_rho_mult(Omega_m) applied on
// the host copy before launch); the (lnM, z) contraction against the
// z-resolved selection weight runs inside the same kernel, one thread
// per (bin, R, lnM) node, reduced over z with atomicAdd into the
// (bin, R) accumulator. Everything else (HMF/selection-weight build,
// the 2-halo dSigma_hh/bias tables, the centred-NFW table) stays
// host-side with the same immutable Interp1D/Interp2D models the C++
// class uses -- none of that is the bottleneck (O(n_bins*N_lnm*N_z)
// bilinear lookups, cheap relative to the transcendental-heavy
// miscentred-NFW piece).
//
// All (bin, R) wall results are assembled at the end of set_sample;
// evaluate() only reads the cache.
//
// dSigma_hh carries NaN over ~60% of its (R, z) table by construction
// (see docs/known_issues/dsigma_hh_debug_flag.md); sanitized to 0 before Interp2D,
// same convention as the CPU backend.
//
// Options: bin_index x r_perp cartesian grid (bin slow / R fast),
// zt_low/zt_high/lnm_low/lnm_high (required), n_lnm (96), n_z (64),
// lob_centers, include_miscentering (default T). Requires
// compute_lensing_2h = T (same haloModel contract as the CPU backend).
// f_mis and tau_mis are REQUIRED datablock values (miscentering/f_mis,
// miscentering/tau_mis): set_sample throws if the section is missing —
// no silent fallback to the Y3 fiducial defaults.
// Output: shear1h2h_max_gpu/vals -- own namespace section (distinct
// from the CPU Shear1h2hMax.so's shear1h2h_max, so both can co-run in
// one pipeline for direct comparison; DataBlock sections do not
// overwrite).
// Status: reference/optimization backend; the CPU Shear1h2hMax.so
// remains the reference implementation for this observable. No
// existing template, model, or module is modified; registry entry
// added.
#ifndef Y3_CLUSTER_CPP_SHEAR1H2H_MAX_GPU_T_CUH
#define Y3_CLUSTER_CPP_SHEAR1H2H_MAX_GPU_T_CUH

#include "cosmosis/datablock/datablock.hh"
#include "cosmosis/datablock/ndarray.hh"

#include "models/dv_do_dz_t.hh"
#include "models/hmf_t.hh"
#include "models/n_operator_sel_gl_t.hh"
#include "models/nfw_dsigma_mis.cuh"
#include "models/omega_z_des.hh"
#include "models/p_operator_cuhre_t.hh"
#include "models/sel_function_t.hh"
#include "pipelines/shared/lensing_helpers.hh"
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
#include <vector>

namespace y3_cuda_des_y3 {

  inline void
  cuda_check(cudaError_t rc, char const* what)
  {
    if (rc != cudaSuccess)
      throw std::runtime_error(std::string("Shear1h2hMaxGpu: ") + what +
                               ": " + cudaGetErrorString(rc));
  }

  // One thread per (bin, R, lnM-node): compute the miscentred-NFW piece
  // of the 1-halo term (the dominant transcendental cost) and reduce
  // the max(1h, 2h) contraction over the z nodes, atomicAdd-ing the
  // partial sum into the (bin, R) accumulator. The nfw object arrives
  // with rho_mult already set to Omega_m by the host.
  inline __global__ void
  max_model_contract(y3_cuda::NFW_DSIGMA_MIS nfw,
                     int n_bins, int n_R, int n_lnm, int n_z,
                     double const* lnm_x, double const* unique_R,
                     double const* r_mis, double const* f_mis,
                     double const* dsigma_nfw_Rk, double const* two_Rq,
                     double const* bias_kq, double const* w2d,
                     double* acc)
  {
    int const tid = blockIdx.x * blockDim.x + threadIdx.x;
    int const total = n_bins * n_R * n_lnm;
    if (tid >= total) return;
    int const k = tid % n_lnm;
    int const ridx = (tid / n_lnm) % n_R;
    int const b = tid / (n_lnm * n_R);

    double const R = unique_R[ridx];
    double const d_cen = dsigma_nfw_Rk[ridx * n_lnm + k];
    double const d_mis = nfw(R, r_mis[b], lnm_x[k]);
    double const one_k = (1.0 - f_mis[b]) * d_cen + f_mis[b] * d_mis;

    double const* wrow = w2d + (static_cast<std::size_t>(b) * n_lnm + k) * n_z;
    double const* brow = bias_kq + static_cast<std::size_t>(k) * n_z;
    double const* trow = two_Rq + static_cast<std::size_t>(ridx) * n_z;
    double partial = 0.0;
    for (int q = 0; q != n_z; ++q)
      partial += wrow[q] * fmax(one_k, brow[q] * trow[q]);

    atomicAdd(&acc[b * n_R + ridx], partial);
  }

}  // namespace y3_cuda_des_y3

class Shear1h2hMaxGpu {
public:
  using grid_t = y3_cluster::grid_t<2>;
  using grid_point_t = grid_t::value_type;
  static constexpr std::size_t n_outputs = 1;

  explicit Shear1h2hMaxGpu(cosmosis::DataBlock& cfg)
    : N_lnm_(cfg.has_val(module_label(), "n_lnm")
               ? cfg.view<int>(module_label(), "n_lnm") : 96)
    , N_z_(cfg.has_val(module_label(), "n_z")
             ? cfg.view<int>(module_label(), "n_z") : 64)
    , zt_lo_(cfg.view<double>(module_label(), "zt_low"))
    , zt_hi_(cfg.view<double>(module_label(), "zt_high"))
    , lnm_lo_(cfg.view<double>(module_label(), "lnm_low"))
    , lnm_hi_(cfg.view<double>(module_label(), "lnm_high"))
    , include_mis_(cfg.has_val(module_label(), "include_miscentering")
                     ? cfg.view<int>(module_label(),
                                     "include_miscentering") != 0
                     : true)
    // TODO(#14): drop the hardcoded c=4 — once the concentration-table
    // mirror lands in nfw_dsigma_mis.cuh (CPU half is
    // claude/issue-4-dsigma-hh-2h-term; the CUDA half needs a Perlmutter
    // build), read haloModel/concentration so the miscentred term uses
    // the same per-mass Child18 concentration as the centred table.
    , dsigma_mis_dev_(y3_cuda::DSIGMA_MIS_CONC, y3_cuda::DSIGMA_MIS_RHOC,
                      y3_cuda::DSIGMA_MIS_GAMMA)
  {
    y3_cluster::p_op_detail::gl_nodes(lnm_lo_, lnm_hi_, N_lnm_, lnm_x_, lnm_w_);
    y3_cluster::p_op_detail::gl_nodes(zt_lo_, zt_hi_, N_z_, z_x_, z_w_);
    lob_centers_ =
      y3_pipelines::read_lob_centers(cfg, module_label());
    if (lob_centers_.empty())
      throw std::runtime_error("Shear1h2hMaxGpu: lob_centers is empty");

    r_perp_ = get_vector_double(cfg, module_label(), "r_perp");
    if (r_perp_.empty())
      throw std::runtime_error("Shear1h2hMaxGpu: r_perp is empty");
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

    std::optional<y3_cluster::Interp2D> dsigma_nfw, bias, dsigma_hh;
    dsigma_nfw.emplace(y3_cluster::make_Interp2D(
      s, "haloModel", "r_sigma", "lnM", "dSigma_nfw"));
    bias.emplace(
      y3_cluster::make_Interp2D(s, "haloModel", "lnM", "z", "bias"));
    dsigma_hh.emplace(make_sanitized_hh(s));

    // Required: no fallback to the fiducial defaults — a pipeline that
    // has not published the miscentering section must fail loudly.
    double const f_mis_scalar =
      include_mis_ ? s.view<double>("miscentering", "f_mis") : 0.0;
    double const tau_mis = s.view<double>("miscentering", "tau_mis");
    dsigma_mis_dev_.set_rho_mult(
      s.view<double>("cosmological_parameters", "omega_M"));

    // z-only factors (Sigma_crit_inv folded in) and the z-RESOLVED
    // weight W2d(bin; lnM, z) -- identical construction to the CPU
    // backend, host-side (not the bottleneck: O(n_bins*N_lnm*N_z)
    // bilinear lookups via HMF_t/SelFunction_t).
    std::vector<double> zfac(N_z_);
    for (std::size_t q = 0; q != N_z_; ++q)
      zfac[q] = z_w_[q] * dv(z_x_[q]) * omega(z_x_[q]) * sci.clamp(z_x_[q]);

    int const n_bins = y3_cluster::nosel_detail::n_bins_from_block(s);
    n_bins_ = static_cast<std::size_t>(n_bins);
    std::vector<double> w2d(n_bins_ * N_lnm_ * N_z_, 0.0);
    for (int b = 0; b != n_bins; ++b) {
      y3_cluster::SelFunction_t const sel(s, b);
      for (std::size_t k = 0; k != N_lnm_; ++k)
        for (std::size_t q = 0; q != N_z_; ++q)
          w2d[(static_cast<std::size_t>(b) * N_lnm_ + k) * N_z_ + q] =
            lnm_w_[k] * zfac[q] * hmf(lnm_x_[k], z_x_[q]) *
            sel(lnm_x_[k], z_x_[q]);
    }

    std::vector<double> bias_kq(N_lnm_ * N_z_, 0.0);
    for (std::size_t k = 0; k != N_lnm_; ++k)
      for (std::size_t q = 0; q != N_z_; ++q)
        bias_kq[k * N_z_ + q] = bias->clamp(lnm_x_[k], z_x_[q]);

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

    std::vector<double> r_mis(n_bins_, 0.0), f_mis(n_bins_, f_mis_scalar);
    for (std::size_t b = 0; b != n_bins_; ++b)
      r_mis[b] = tau_mis * w::R_lambda(
                             lob_centers_[b % lob_centers_.size()]);

    // ---- device part: one kernel, bin x R x lnM threads.
    auto dalloc = [](auto const& host, double*& dev) {
      cuda_check(cudaMalloc(&dev, host.size() * sizeof(double)), "cudaMalloc");
      cuda_check(cudaMemcpy(dev, host.data(), host.size() * sizeof(double),
                            cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    };
    double *d_lnm = nullptr, *d_R = nullptr, *d_rmis = nullptr,
           *d_fmis = nullptr, *d_nfwRk = nullptr, *d_twoRq = nullptr,
           *d_biaskq = nullptr, *d_w2d = nullptr, *d_acc = nullptr;
    dalloc(lnm_x_, d_lnm);
    dalloc(r_perp_, d_R);
    dalloc(r_mis, d_rmis);
    dalloc(f_mis, d_fmis);
    dalloc(dsigma_nfw_Rk, d_nfwRk);
    dalloc(two_Rq, d_twoRq);
    dalloc(bias_kq, d_biaskq);
    dalloc(w2d, d_w2d);
    std::size_t const n_acc = n_bins_ * n_R;
    cuda_check(cudaMalloc(&d_acc, n_acc * sizeof(double)), "cudaMalloc acc");
    cuda_check(cudaMemset(d_acc, 0, n_acc * sizeof(double)), "cudaMemset acc");

    int const total = int(n_bins_ * n_R * N_lnm_);
    int const threads = 256;
    int const blocks = (total + threads - 1) / threads;
    y3_cuda_des_y3::max_model_contract<<<blocks, threads>>>(
      dsigma_mis_dev_, int(n_bins_), int(n_R), int(N_lnm_),
      int(N_z_), d_lnm, d_R, d_rmis, d_fmis, d_nfwRk, d_twoRq, d_biaskq,
      d_w2d, d_acc);
    cuda_check(cudaGetLastError(), "kernel launch");
    cuda_check(cudaDeviceSynchronize(), "kernel sync");

    results_.assign(n_acc, 0.0);
    cuda_check(cudaMemcpy(results_.data(), d_acc, n_acc * sizeof(double),
                          cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    for (auto* p : {d_lnm, d_R, d_rmis, d_fmis, d_nfwRk, d_twoRq, d_biaskq,
                    d_w2d, d_acc})
      cudaFree(p);

    n_R_ = n_R;
  }

  std::array<double, n_outputs>
  evaluate(grid_point_t const& pt) const
  {
    int const b = static_cast<int>(pt[0]);
    double const R = pt[1];
    if (b < 0 || static_cast<std::size_t>(b) >= n_bins_)
      throw std::out_of_range("Shear1h2hMaxGpu: bin_index out of range");
    for (std::size_t ridx = 0; ridx != r_perp_.size(); ++ridx) {
      if (std::abs(r_perp_[ridx] - R) < 1e-9)
        return {results_[static_cast<std::size_t>(b) * n_R_ + ridx]};
    }
    throw std::out_of_range("Shear1h2hMaxGpu: r_perp not in configured grid");
  }

  static char const* module_label() { return "Shear1h2hMaxGpu"; }

  static std::array<char const*, n_outputs>
  output_sections()
  {
    return {"shear1h2h_max_gpu"};
  }

  static grid_t
  make_grid_points(cosmosis::DataBlock& cfg)
  {
    return y3_cluster::make_grid_points_cartesian_product(
      cfg, module_label(), "bin_index", "r_perp");
  }

private:
  // haloModel/dSigma_hh through Interp2D, with the producer's NaNs
  // replaced by 0 first (docs/known_issues/dsigma_hh_debug_flag.md), same convention
  // as the CPU backend.
  static y3_cluster::Interp2D
  make_sanitized_hh(cosmosis::DataBlock& s)
  {
    using doubles = std::vector<double>;
    auto const r = s.view<doubles>("haloModel", "r_sigma");
    auto const z = s.view<doubles>("haloModel", "z");
    auto const& nd =
      s.view<cosmosis::ndarray<double>>("haloModel", "dSigma_hh");
    std::vector<double> vals(nd.begin(), nd.end());
    if (vals.size() != r.size() * z.size())
      throw std::runtime_error("Shear1h2hMaxGpu: dSigma_hh extents mismatch");
    for (auto& v : vals)
      if (!std::isfinite(v)) v = 0.0;
    return y3_cluster::Interp2D(r, z, vals);
  }

  std::size_t N_lnm_, N_z_;
  double zt_lo_, zt_hi_, lnm_lo_, lnm_hi_;
  bool include_mis_;
  y3_cuda::NFW_DSIGMA_MIS dsigma_mis_dev_;
  std::vector<double> lnm_x_, lnm_w_, z_x_, z_w_, lob_centers_, r_perp_;

  std::size_t n_bins_{0};
  std::size_t n_R_{0};
  std::vector<double> results_;
};

#endif
