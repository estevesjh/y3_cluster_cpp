// Projection shear, frozen-physics fast path — GPU adaptation.
//
// A faithful port of the production ShearPrjFrozenPhysics machinery
// (systematics/shear_prj/cpp/sigma_prj_frozen_t.hh, the
// mock_mcmc_buzzard.ini stage):
// identical theta grid (y3_cluster::sp_detail::build_theta_grid), identical
// ring+wings z grid, identical frozen clustered channel (mass shape at
// z_ob, r_s-anchored scalar drift a_b(z)), identical b_sel plateaus and
// photo-z weight. The one structural change is WHERE the dominant cost
// runs: the per-slice DSigma_mis cache and its mass contraction
// (~550k single-kernel NFW table lookups + ~1.1M FMA per sample) are a
// single CUDA kernel over y3_cuda::NFW_DSIGMA_MIS (device-resident
// quad::Interp2D table, passed by value exactly as the PAGANI modules
// pass their integrands). Everything cheap stays host-side with the
// same immutable Interp1D/Interp2D models the C++ class uses.
//
// All 180 wall results are assembled at the end of set_sample;
// evaluate() only reads the cache. Outputs (namespace sections, no
// production shear_prj alias — DataBlock sections do not overwrite):
//   dsigma_prj_frozen_gpu/{vals,rnd,cl}   Msun/(h pc^2)
//   shear_prj_frozen_gpu/{vals,rnd,cl}    dimensionless
// Ini section ShearPrjFrozenGpu: same knobs and wall as production
// shear_prj_frozen_physics (zt/lnm bounds, R_max_cMpch, n_lnm,
// n_per_seg, n_zring, n_zouter, include_omega_z, lob_centers).
// Status: reference/optimization backend; production remains
// ShearPrjFrozenPhysics.so.
#include "cosmosis/datablock/datablock.hh"

#include "models/dv_do_dz_t.hh"
#include "models/hmf_t.hh"
#include "models/nfw_dsigma_mis.cuh"
#include "models/omega_z_des.hh"
#include "pipelines/systematics/shear_prj/cpp/sigma_prj_t.hh"
#include "models/z_kernel_data.hh"
#include "utils/datablock_reader.hh"
#include "utils/interp_1d.hh"
#include "utils/interp_2d.hh"
#include "utils/make_grid_points.hh"
#include "utils/make_interp_1d.hh"
#include "utils/make_interp_2d.hh"
#include "utils/module_macros.hh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>

namespace {

  inline void
  cuda_check(cudaError_t rc, char const* what)
  {
    if (rc != cudaSuccess)
      throw std::runtime_error(std::string("ShearPrjFrozenGpu: ") + what +
                               ": " + cudaGetErrorString(rc));
  }

  // One thread per (wall row, theta node): contract the single-kernel
  // miscentred NFW over the fixed GL mass nodes with the rnd and
  // frozen-cl mass weights. rho_mult (= Omega_m) is applied here — the
  // CUDA NFW_DSIGMA_MIS predates set_rho_mult.
  __global__ void
  dsmis_contract(y3_cuda::NFW_DSIGMA_MIS nfw, double rho_mult,
                 int n_rows, int n_theta_max, int n_lnm,
                 double const* lnm_x, double const* row_R,
                 int const* row_slice, int const* slice_ntheta,
                 int const* slice_theta_off, double const* theta_flat,
                 double const* slice_dao, double const* wrnd_flat,
                 double const* wcl_flat, double* out_rnd, double* out_cl)
  {
    int const tid = blockIdx.x * blockDim.x + threadIdx.x;
    int const row = tid / n_theta_max;
    int const it = tid % n_theta_max;
    if (row >= n_rows) return;
    int const k = row_slice[row];
    if (it >= slice_ntheta[k]) return;

    double const theta = theta_flat[slice_theta_off[k] + it];
    double const rmis = theta * slice_dao[k];
    double const r_perp = row_R[row];
    double srnd = 0.0, scl = 0.0;
    for (int im = 0; im != n_lnm; ++im) {
      double const ds = rho_mult * nfw(r_perp, rmis, lnm_x[im]);
      srnd += wrnd_flat[k * n_lnm + im] * ds;
      scl += wcl_flat[k * n_lnm + im] * ds;
    }
    out_rnd[row * n_theta_max + it] = srnd;
    out_cl[row * n_theta_max + it] = scl;
  }

} // namespace

class ShearPrjFrozenGpu {
public:
  using grid_t = y3_cluster::grid_t<4>;
  using grid_point_t = grid_t::value_type;
  static constexpr std::size_t n_outputs = 6;

private:
  // Configuration (frozen-class knobs and defaults).
  std::size_t N_lnm_, N_per_seg_, N_zring_, N_zouter_;
  double zt_lo_, zt_hi_, lnm_lo_, lnm_hi_, R_max_cMpch_;
  bool include_omega_z_;
  std::vector<double> lnm_x_, lnm_w_, lob_centers_;

  // Wall bookkeeping (frozen-class ctor semantics).
  std::vector<int> lzob_lb_, gp_lzob_idx_;
  std::vector<double> lzob_zob_;
  std::vector<std::vector<double>> lzob_Rs_;
  std::vector<int> gp_row_;                 // wall entry -> kernel row
  std::vector<int> row_slice_;
  std::vector<double> row_R_;

  // Host models (same immutable pieces the C++ class composes).
  std::optional<y3_cluster::HMF_t> hmf_;
  std::optional<y3_cluster::Interp2D> hmb_, xi_nl_;
  std::optional<y3_cluster::DV_DO_DZ_t> dv_do_dz_;
  std::optional<y3_cluster::OMEGA_Z_DES> omega_z_;
  std::optional<y3_cluster::Interp1D> chi_, sci_, sigma_z_;
  std::optional<y3_cuda::NFW_DSIGMA_MIS> dsigma_mis_dev_;
  double h0_{0.7};

  // Per-sample cached wall results.
  std::vector<std::array<double, 6>> results_;

  double
  chi_of(double z) const
  {
    return chi_->clamp(z) * h0_;
  }

  // Port of ShearPrjFrozenPhysics::build_z_grid_ (ring + log-|dchi|
  // wings, 40-iteration chi inversion).
  void
  build_z_grid(double zob, double chi_o, double r_excl,
               std::vector<double>& zs, std::vector<double>& wzs) const
  {
    zs.clear();
    wzs.clear();
    double const chi_fg_lo = chi_of(zt_lo_);
    double const chi_bg_hi = chi_of(zt_hi_);
    double const dz = 1.0e-3;
    double const dchi_dz = (chi_of(zob + dz) - chi_of(zob - dz)) / (2 * dz);
    double const dz_excl = r_excl / dchi_dz;
    double const ring_lo = std::max(zob - dz_excl, zt_lo_);
    double const ring_hi = std::min(zob + dz_excl, zt_hi_);

    auto invert_chi = [this](double target) {
      double lo = 0.001, hi = 2.0;
      for (int it = 0; it < 40; ++it) {
        double const mid = 0.5 * (lo + hi);
        if (chi_of(mid) < target)
          lo = mid;
        else
          hi = mid;
      }
      return 0.5 * (lo + hi);
    };

    std::vector<double> z_ring, w_ring, u_x, u_w;
    if (ring_hi > ring_lo)
      y3_cluster::p_op_detail::gl_nodes(ring_lo, ring_hi, N_zring_, z_ring, w_ring);

    std::vector<double> z_fg, w_fg, z_bg, w_bg;
    double const dis_fg_max = chi_o - chi_fg_lo;
    if (r_excl < dis_fg_max) {
      y3_cluster::p_op_detail::gl_nodes(std::log(r_excl), std::log(dis_fg_max),
                            N_zouter_, u_x, u_w);
      z_fg.resize(N_zouter_);
      w_fg.resize(N_zouter_);
      for (std::size_t i = 0; i != N_zouter_; ++i) {
        double const dis = std::exp(u_x[i]);
        double const z_i = invert_chi(chi_o - dis);
        double const ddz = (chi_of(z_i + dz) - chi_of(z_i - dz)) / (2 * dz);
        z_fg[i] = z_i;
        w_fg[i] = u_w[i] * dis / ddz;
      }
    }
    double const dis_bg_max = chi_bg_hi - chi_o;
    if (r_excl < dis_bg_max) {
      y3_cluster::p_op_detail::gl_nodes(std::log(r_excl), std::log(dis_bg_max),
                            N_zouter_, u_x, u_w);
      z_bg.resize(N_zouter_);
      w_bg.resize(N_zouter_);
      for (std::size_t i = 0; i != N_zouter_; ++i) {
        double const dis = std::exp(u_x[i]);
        double const z_i = invert_chi(chi_o + dis);
        double const ddz = (chi_of(z_i + dz) - chi_of(z_i - dz)) / (2 * dz);
        z_bg[i] = z_i;
        w_bg[i] = u_w[i] * dis / ddz;
      }
    }
    for (std::size_t i = z_fg.size(); i--;) {
      zs.push_back(z_fg[i]);
      wzs.push_back(w_fg[i]);
    }
    zs.insert(zs.end(), z_ring.begin(), z_ring.end());
    wzs.insert(wzs.end(), w_ring.begin(), w_ring.end());
    zs.insert(zs.end(), z_bg.begin(), z_bg.end());
    wzs.insert(wzs.end(), w_bg.begin(), w_bg.end());
  }

public:
  explicit ShearPrjFrozenGpu(cosmosis::DataBlock& cfg)
    : N_lnm_(cfg.has_val(module_label(), "n_lnm")
               ? cfg.view<int>(module_label(), "n_lnm") : 24)
    , N_per_seg_(cfg.has_val(module_label(), "n_per_seg")
                   ? cfg.view<int>(module_label(), "n_per_seg") : 10)
    , N_zring_(cfg.has_val(module_label(), "n_zring")
                 ? cfg.view<int>(module_label(), "n_zring") : 20)
    , N_zouter_(cfg.has_val(module_label(), "n_zouter")
                  ? cfg.view<int>(module_label(), "n_zouter") : 20)
    , zt_lo_(cfg.view<double>(module_label(), "zt_low"))
    , zt_hi_(cfg.view<double>(module_label(), "zt_high"))
    , lnm_lo_(cfg.view<double>(module_label(), "lnm_low"))
    , lnm_hi_(cfg.view<double>(module_label(), "lnm_high"))
    , R_max_cMpch_(cfg.has_val(module_label(), "R_max_cMpch")
                     ? cfg.view<double>(module_label(), "R_max_cMpch")
                     : 30.0)
    , include_omega_z_(cfg.has_val(module_label(), "include_omega_z")
                         ? cfg.view<int>(module_label(),
                                         "include_omega_z") != 0
                         : true)
  {
    y3_cluster::p_op_detail::gl_nodes(lnm_lo_, lnm_hi_, N_lnm_, lnm_x_, lnm_w_);
    if (cfg.has_val(module_label(), "lob_centers")) {
      lob_centers_ = get_vector_double(cfg, module_label(), "lob_centers");
    } else {
      auto const& d = y3_cluster::sp_detail::default_lob_centers();
      lob_centers_.assign(d.begin(), d.end());
    }

    auto const lamb = get_vector_double(cfg, module_label(), "lambda_bin");
    auto const zlo = get_vector_double(cfg, module_label(), "zo_low");
    auto const zhi = get_vector_double(cfg, module_label(), "zo_high");
    auto const rad = get_vector_double(cfg, module_label(), "radii");
    std::size_t const Ng = lamb.size();
    gp_lzob_idx_.resize(Ng);
    for (std::size_t i = 0; i != Ng; ++i) {
      int const lb = static_cast<int>(lamb[i]);
      double const zob = 0.5 * (zlo[i] + zhi[i]);
      int found = -1;
      for (std::size_t k = 0; k != lzob_lb_.size(); ++k)
        if (lzob_lb_[k] == lb && std::abs(lzob_zob_[k] - zob) < 1e-12) {
          found = int(k);
          break;
        }
      if (found < 0) {
        lzob_lb_.push_back(lb);
        lzob_zob_.push_back(zob);
        found = int(lzob_lb_.size() - 1);
      }
      gp_lzob_idx_[i] = found;
    }
    // One kernel row per wall entry, in wall order.
    gp_row_.resize(Ng);
    row_slice_.resize(Ng);
    row_R_.resize(Ng);
    for (std::size_t i = 0; i != Ng; ++i) {
      gp_row_[i] = int(i);
      row_slice_[i] = gp_lzob_idx_[i];
      row_R_[i] = rad[i];
    }
    lzob_Rs_.assign(lzob_lb_.size(), {});
    for (std::size_t i = 0; i != Ng; ++i)
      lzob_Rs_[gp_lzob_idx_[i]].push_back(rad[i]);

    dsigma_mis_dev_.emplace(4.0, 2.77533742639e+11, "single");
  }

  void
  set_sample(cosmosis::DataBlock& sample)
  {
    hmf_.emplace(sample);
    hmb_.emplace(y3_cluster::make_Interp2D(sample, "haloModel", "lnM", "z",
                                           "bias"));
    dv_do_dz_.emplace(sample);
    omega_z_.emplace(sample);
    xi_nl_.emplace(y3_cluster::make_Interp2D(sample, "xi_nl", "r", "z",
                                             "xi_nl"));
    chi_.emplace(y3_cluster::make_Interp1D(sample, "distances", "z", "d_c"));
    if (sample.has_val("average_sigma_crit_inv", "zlense"))
      sci_.emplace(y3_cluster::make_Interp1D(sample, "average_sigma_crit_inv",
                                             "zlense", "sci_average"));
    else
      sci_.reset();
    sigma_z_.emplace(y3_cluster::Interp1D(y3_cluster::z_kernel_z(),
                                          y3_cluster::z_kernel_sigma()));
    double const omm =
      sample.view<double>("cosmological_parameters", "omega_M");
    h0_ = sample.view<double>("cosmological_parameters", "h0");

    auto const b_sel_lob =
      sample.view<std::vector<double>>("b_sel_marginalised", "lob");
    auto const b_sel_zob =
      sample.view<std::vector<double>>("b_sel_marginalised", "zob");
    int const n_lob = int(b_sel_lob.size());
    int const n_zob = int(b_sel_zob.size());
    auto const& nd_s =
      sample.view<cosmosis::ndarray<double>>("b_sel_marginalised", "b_small");
    auto const& nd_l =
      sample.view<cosmosis::ndarray<double>>("b_sel_marginalised", "b_large");
    std::vector<double> const b_small(nd_s.begin(), nd_s.end());
    std::vector<double> const b_large(nd_l.begin(), nd_l.end());

    // ---- host part: per-slice grids and frozen weights (verbatim port).
    std::size_t const Nlz = lzob_lb_.size();
    std::vector<std::vector<double>> theta_k(Nlz), geom_k(Nlz), bsel_k(Nlz),
      psi_k(Nlz), wrnd_k(Nlz), wcl_k(Nlz);
    std::vector<double> dao_k(Nlz), sci_k(Nlz);

    std::vector<double> n_o(N_lnm_), b_o(N_lnm_), anchor(N_lnm_),
      hmf_row(N_lnm_), hmb_row(N_lnm_);

    for (std::size_t k = 0; k != Nlz; ++k) {
      int const lob_bin = lzob_lb_[k];
      double const zob = lzob_zob_[k];
      double const lobc = lob_centers_.at(lob_bin);
      double const chi_o = chi_of(zob);
      double const D_A_o = chi_o / (1.0 + zob);
      double const R_excl = y3_cluster::sp_detail::R_lambda(lobc) * (1.0 + zob);
      dao_k[k] = D_A_o;
      sci_k[k] = sci_ ? sci_->clamp(zob) : 0.0;

      auto const tg = y3_cluster::sp_detail::build_theta_grid(
        lobc, zob, lzob_Rs_[k], chi_o, D_A_o, R_excl, N_per_seg_,
        R_max_cMpch_, {});
      std::size_t const Nth = tg.theta.size();
      theta_k[k] = tg.theta;
      geom_k[k].resize(Nth);
      std::vector<double> cos_t(Nth);
      for (std::size_t it = 0; it != Nth; ++it) {
        cos_t[it] = std::cos(tg.theta[it]);
        geom_k[k][it] =
          tg.weight[it] * 2.0 * y3_cluster::sp_detail::PI * std::sin(tg.theta[it]);
      }

      // b_sel plateaus + sigmoid (frozen-class arithmetic).
      double Bs = 0.0, Bl = 0.0;
      {
        int j = 0;
        while (j + 1 < n_zob && b_sel_zob[j + 1] < zob) ++j;
        int const j0 = (j + 1 < n_zob) ? j : n_zob - 2;
        int const j1 = (j + 1 < n_zob) ? j + 1 : n_zob - 1;
        double f = (b_sel_zob[j1] > b_sel_zob[j0])
                     ? (zob - b_sel_zob[j0]) /
                         (b_sel_zob[j1] - b_sel_zob[j0])
                     : 0.0;
        f = std::clamp(f, 0.0, 1.0);
        Bs = (1 - f) * b_small[j0 * n_lob + lob_bin] +
             f * b_small[j1 * n_lob + lob_bin];
        Bl = (1 - f) * b_large[j0 * n_lob + lob_bin] +
             f * b_large[j1 * n_lob + lob_bin];
      }
      double const theta_lam =
        y3_cluster::sp_detail::R_lambda(lobc) * (1.0 + zob) / chi_o;
      double const k_sig = 2.5 / theta_lam;
      double const theta0 = 0.5 * theta_lam;
      bsel_k[k].resize(Nth);
      for (std::size_t it = 0; it != Nth; ++it)
        bsel_k[k][it] =
          Bs + (Bl - Bs) /
                 (1.0 + std::exp(-k_sig * (tg.theta[it] - theta0)));

      // Frozen mass shapes at zob + r_s-anchored drift denominator.
      double denom = 0.0;
      wcl_k[k].resize(N_lnm_);
      for (std::size_t im = 0; im != N_lnm_; ++im) {
        n_o[im] = (*hmf_)(lnm_x_[im], zob);
        b_o[im] = hmb_->clamp(lnm_x_[im], zob);
        double const r_200 =
          std::cbrt(3.0 * std::exp(lnm_x_[im]) /
                    (800.0 * M_PI * 2.77533742639e+11));
        anchor[im] = lnm_w_[im] * (r_200 / 4.0);
        denom += anchor[im] * n_o[im] * b_o[im];
        wcl_k[k][im] = lnm_w_[im] * n_o[im] * b_o[im];
      }
      denom += 1.0e-300;

      std::vector<double> zs, wzs;
      build_z_grid(zob, chi_o, R_excl, zs, wzs);

      wrnd_k[k].assign(N_lnm_, 0.0);
      psi_k[k].assign(Nth, 0.0);
      for (std::size_t iz = 0; iz != zs.size(); ++iz) {
        double const z = zs[iz];
        double const chi_z = chi_of(z);
        double const om = include_omega_z_ ? (*omega_z_)(z) : 1.0;
        double const sig_z = sigma_z_->clamp(z);
        double const u = (z - zob) / sig_z;
        double const wz_phot = (std::abs(u) < 1.0) ? 1.0 - u * u : 0.0;
        double const common_z = (*dv_do_dz_)(z) * om * wzs[iz] * wz_phot;
        if (common_z == 0.0) continue;

        double numer_ab = 0.0;
        for (std::size_t im = 0; im != N_lnm_; ++im) {
          hmf_row[im] = (*hmf_)(lnm_x_[im], z);
          hmb_row[im] = hmb_->clamp(lnm_x_[im], z);
          numer_ab += anchor[im] * hmf_row[im] * hmb_row[im];
        }
        double const a_b_z = numer_ab / denom;
        for (std::size_t im = 0; im != N_lnm_; ++im)
          wrnd_k[k][im] += common_z * hmf_row[im] * lnm_w_[im];

        double const th_ex =
          y3_cluster::sp_detail::theta_excl_at_z(chi_z, chi_o, R_excl);
        for (std::size_t it = 0; it != Nth; ++it) {
          if (theta_k[k][it] <= th_ex) continue;
          double const dchi = std::sqrt(std::max(
            chi_z * chi_z + chi_o * chi_o - 2.0 * chi_z * chi_o * cos_t[it],
            0.0));
          psi_k[k][it] += common_z * a_b_z * xi_nl_->clamp(dchi, zob);
        }
      }
    }

    // ---- device part: DSigma_mis cache + mass contraction, one kernel.
    int const n_rows = int(row_R_.size());
    int n_theta_max = 0;
    std::vector<int> ntheta(Nlz), toff(Nlz);
    std::vector<double> theta_flat;
    for (std::size_t k = 0; k != Nlz; ++k) {
      ntheta[k] = int(theta_k[k].size());
      toff[k] = int(theta_flat.size());
      theta_flat.insert(theta_flat.end(), theta_k[k].begin(),
                        theta_k[k].end());
      n_theta_max = std::max(n_theta_max, ntheta[k]);
    }
    std::vector<double> wrnd_flat(Nlz * N_lnm_), wcl_flat(Nlz * N_lnm_);
    for (std::size_t k = 0; k != Nlz; ++k)
      for (std::size_t im = 0; im != N_lnm_; ++im) {
        wrnd_flat[k * N_lnm_ + im] = wrnd_k[k][im];
        wcl_flat[k * N_lnm_ + im] = wcl_k[k][im];
      }

    auto dalloc = [](auto const& host, auto*& dev) {
      using T = typename std::decay_t<decltype(host)>::value_type;
      cuda_check(cudaMalloc(&dev, host.size() * sizeof(T)), "cudaMalloc");
      cuda_check(cudaMemcpy(dev, host.data(), host.size() * sizeof(T),
                            cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    };
    double *d_lnm = nullptr, *d_rowR = nullptr, *d_theta = nullptr,
           *d_dao = nullptr, *d_wrnd = nullptr, *d_wcl = nullptr,
           *d_orn = nullptr, *d_ocl = nullptr;
    int *d_rowk = nullptr, *d_nth = nullptr, *d_toff = nullptr;
    dalloc(lnm_x_, d_lnm);
    dalloc(row_R_, d_rowR);
    dalloc(theta_flat, d_theta);
    dalloc(dao_k, d_dao);
    dalloc(wrnd_flat, d_wrnd);
    dalloc(wcl_flat, d_wcl);
    dalloc(row_slice_, d_rowk);
    dalloc(ntheta, d_nth);
    dalloc(toff, d_toff);
    std::size_t const n_out = std::size_t(n_rows) * n_theta_max;
    cuda_check(cudaMalloc(&d_orn, n_out * sizeof(double)), "out rnd");
    cuda_check(cudaMalloc(&d_ocl, n_out * sizeof(double)), "out cl");

    int const threads = 256;
    int const blocks = int((n_out + threads - 1) / threads);
    dsmis_contract<<<blocks, threads>>>(
      *dsigma_mis_dev_, omm, n_rows, n_theta_max, int(N_lnm_), d_lnm,
      d_rowR, d_rowk, d_nth, d_toff, d_theta, d_dao, d_wrnd, d_wcl, d_orn,
      d_ocl);
    cuda_check(cudaGetLastError(), "kernel launch");
    cuda_check(cudaDeviceSynchronize(), "kernel sync");

    std::vector<double> srnd(n_out), scl(n_out);
    cuda_check(cudaMemcpy(srnd.data(), d_orn, n_out * sizeof(double),
                          cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    cuda_check(cudaMemcpy(scl.data(), d_ocl, n_out * sizeof(double),
                          cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    for (auto* p : {d_lnm, d_rowR, d_theta, d_dao, d_wrnd, d_wcl, d_orn,
                    d_ocl})
      cudaFree(p);
    for (auto* p : {d_rowk, d_nth, d_toff}) cudaFree(p);

    // ---- final theta dot products per wall row (host, tiny).
    results_.assign(n_rows, {});
    for (int r = 0; r != n_rows; ++r) {
      int const k = row_slice_[r];
      double acc_rnd = 0.0, acc_cl = 0.0;
      for (int it = 0; it != ntheta[k]; ++it) {
        acc_rnd += geom_k[k][it] * srnd[std::size_t(r) * n_theta_max + it];
        acc_cl += geom_k[k][it] * bsel_k[k][it] * psi_k[k][it] *
                  scl[std::size_t(r) * n_theta_max + it];
      }
      double const sci_v = sci_k[k];
      results_[r] = {acc_rnd + acc_cl, acc_rnd, acc_cl,
                     (acc_rnd + acc_cl) * sci_v, acc_rnd * sci_v,
                     acc_cl * sci_v};
    }
  }

  std::array<double, n_outputs>
  evaluate(grid_point_t const& pt) const
  {
    for (std::size_t r = 0; r != row_R_.size(); ++r) {
      if (gp_lzob_idx_[r] >= 0 &&
          lzob_lb_[row_slice_[r]] == int(pt[0]) &&
          std::abs(lzob_zob_[row_slice_[r]] - 0.5 * (pt[1] + pt[2])) <
            1e-12 &&
          std::abs(row_R_[r] - pt[3]) < 1e-12)
        return results_[r];
    }
    throw std::out_of_range(
      "ShearPrjFrozenGpu: wall point not in configured grid");
  }

  static char const* module_label() { return "ShearPrjFrozenGpu"; }

  static std::array<char const*, n_outputs>
  output_sections()
  {
    return {"dsigma_prj_frozen_gpu", "dsigma_prj_frozen_gpu",
            "dsigma_prj_frozen_gpu", "shear_prj_frozen_gpu",
            "shear_prj_frozen_gpu", "shear_prj_frozen_gpu"};
  }

  static std::array<char const*, n_outputs>
  output_names()
  {
    return {"vals", "rnd", "cl", "vals", "rnd", "cl"};
  }

  static grid_t
  make_grid_points(cosmosis::DataBlock& cfg)
  {
    return y3_cluster::make_grid_points_wall_of_numbers(
      cfg, module_label(), "lambda_bin", "zo_low", "zo_high", "radii");
  }
};

DEFINE_COSMOSIS_SCALAR_EVALUATOR_MODULE(ShearPrjFrozenGpu)
