// P_operator -- fixed-GL evaluator for the three Costanzi-2026 P[X]
// scalars (P1, I1, I2) on a wall grid of (zob_low, zob_high, lambda_bin)
// points.  Mirrors richness_selection/sel_bias.py::_P_operator.
//
// REFERENCE.  The quadrature recipe below follows the
// "Numerical pitfalls" chapter of the companion doc
//   RichnessSelection/docs/richness_selection.tex
// specifically:
//   * "The theta-axis: split at exclusion" -- per-z GL on
//     [theta_excl(z), 2*theta_lob] with NO masking of xi_NL; N_theta=10
//     reaches sub-0.01% error.
//   * "The z-axis: split by physics" -- outer z-integral split into
//     three regions, integrated on the coordinate in which each
//     region is smooth:
//       (1) ring band  [zob - dz_excl, zob + dz_excl]  GL in z,
//           N_ring = max(9, N_z/4).
//       (2) fg wing    [ln R_excl, ln(chi_o - chi(z_fg_lo))]
//           GL in u = ln|Delta_chi_parallel|, Jacobian du -> dz via
//           (d chi/dz)^{-1}.  N_outer = max(15, (N_z - N_ring)/2).
//       (3) bg wing    symmetric on the backside.
// At N_z = 80 the combined grid hits sub-0.01% on P1, I1, I2 vs
// scipy.integrate.quad with matched inner grids (doc Tab. near §zgrid).
//
// Structure (chosen to match what Python gets from numpy broadcast):
//
//   set_sample() -- grid-point independent work, once per CosmoSIS sample:
//     * build the reference zt grid (linearly spaced).
//     * build the 4 per-bin lt GL grids (one per richness bin).
//     * pre-evaluate MOR(lt_gl, lnM, zt_ref) on a (4, N_lt, N_lnm, N_zt_ref)
//       tensor -- this is the expensive Poisson-Gaussian sum, and it has
//       no dependence on (lob, zob) beyond selecting a lt_gl grid.
//     * pre-evaluate HMF(lnM, zt_ref), HMB(lnM, zt_ref), chi(zt_ref),
//       dV/dOm/dz(zt_ref), sigma_z(zt_ref) on the reference grid.
//     * copy the xi_NL flat table + log-r axis for the hand-rolled
//       bilinear lookup (GSL's Interp2D is ~10x slower for inner loops).
//
//   evaluate(pt) -- per grid point:
//     * construct the ring + fg/bg log-|dchi| zt GL nodes.
//     * interp chi/dV/sigma_z/HMF/HMB/MOR onto the zt GL nodes via plain
//       1-D linear interp on the reference grid (bracket indices shared
//       across all tensors for each iz).
//     * per zt node: cache sin_th, cos_th, sigmoid, dchi_3d, xi_th on the
//       inner theta grid, and the fA[il, it] tensor.  Reduce the theta
//       loop to three (N_lt,) contraction vectors th_sum_{P1,I1,I2}[il].
//     * final (iM, il) accumulation is a flat multiply-add -- no
//       transcendentals inside.
#ifndef Y3_CLUSTER_CPP_P_OPERATOR_T_HH
#define Y3_CLUSTER_CPP_P_OPERATOR_T_HH

#include "utils/make_grid_points.hh"

#include "cosmosis/datablock/datablock.hh"
#include "cosmosis/datablock/ndarray.hh"

#include "utils/interp_1d.hh"
#include "utils/interp_2d.hh"
#include "utils/make_interp_1d.hh"
#include "utils/make_interp_2d.hh"
#include "models/z_kernel_data.hh"

#include "models/dv_do_dz_t.hh"
#include "models/hmf_t.hh"
#include "models/mor_hod_t.hh"

#include "models/p_operator_cuhre_t.hh"  // p_op_detail:: helpers

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <vector>

namespace y3_cluster {

  class P_operator {
   public:
    using grid_t       = y3_cluster::grid_t<3>;
    using grid_point_t = grid_t::value_type;

    static constexpr std::size_t n_outputs = 3;
    static constexpr std::size_t n_bins    = 4;

    // Default GL node counts follow the sub-0.01% recipe in
    // RichnessSelection/docs/richness_selection.tex
    // §"The z-axis: split by physics": for N_z = 80 the doc
    // recommends N_ring = max(9, N_z/4) = 20 ring nodes in z, and
    // N_outer = max(15, (N_z - N_ring)/2) = 30 nodes per fg/bg
    // log|Delta_chi| wing.  N_theta = 10 (post-exclusion split)
    // and N_lnM = 24 are the doc's inner-axis defaults.
    explicit P_operator(cosmosis::DataBlock& cfg)
      : N_lt_(cfg.has_val(module_label(), "n_lt")
                ? cfg.view<int>(module_label(), "n_lt") : 60)
      , N_zring_(cfg.has_val(module_label(), "n_zring")
                   ? cfg.view<int>(module_label(), "n_zring") : 20)
      , N_zouter_(cfg.has_val(module_label(), "n_zouter")
                    ? cfg.view<int>(module_label(), "n_zouter") : 30)
      , N_lnm_(cfg.has_val(module_label(), "n_lnm")
                 ? cfg.view<int>(module_label(), "n_lnm") : 24)
      , N_th_(cfg.has_val(module_label(), "n_theta")
                ? cfg.view<int>(module_label(), "n_theta") : 10)
      , N_zt_ref_(cfg.has_val(module_label(), "n_zt_ref")
                    ? cfg.view<int>(module_label(), "n_zt_ref") : 80)
      , zt_ref_lo_(cfg.has_val(module_label(), "zt_ref_low")
                     ? cfg.view<double>(module_label(), "zt_ref_low") : 0.05)
      , zt_ref_hi_(cfg.has_val(module_label(), "zt_ref_high")
                     ? cfg.view<double>(module_label(), "zt_ref_high") : 0.90)
      , lnm_lo_(cfg.view<double>(module_label(), "lnm_low"))
      , lnm_hi_(cfg.view<double>(module_label(), "lnm_high"))
    {
      // Grid-point-independent GL nodes.
      p_op_detail::gl_nodes(lnm_lo_, lnm_hi_, N_lnm_, lnm_x_, lnm_w_);

      // Reference zt grid: linear between [zt_ref_lo_, zt_ref_hi_].
      // Chosen wider than any photo-z support so the per-grid-point ring
      // + wings always land inside.  Linear spacing is fine because
      // chi/dV/HMF are smooth in z at this resolution.
      zt_ref_.resize(N_zt_ref_);
      double const dz_ref = (zt_ref_hi_ - zt_ref_lo_) / (N_zt_ref_ - 1);
      for (std::size_t i = 0; i != N_zt_ref_; ++i)
        zt_ref_[i] = zt_ref_lo_ + i * dz_ref;
      inv_dz_ref_ = 1.0 / dz_ref;

      // Per-bin lt GL grids (lt range = (eps, lob_centre]).
      for (std::size_t b = 0; b != n_bins; ++b) {
        double const lob_b = p_op_detail::lob_center(b);
        p_op_detail::gl_nodes(1.0e-6, lob_b, N_lt_, lt_gl_[b], lt_gl_w_[b]);
      }
    }

    void
    set_sample(cosmosis::DataBlock& sample)
    {
      hmf_.emplace(sample);
      hmb_.emplace(make_Interp2D(sample, "haloModel", "lnM", "z", "bias"));
      dv_do_dz_.emplace(sample);
      mor_.emplace(sample);
      chi_.emplace(make_Interp1D(sample, "distances", "z", "d_c"));
      sigma_z_.emplace(Interp1D(y3_cluster::z_kernel_z(),
                                y3_cluster::z_kernel_sigma()));

      double const omm =
          sample.view<double>("cosmological_parameters", "omega_M");
      double const omn =
          sample.view<double>("cosmological_parameters", "omega_nu");
      // HMF_t already rescales its spline axis by (omm - omn) internally
      // (hmf_t.hh::_adjust_to_log), so the shift here must stay zero.
      ln_mass_shift_ = 0.0;
      h0_ = sample.view<double>("cosmological_parameters", "h0");

      // --- Reference zt tables ------------------------------------------
      chi_ref_.resize(N_zt_ref_);
      dv_ref_.resize(N_zt_ref_);
      sig_ref_.resize(N_zt_ref_);
      for (std::size_t iz = 0; iz != N_zt_ref_; ++iz) {
        double const z = zt_ref_[iz];
        chi_ref_[iz] = (*chi_).clamp(z) * h0_;
        dv_ref_[iz]  = (*dv_do_dz_)(z);
        sig_ref_[iz] = (*sigma_z_).clamp(z);
      }

      // --- HMF / HMB on (lnM, zt_ref) ----------------------------------
      // Layout: hmf_ref_[iz * N_lnm + iM] (row-major, z-slow, M-fast)
      // so that per-zt slice is contiguous in memory.
      hmf_ref_.assign(N_lnm_ * N_zt_ref_, 0.0);
      hmb_ref_.assign(N_lnm_ * N_zt_ref_, 0.0);
      for (std::size_t iz = 0; iz != N_zt_ref_; ++iz) {
        double const z = zt_ref_[iz];
        double* hmf_row = &hmf_ref_[iz * N_lnm_];
        double* hmb_row = &hmb_ref_[iz * N_lnm_];
        for (std::size_t iM = 0; iM != N_lnm_; ++iM) {
          hmf_row[iM] = (*hmf_)(lnm_x_[iM] + ln_mass_shift_, z);
          hmb_row[iM] = hmb_->clamp(lnm_x_[iM], z);
        }
      }

      // --- MOR(lt_gl[bin], lnM, zt_ref) on (4, N_lt, N_lnm, N_zt_ref) --
      // Layout: mor_ref_[bin][(iz * N_lnm + iM) * N_lt + il]
      //
      // Costanzi-2026 shifted-Poisson form (continuous, NOT the
      // Costanzi-2019 Poisson x Gaussian form that MOR_HOD_t
      // implements).  Mirrors richness_selection.mor.MOR.pdf:
      //
      //   l_sat(M, z) = ((M - Mmin) / (M1 - Mmin))^alpha
      //                 * ((1+z) / (1+z_pivot))^epsilon         (no central)
      //   mi          = (l_sat * sigma_intr)^2
      //   lam         = l_sat + mi
      //   x           = ltr + mi
      //   pdf(ltr)    = exp(-lam + (x - 1) ln(lam) - lgamma(x))    ltr >= 0
      //               = 0                                          ltr <  0
      //
      // This is continuous in ltr -- no Poisson-k sum, no Gaussian
      // convolution, no renormalisation.  One std::lgamma per lt node
      // per (M, z).  The lt GL grid starts at eps > 0, so the
      // ltr < 0 branch is dead and we drop it to vectorise the loop.
      double const log10_Mmin = mor_->log10_Mmin();
      double const log10_M1   = mor_->log10_M1();
      double const alpha      = mor_->alpha();
      double const epsilon    = mor_->epsilon();
      double const z_pivot    = mor_->z_pivot();
      double const sigma_intr = mor_->sigma_lambda();
      double const Mmin       = std::pow(10.0, log10_Mmin);
      double const M1         = std::pow(10.0, log10_M1);
      double const dM1        = M1 - Mmin;

      for (std::size_t bb = 0; bb != n_bins; ++bb)
        mor_ref_[bb].assign(N_lt_ * N_lnm_ * N_zt_ref_, 0.0);

      for (std::size_t iz = 0; iz != N_zt_ref_; ++iz) {
        double const z    = zt_ref_[iz];
        double const zfac = std::pow((1.0 + z) / (1.0 + z_pivot), epsilon);
        for (std::size_t iM = 0; iM != N_lnm_; ++iM) {
          double const M      = std::exp(lnm_x_[iM]);
          double const dM     = std::max(M - Mmin, 0.0);
          double const l_sat  = (dM > 0.0 && dM1 > 0.0)
                              ? std::pow(dM / dM1, alpha) * zfac
                              : 0.0;
          double const mi     = (l_sat * sigma_intr) * (l_sat * sigma_intr);
          double const lam    = std::max(l_sat + mi, 1.0e-300);
          double const log_lam = std::log(lam);

          // shifted-Poisson pdf on each lt grid, vectorized.
          for (std::size_t bb = 0; bb != n_bins; ++bb) {
            double* dst = &mor_ref_[bb][(iz * N_lnm_ + iM) * N_lt_];
            double const* lt_x_bb = lt_gl_[bb].data();
            #pragma omp simd
            for (std::size_t il = 0; il != N_lt_; ++il) {
              double const lt = lt_x_bb[il];
              double const x  = lt + mi;
              dst[il] = std::exp(-lam + (x - 1.0) * log_lam
                                 - std::lgamma(x));
            }
          }
        }
      }

      // --- xi_NL: flat table + log-r axis for hand-rolled bilinear ----
      r_xi_ = sample.view<std::vector<double>>("xi_nl", "r");
      z_xi_ = sample.view<std::vector<double>>("xi_nl", "z");
      auto xi_nd = sample.view<cosmosis::ndarray<double>>("xi_nl", "xi_nl");
      xi_flat_.assign(xi_nd.begin(), xi_nd.end());
      N_r_xi_ = r_xi_.size();
      N_z_xi_ = z_xi_.size();
      log_r0_      = std::log(r_xi_[0]);
      inv_dlog_r_  = 1.0 / std::log(r_xi_[1] / r_xi_[0]);
      z_xi_0_      = z_xi_[0];
      inv_dz_xi_   = 1.0 / (z_xi_[1] - z_xi_[0]);
    }

    std::array<double, 3>
    evaluate(grid_point_t const& pt) const
    {
      double const zob      = 0.5 * (pt[0] + pt[1]);
      int    const bin      = static_cast<int>(pt[2]);
      double const lob      = p_op_detail::lob_center(bin);
      double const chi_o    = (*chi_).clamp(zob) * h0_;
      double const theta_lob= p_op_detail::R_lambda(lob) * (1.0 + zob) / chi_o;
      double const R_excl   = p_op_detail::R_lambda(lob) * (1.0 + zob);
      double const k_sig    = 2.5 / theta_lob;
      double const th0_sig  = 0.5 * theta_lob;

      // Per-grid-point zt grid (ring + fg/bg wings).
      std::vector<double> zs, wzs;
      build_z_grid_(zob, chi_o, R_excl, zs, wzs);
      std::size_t const Nz = zs.size();

      // Pre-cache xi_NL(r, zob) as a function of r only -- one bilinear
      // in z, and subsequent lookups reduce to 1-D log-r interp.
      double const u_zob = (zob - z_xi_0_) * inv_dz_xi_;
      int    const iz_xi = std::clamp(int(std::floor(u_zob)),
                                       0, int(N_z_xi_) - 2);
      double const fz_xi = u_zob - iz_xi;
      std::vector<double> xi_r_at_zob(N_r_xi_);
      double const* xi_row0 = &xi_flat_[iz_xi       * N_r_xi_];
      double const* xi_row1 = &xi_flat_[(iz_xi + 1) * N_r_xi_];
      for (std::size_t ir = 0; ir != N_r_xi_; ++ir)
        xi_r_at_zob[ir] = xi_row0[ir] + fz_xi * (xi_row1[ir] - xi_row0[ir]);

      // Interp zt-dependent tables onto the per-grid-point zt grid.
      // Pre-compute bracket (iz_lo, frac) once per iz.
      std::vector<std::size_t> iz_lo(Nz);
      std::vector<double>      fz(Nz);
      for (std::size_t iz = 0; iz != Nz; ++iz) {
        double const u = (zs[iz] - zt_ref_lo_) * inv_dz_ref_;
        std::size_t const i = std::min<std::size_t>(
                                  std::max<std::size_t>(0, std::size_t(u)),
                                  N_zt_ref_ - 2);
        iz_lo[iz] = i;
        fz[iz]    = u - i;
      }

      // Interpolated zt-only scalars.
      std::vector<double> chi_zt(Nz), dv_zt(Nz), sig_zt(Nz);
      for (std::size_t iz = 0; iz != Nz; ++iz) {
        auto const i = iz_lo[iz]; auto const f = fz[iz];
        // chi: evaluate the distances table DIRECTLY at the z node
        // (issue #12): resampling the (piecewise-linear, possibly
        // coarse) chi table through the linear zt_ref grid distorted
        // the exclusion-ring |Delta chi| by up to 0.21 cMpc/h (17% of
        // R_excl) where a table kink lands inside a zt_ref segment
        // near zob, while chi_o was table-direct -- an inconsistent
        // mix that hit only the xi-carrying operators. dV and sigma_z
        // are smooth and enter multiplicatively, so their zt_ref
        // interpolation stays.
        chi_zt[iz] = (*chi_).clamp(zs[iz]) * h0_;
        dv_zt [iz] = dv_ref_ [i] + f * (dv_ref_ [i+1] - dv_ref_ [i]);
        sig_zt[iz] = sig_ref_[i] + f * (sig_ref_[i+1] - sig_ref_[i]);
      }

      // Interpolated HMF(lnM, iz) and HMB(lnM, iz): layout iz-fast to
      // match the inner loop access pattern in the main contraction.
      std::vector<double> hmf_z(N_lnm_ * Nz), hmb_z(N_lnm_ * Nz);
      for (std::size_t iz = 0; iz != Nz; ++iz) {
        auto const i = iz_lo[iz]; auto const f = fz[iz];
        double const* h0r = &hmf_ref_[i       * N_lnm_];
        double const* h1r = &hmf_ref_[(i + 1) * N_lnm_];
        double const* b0r = &hmb_ref_[i       * N_lnm_];
        double const* b1r = &hmb_ref_[(i + 1) * N_lnm_];
        for (std::size_t iM = 0; iM != N_lnm_; ++iM) {
          hmf_z[iM * Nz + iz] = h0r[iM] + f * (h1r[iM] - h0r[iM]);
          hmb_z[iM * Nz + iz] = b0r[iM] + f * (b1r[iM] - b0r[iM]);
        }
      }

      // Interpolated MOR(lt, lnM, iz): layout (iM, iz, il) with il fast
      // for the theta-loop inner contraction.
      std::vector<double> mor_z(N_lnm_ * Nz * N_lt_);
      auto const& mor_bin = mor_ref_[bin];
      for (std::size_t iz = 0; iz != Nz; ++iz) {
        auto const i = iz_lo[iz]; auto const f = fz[iz];
        for (std::size_t iM = 0; iM != N_lnm_; ++iM) {
          double const* m0 = &mor_bin[(i     * N_lnm_ + iM) * N_lt_];
          double const* m1 = &mor_bin[((i+1) * N_lnm_ + iM) * N_lt_];
          double* dst = &mor_z[(iM * Nz + iz) * N_lt_];
          #pragma omp simd
          for (std::size_t il = 0; il != N_lt_; ++il)
            dst[il] = m0[il] + f * (m1[il] - m0[il]);
        }
      }

      auto const& lt_x = lt_gl_[bin];
      auto const& lt_w = lt_gl_w_[bin];

      // -----------------------------------------------------------------
      // theta-outer restructure (2026-05-06).  Same math, different loop
      // order:
      //
      //   1.  ONE theta grid for this (lob, zob), split at theta_lob so
      //       the sigmoid transition and the saturated large-theta
      //       plateau each get N_th_/2 GL nodes.  All theta-only caches
      //       (sin, cos, sigmoid, omega_theta = w·2π·sin) are computed
      //       ONCE here rather than Nz times inside the z-loop.
      //
      //   2.  Pre-contract the M integral into z-contracted scalars:
      //         W_P1 [il, iz] = Σ_M  wM · n(M,z) · p_mor(lt_il | M, z)
      //         W_I  [il, iz] = Σ_M  wM · n(M,z) · b(M,z) · p_mor(...)
      //       so the outer-theta loop only does multiply-adds over W_{P1,I}.
      //
      //   3.  theta outer, z-inner, lt-innermost.  Per-z photo-z gating
      //       and per-z exclusion gating remain element-wise on theta.
      //
      // Wall-clock ~1.5-2x faster thanks to one trig pass, one gl_nodes
      // call, and a flat inner loop the compiler can vectorize.
      // -----------------------------------------------------------------

      // ---- ONE theta grid per (lob, zob) -------------------------------
      // Outer coverage: [min(theta_excl_z), 2·theta_lob].  We gate out
      // the [th, theta_excl(z)] region at each iz via a per-z mask.
      double const th_hi_global = 2.0 * theta_lob;

      // Find the minimum theta_excl over z (-> global integration lower
      // limit); if no z has exclusion, use eps.
      std::vector<double> theta_excl_z(Nz, 0.0);
      double th_lo_global = th_hi_global;
      for (std::size_t iz = 0; iz != Nz; ++iz) {
        double const chi_z = chi_zt[iz];
        double cos_excl = (chi_z*chi_z + chi_o*chi_o - R_excl*R_excl)
                        / (2.0 * chi_z * chi_o);
        cos_excl = std::clamp(cos_excl, -1.0, 1.0);
        double const th_e = (cos_excl >= 1.0 - 1e-12) ? 1e-6
                                                     : std::acos(cos_excl);
        theta_excl_z[iz] = th_e;
        if (th_e < th_lo_global) th_lo_global = th_e;
      }
      if (th_lo_global >= th_hi_global) {
        // Fully excluded across all z — integrand is zero.
        return {0.0, 0.0, 0.0};
      }

      std::vector<double> ths, wths;
      {
        std::size_t const N1 = N_th_ / 2;
        std::size_t const N2 = N_th_ - N1;
        double const th_mid = std::clamp(theta_lob, th_lo_global,
                                         th_hi_global);
        if (N1 == 0 || th_mid <= th_lo_global ||
            N2 == 0 || th_mid >= th_hi_global) {
          p_op_detail::gl_nodes(th_lo_global, th_hi_global, N_th_,
                                ths, wths);
        } else {
          std::vector<double> x1, w1, x2, w2;
          p_op_detail::gl_nodes(th_lo_global, th_mid, N1, x1, w1);
          p_op_detail::gl_nodes(th_mid,      th_hi_global, N2, x2, w2);
          ths.resize(N_th_);
          wths.resize(N_th_);
          for (std::size_t i = 0; i < N1; ++i) {
            ths[i]  = x1[i]; wths[i] = w1[i];
          }
          for (std::size_t i = 0; i < N2; ++i) {
            ths[N1 + i]  = x2[i]; wths[N1 + i] = w2[i];
          }
        }
      }

      // ---- theta-only caches (computed once) ---------------------------
      std::vector<double> sin_th(N_th_), cos_th(N_th_);
      std::vector<double> sigmoid(N_th_);
      std::vector<double> omega_th(N_th_);  // w_θ · 2π · sin(θ)
      for (std::size_t it = 0; it != N_th_; ++it) {
        sin_th [it] = std::sin(ths[it]);
        cos_th [it] = std::cos(ths[it]);
        sigmoid[it] = 1.0 / (1.0 + std::exp(-k_sig * (ths[it] - th0_sig)));
        omega_th[it] = wths[it] * 2.0 * p_op_detail::PI * sin_th[it];
      }

      // ---- z-contracted mass caches: W_P1[il, iz], W_I[il, iz] ---------
      // Build once, over the FULL z grid.  Inside the theta loop we gate
      // on (θ > theta_excl_z[iz]) and on (photo-z kernel > 0).
      //
      // Layout (il, iz) with iz fast gives contiguous inner strides for
      // the outer theta loop.
      std::vector<double> W_P1(N_lt_ * Nz, 0.0);
      std::vector<double> W_I (N_lt_ * Nz, 0.0);

      // Pre-compute the photo-z parabola wz[iz] = max(0, 1 - u²) and
      // capture the vector of active z's.
      std::vector<double> wz_vec(Nz, 0.0);
      for (std::size_t iz = 0; iz != Nz; ++iz) {
        double const u = (zs[iz] - zob) / sig_zt[iz];
        wz_vec[iz] = (std::abs(u) < 1.0) ? (1.0 - u * u) : 0.0;
      }

      // Contract the M integral.  This is the ex-inner loop, hoisted.
      for (std::size_t iz = 0; iz != Nz; ++iz) {
        if (wz_vec[iz] == 0.0) continue;   // outside photo-z support
        double const dv   = dv_zt[iz];
        double const w_wz = wzs[iz];
        double const pref_P1 = wz_vec[iz] * dv * w_wz;
        for (std::size_t il = 0; il != N_lt_; ++il) {
          double const lt_wlt = lt_x[il] * lt_w[il];
          double acc_P1 = 0.0, acc_I = 0.0;
          #pragma omp simd reduction(+:acc_P1,acc_I)
          for (std::size_t iM = 0; iM != N_lnm_; ++iM) {
            double const w_M = lnm_w_[iM];
            double const nM  = hmf_z[iM * Nz + iz];
            double const bM  = hmb_z[iM * Nz + iz];
            double const p_m = mor_z[(iM * Nz + iz) * N_lt_ + il];
            double const common = w_M * nM * p_m;
            acc_P1 += common;
            acc_I  += common * bM;
          }
          W_P1[il * Nz + iz] = pref_P1 * lt_wlt * acc_P1;
          W_I [il * Nz + iz] = pref_P1 * lt_wlt * acc_I;
        }
      }

      // ---- theta outer, z inner, lt innermost --------------------------
      // Final integrals: three scalar accumulators.  f_A and xi are the
      // only (θ, z, lt)-dependent pieces in the inner body; everything
      // else is cached.
      double sum_P1 = 0.0, sum_I1 = 0.0, sum_J = 0.0;

      for (std::size_t it = 0; it != N_th_; ++it) {
        double const theta    = ths[it];
        double const w_theta  = omega_th[it];
        double const sig_t    = sigmoid[it];
        double const oneMsig  = 1.0 - sig_t;
        double const cos_t    = cos_th[it];

        // Per-z scalars at this theta: exclusion gate, xi(dchi(θ, z)).
        // Reuse xi_r_at_zob: dchi depends on z, but xi is evaluated at
        // the single reference zob (same choice the z-outer loop used).
        double acc_P1_t = 0.0, acc_I1_t = 0.0, acc_J_t = 0.0;

        for (std::size_t iz = 0; iz != Nz; ++iz) {
          if (wz_vec[iz] == 0.0) continue;
          if (theta <= theta_excl_z[iz]) continue;

          double const chi_z = chi_zt[iz];
          double const dchi  = std::sqrt(std::max(
              chi_z*chi_z + chi_o*chi_o - 2.0*chi_z*chi_o*cos_t, 0.0));
          double const xi    = xi_lookup_(dchi, xi_r_at_zob);

          // lt-innermost contraction against W_{P1,I}[il, iz].
          // f_A depends on (theta, theta_lt(lt, z)); theta_lt is
          // monotone in lt and linear in (1+z)/chi_z.
          double sP1 = 0.0, sI1 = 0.0, sJ = 0.0;
          double const inv_chi_1pz = (1.0 + zs[iz]) / chi_z;
          double const* lt_p = lt_x.data();
          for (std::size_t il = 0; il != N_lt_; ++il) {
            double const theta_lt = p_op_detail::R_lambda(lt_p[il])
                                  * inv_chi_1pz;
            double const fA = p_op_detail::area_overlap(theta,
                                                        theta_lob,
                                                        theta_lt);
            double const wb = fA * W_P1[il * Nz + iz];
            double const wi = fA * W_I [il * Nz + iz];
            sP1 += wb;
            sI1 += wi * xi;
            sJ  += wi * xi;
          }
          // apply sigmoid weight (common to all il, so hoist out of il):
          acc_P1_t += sP1;
          acc_I1_t += sI1 * sig_t;
          acc_J_t  += sJ  * oneMsig;
        }

        sum_P1 += w_theta * acc_P1_t;
        sum_I1 += w_theta * acc_I1_t;
        sum_J  += w_theta * acc_J_t;
      }

      return {sum_P1, sum_I1, sum_J};
    }

    static char const* module_label() { return "b_sel_marg"; }

    static std::array<char const*, 3>
    output_sections()
    {
      // (P1, I1, J) where J = I2 - I1, computed directly to avoid
      // catastrophic cancellation at large theta where sigma(theta)->1.
      return {"b_sel_marg_P1", "b_sel_marg_I1", "b_sel_marg_J"};
    }

    static grid_t
    make_grid_points(cosmosis::DataBlock& cfg)
    {
      return y3_cluster::make_grid_points_wall_of_numbers(
          cfg, module_label(), "zo_low", "zo_high", "lambda_bin");
    }

   private:
    // --- Ring + fg/bg log-|Delta_chi| zt grid (unchanged from before) ---
    void
    build_z_grid_(double zob, double chi_o, double R_excl,
                  std::vector<double>& zs,
                  std::vector<double>& wzs) const
    {
      zs.clear();
      wzs.clear();

      // Solve z_fg_lo + sigma_z(z_fg_lo) = zob  and
      //       z_bg_hi - sigma_z(z_bg_hi) = zob  by bisection on [0.01, 2.0].
      // This is the exact Python convention; sigma_z evaluated at the
      // support endpoints, not at zob.
      double const z_fg_lo = bisect_photoz_fg_(zob);
      double const z_bg_hi = bisect_photoz_bg_(zob);
      double const chi_fg_lo = (*chi_).clamp(z_fg_lo) * h0_;
      double const chi_bg_hi = (*chi_).clamp(z_bg_hi) * h0_;

      double const dz = 1.0e-3;
      double const chi_plus  = (*chi_).clamp(zob + dz) * h0_;
      double const chi_minus = (*chi_).clamp(zob - dz) * h0_;
      double const dchi_dz = (chi_plus - chi_minus) / (2.0 * dz);

      double const dz_excl = R_excl / dchi_dz;
      double const z_ring_lo = std::max(zob - dz_excl, z_fg_lo);
      double const z_ring_hi = std::min(zob + dz_excl, z_bg_hi);

      std::vector<double> z_ring, w_ring;
      if (z_ring_hi > z_ring_lo)
        p_op_detail::gl_nodes(z_ring_lo, z_ring_hi, N_zring_, z_ring, w_ring);

      std::vector<double> z_fg, w_fg;
      double const dis_fg_max = chi_o - chi_fg_lo;
      if (R_excl < dis_fg_max) {
        std::vector<double> u_fg, w_u_fg;
        p_op_detail::gl_nodes(std::log(R_excl), std::log(dis_fg_max),
                              N_zouter_, u_fg, w_u_fg);
        z_fg.resize(N_zouter_);
        w_fg.resize(N_zouter_);
        for (std::size_t i = 0; i != N_zouter_; ++i) {
          double const dis = std::exp(u_fg[i]);
          double const z_i = invert_chi_(chi_o - dis);
          double const chip = (*chi_).clamp(z_i + dz) * h0_;
          double const chim = (*chi_).clamp(z_i - dz) * h0_;
          double const ddz  = (chip - chim) / (2.0 * dz);
          z_fg[i] = z_i;
          w_fg[i] = w_u_fg[i] * dis / ddz;
        }
      }

      std::vector<double> z_bg, w_bg;
      double const dis_bg_max = chi_bg_hi - chi_o;
      if (R_excl < dis_bg_max) {
        std::vector<double> u_bg, w_u_bg;
        p_op_detail::gl_nodes(std::log(R_excl), std::log(dis_bg_max),
                              N_zouter_, u_bg, w_u_bg);
        z_bg.resize(N_zouter_);
        w_bg.resize(N_zouter_);
        for (std::size_t i = 0; i != N_zouter_; ++i) {
          double const dis = std::exp(u_bg[i]);
          double const z_i = invert_chi_(chi_o + dis);
          double const chip = (*chi_).clamp(z_i + dz) * h0_;
          double const chim = (*chi_).clamp(z_i - dz) * h0_;
          double const ddz  = (chip - chim) / (2.0 * dz);
          z_bg[i] = z_i;
          w_bg[i] = w_u_bg[i] * dis / ddz;
        }
      }

      zs.reserve(z_fg.size() + z_ring.size() + z_bg.size());
      wzs.reserve(zs.capacity());
      for (std::size_t i = z_fg.size(); i--;) {
        zs.push_back(z_fg[i]);
        wzs.push_back(w_fg[i]);
      }
      for (std::size_t i = 0; i != z_ring.size(); ++i) {
        zs.push_back(z_ring[i]);
        wzs.push_back(w_ring[i]);
      }
      for (std::size_t i = 0; i != z_bg.size(); ++i) {
        zs.push_back(z_bg[i]);
        wzs.push_back(w_bg[i]);
      }
    }

    // Bisect z s.t. f(z) = 0, where f is either
    //   z + sigma_z(z) - zob   (fg endpoint: want f(z_fg_lo) = 0)
    //   z - sigma_z(z) - zob   (bg endpoint: want f(z_bg_hi) = 0)
    double
    bisect_photoz_fg_(double zob) const
    {
      double lo = 0.01, hi = zob;
      if (lo + (*sigma_z_).clamp(lo) - zob > 0.0)
        return lo;   // photoz extends below 0.01, clip
      for (int it = 0; it < 50; ++it) {
        double const mid = 0.5 * (lo + hi);
        double const f   = mid + (*sigma_z_).clamp(mid) - zob;
        if (f < 0.0) lo = mid; else hi = mid;
      }
      return 0.5 * (lo + hi);
    }

    double
    bisect_photoz_bg_(double zob) const
    {
      double lo = zob, hi = 1.5;
      if (hi - (*sigma_z_).clamp(hi) - zob < 0.0)
        return hi;
      for (int it = 0; it < 50; ++it) {
        double const mid = 0.5 * (lo + hi);
        double const f   = mid - (*sigma_z_).clamp(mid) - zob;
        if (f < 0.0) lo = mid; else hi = mid;
      }
      return 0.5 * (lo + hi);
    }

    double
    invert_chi_(double chi_target) const
    {
      double lo = 0.001, hi = 2.0;
      for (int it = 0; it < 40; ++it) {
        double const mid = 0.5 * (lo + hi);
        double const c   = (*chi_).clamp(mid) * h0_;
        if (c < chi_target) lo = mid;
        else                hi = mid;
      }
      return 0.5 * (lo + hi);
    }

    // Hand-rolled 1-D bilinear on xi_r_at_zob[] along a log-spaced r axis.
    // ~5 ns per call vs GSL's Interp2D::clamp (~80 ns).
    double
    xi_lookup_(double dchi, std::vector<double> const& xi_r) const
    {
      double const r_lo = r_xi_[0];
      double const r_hi = r_xi_.back();
      double const r    = std::clamp(dchi, r_lo, r_hi);
      double const u    = (std::log(r) - log_r0_) * inv_dlog_r_;
      int    const ir   = std::clamp(int(std::floor(u)),
                                      0, int(N_r_xi_) - 2);
      double const f    = u - ir;
      return xi_r[ir] + f * (xi_r[ir + 1] - xi_r[ir]);
    }

    // --- member data --------------------------------------------------
    std::size_t N_lt_, N_zring_, N_zouter_, N_lnm_, N_th_;
    std::size_t N_zt_ref_;
    double zt_ref_lo_, zt_ref_hi_, inv_dz_ref_;
    double lnm_lo_, lnm_hi_;

    std::vector<double> lnm_x_, lnm_w_;
    std::vector<double> zt_ref_;

    // Per-bin lt GL grids (bin-independent across samples).
    std::array<std::vector<double>, n_bins> lt_gl_;
    std::array<std::vector<double>, n_bins> lt_gl_w_;

    // Term objects (populated in set_sample).
    std::optional<y3_cluster::HMF_t>       hmf_;
    std::optional<Interp2D>                hmb_;   // haloModel/bias(lnM, z)
    std::optional<y3_cluster::DV_DO_DZ_t>  dv_do_dz_;
    std::optional<y3_cluster::MOR_HOD_t>   mor_;
    std::optional<Interp1D>                chi_;
    std::optional<Interp1D>                sigma_z_;

    // Per-sample reference tables.
    std::vector<double> chi_ref_, dv_ref_, sig_ref_;
    std::vector<double> hmf_ref_, hmb_ref_;                 // (zt_ref, lnM)
    std::array<std::vector<double>, n_bins> mor_ref_;       // per bin

    // xi_NL hand-rolled lookup state.
    std::vector<double> r_xi_, z_xi_;
    std::vector<double> xi_flat_;                           // (N_z, N_r)
    std::size_t N_r_xi_ = 0, N_z_xi_ = 0;
    double      log_r0_ = 0.0, inv_dlog_r_ = 0.0;
    double      z_xi_0_ = 0.0, inv_dz_xi_  = 0.0;

    double ln_mass_shift_ = 0.0;
    double h0_            = 1.0;
  };

}  // namespace y3_cluster

#endif
