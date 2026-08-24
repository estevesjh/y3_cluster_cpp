// Pipeline-owned fixed-GL selection-weight core (the C++ sibling of
// shared/datablock_models.py::MassZWeights, and the pipeline
// re-implementation of the production SelGLCore in
// src/models/n_operator_sel_gl_t.hh — pipeline code must not include
// production operator headers, so the physics lives here, next to the
// Python shared layer it mirrors).
//
// Both 0d lensing/counts observables share the structure
//
//   N_ij[f]  = ∫ dlnM  W_ij(lnM) · f(...)
//   W_ij(lnM) = ∫ dz  n(M,z) dV/dΩdz(z) Ω(z) S_ij(lnM,z)
//               [ · Σ_crit^-1(z), shear only ]
//
// evaluated on fixed Gauss–Legendre nodes: lnM on [lnm_low, lnm_high]
// with n_lnm (default 96) nodes, z on [zt_low, zt_high] with n_z
// (default 64). Only immutable leaf models are composed (HMF_t,
// DV_DO_DZ_t, OMEGA_Z_DES, SelFunction_t) plus the shared
// lensing_helpers Σ_crit^-1 loader. Numerically identical to the
// production core by construction (same Newton–Legendre node solver,
// same weight chain); the cross-backend identity tests certify it.
#ifndef Y3_CLUSTER_CPP_PIPELINES_SHARED_SEL_GL_WEIGHTS_HH
#define Y3_CLUSTER_CPP_PIPELINES_SHARED_SEL_GL_WEIGHTS_HH

#include "cosmosis/datablock/datablock.hh"
#include "cosmosis/datablock/ndarray.hh"

#include "models/dv_do_dz_t.hh"
#include "models/hmf_t.hh"
#include "models/omega_z_des.hh"
#include "models/sel_function_t.hh"
#include "pipelines/shared/lensing_helpers.hh"
#include "utils/interp_1d.hh"

#include <cmath>
#include <cstddef>
#include <optional>
#include <vector>

namespace y3_pipelines {

  // Gauss–Legendre nodes/weights on [a, b]: Legendre roots by Newton
  // iteration, affine-mapped — same algorithm as the Python twin
  // (numpy leggauss + affine map) to machine precision.
  inline void
  gl_nodes(double a, double b, std::size_t N,
           std::vector<double>& xs, std::vector<double>& ws)
  {
    constexpr double PI = 3.14159265358979323846;
    xs.assign(N, 0.0);
    ws.assign(N, 0.0);
    double const eps  = 1e-14;
    double const mid  = 0.5 * (a + b);
    double const hlen = 0.5 * (b - a);
    for (std::size_t i = 0; i < (N + 1) / 2; ++i) {
      double z = std::cos(PI * (double(i) + 0.75) / (double(N) + 0.5));
      double z1, pp;
      do {
        double p1 = 1.0, p2 = 0.0;
        for (std::size_t j = 0; j < N; ++j) {
          double const p3 = p2;
          p2 = p1;
          p1 = ((2.0 * double(j) + 1.0) * z * p2 - double(j) * p3) /
               (double(j) + 1.0);
        }
        pp = double(N) * (z * p1 - p2) / (z * z - 1.0);
        z1 = z;
        z  = z1 - p1 / pp;
      } while (std::abs(z - z1) > eps);
      xs[i]         = mid - hlen * z;
      xs[N - 1 - i] = mid + hlen * z;
      double const w = 2.0 * hlen / ((1.0 - z * z) * pp * pp);
      ws[i]         = w;
      ws[N - 1 - i] = w;
    }
  }

  // Number of (richness, photo-z) bins in the tabulated selection
  // function: the leading extent of sel_function/S_stack.
  inline int
  n_bins_from_block(cosmosis::DataBlock& sample)
  {
    auto const& nd = sample.view<cosmosis::ndarray<double>>(
      "sel_function", "S_stack");
    return static_cast<int>(nd.extents()[0]);
  }

  // Per-sample builder for the z-marginalised mass weight W_ij(lnM) on
  // fixed GL nodes, plus its plain (amplitude-free) lnM moments.
  class SelGlWeights {
   public:
    SelGlWeights(cosmosis::DataBlock& cfg, char const* label)
      : N_lnm_(cfg.has_val(label, "n_lnm")
                 ? cfg.view<int>(label, "n_lnm") : 96)
      , N_z_(cfg.has_val(label, "n_z")
               ? cfg.view<int>(label, "n_z") : 64)
      , zt_lo_(cfg.view<double>(label, "zt_low"))
      , zt_hi_(cfg.view<double>(label, "zt_high"))
      , lnm_lo_(cfg.view<double>(label, "lnm_low"))
      , lnm_hi_(cfg.view<double>(label, "lnm_high"))
    {
      gl_nodes(lnm_lo_, lnm_hi_, N_lnm_, lnm_x_, lnm_w_);
      gl_nodes(zt_lo_, zt_hi_, N_z_, z_x_, z_w_);
    }

    // z_amp_power: fold (1+z)^p into the z-only weight factor -- the
    // exact amplitude half of the physical-density identity
    // DSigma_phys(R|z) = (1+z)^2 DSigma_com(R (1+z)). Shear evaluators
    // pass 2 when haloModel/one_halo_physical_density is on; number
    // counts never pass it.
    void
    build_weights(cosmosis::DataBlock& s, bool include_sci,
                  double z_amp_power = 0.0)
    {
      y3_cluster::HMF_t const       hmf(s);
      y3_cluster::DV_DO_DZ_t const  dv(s);
      y3_cluster::OMEGA_Z_DES const omega(s);
      std::optional<y3_cluster::Interp1D> sci;
      if (include_sci)
        sci.emplace(load_sigma_crit_inv(s));

      // z-only factors, shared by every bin and mass node.
      std::vector<double> zfac(z_x_.size());
      for (std::size_t q = 0; q != z_x_.size(); ++q) {
        double const z = z_x_[q];
        zfac[q] = z_w_[q] * dv(z) * omega(z) *
                  (sci ? sci->clamp(z) : 1.0);
        if (z_amp_power != 0.0)
          zfac[q] *= std::pow(1.0 + z, z_amp_power);
      }

      int const n_bins = n_bins_from_block(s);
      W_.assign(n_bins, std::vector<double>(lnm_x_.size(), 0.0));
      norm_.assign(n_bins, 0.0);
      lnm_eff_.assign(n_bins, 0.0);
      mu2_.assign(n_bins, 0.0);
      z_eff_.assign(n_bins, 0.0);

      std::vector<double> Wz(lnm_x_.size());
      for (int b = 0; b != n_bins; ++b) {
        y3_cluster::SelFunction_t const sel(s, b);
        auto& Wb = W_[b];
        for (std::size_t k = 0; k != lnm_x_.size(); ++k) {
          double const lnM = lnm_x_[k];
          double acc = 0.0, acc_z = 0.0;
          for (std::size_t q = 0; q != z_x_.size(); ++q) {
            double const t = zfac[q] * hmf(lnM, z_x_[q]) * sel(lnM, z_x_[q]);
            acc += t;
            acc_z += t * z_x_[q];
          }
          Wb[k] = acc;
          Wz[k] = acc_z;
        }

        // Plain moments of lnM under W_ij (the pairing that makes the
        // moment expansion's linear term vanish).
        double n0 = 0.0, n1 = 0.0;
        for (std::size_t k = 0; k != lnm_x_.size(); ++k) {
          n0 += lnm_w_[k] * Wb[k];
          n1 += lnm_w_[k] * Wb[k] * lnm_x_[k];
        }
        norm_[b]    = n0;
        lnm_eff_[b] = (n0 != 0.0) ? n1 / n0 : 0.5 * (lnm_lo_ + lnm_hi_);
        double nz = 0.0;
        for (std::size_t k = 0; k != lnm_x_.size(); ++k)
          nz += lnm_w_[k] * Wz[k];
        z_eff_[b] = (n0 != 0.0) ? nz / n0 : 0.5 * (zt_lo_ + zt_hi_);
        double m2 = 0.0;
        for (std::size_t k = 0; k != lnm_x_.size(); ++k) {
          double const d = lnm_x_[k] - lnm_eff_[b];
          m2 += lnm_w_[k] * Wb[k] * d * d;
        }
        mu2_[b] = (n0 != 0.0) ? m2 / n0 : 0.0;
      }
    }

    std::size_t n_bins() const { return W_.size(); }
    std::vector<double> const& weights(int b) const { return W_[b]; }
    double norm(int b) const { return norm_[b]; }
    double lnm_eff(int b) const { return lnm_eff_[b]; }
    double mu2(int b) const { return mu2_[b]; }
    // Selection-weighted mean redshift per bin (physical-density rescale).
    double z_eff(int b) const { return z_eff_[b]; }
    std::vector<double> const& lnm_x() const { return lnm_x_; }
    std::vector<double> const& lnm_w() const { return lnm_w_; }
    std::vector<double> const& z_x() const { return z_x_; }
    std::vector<double> const& z_w() const { return z_w_; }

   private:
    std::size_t N_lnm_;
    std::size_t N_z_;
    double zt_lo_, zt_hi_, lnm_lo_, lnm_hi_;
    std::vector<double> lnm_x_, lnm_w_;
    std::vector<double> z_x_, z_w_;

    std::vector<std::vector<double>> W_;   // [bin][lnM node]
    std::vector<double> norm_;
    std::vector<double> lnm_eff_;
    std::vector<double> mu2_;
    std::vector<double> z_eff_;
  };

}  // namespace y3_pipelines

#endif
