#ifndef Y3_CLUSTER_CPP_P_OPERATOR_GPU_T_CUH
#define Y3_CLUSTER_CPP_P_OPERATOR_GPU_T_CUH


// Costanzi-2026 P[X] operator integrand for CUDA/PAGANI integration.
// Matches the CPU version physics:
//   - Shifted-Poisson MOR (not skewed-Gaussian)
//   - Table-based photo-z sigma (not polynomial fit)
//   - Per-bin lt range [0, lob_center]
//   - Theta grid split at theta_lob
//
// Integration: 3D over (lt, zt, lnM) with inner GL loop for theta.
// Grid: (zo_low, zo_high, lambda_bin)
//
// Template parameter WeightF selects which scalar is integrated:
//   P1: F = 1
//   I1: F = b(M,z) * xi_NL * sigmoid(theta)
//   I2: F = b(M,z) * xi_NL
//   J:  F = b(M,z) * xi_NL * (1 - sigmoid)  [= I2 - I1, computed directly]


#include "utils/make_grid_points.hh"
#include "utils/datablock_reader.hh"

#include "cosmosis/datablock/datablock.hh"
#include "common/cuda/Volume.cuh"

#include "models/dv_do_dz_t.cuh"
#include "models/hmf_t.cuh"
#include "models/mor_sat_only_t.cuh"   // instead of mor_des_log_t.cuh
#include "models/sigma_photoz_table_t.cuh"    // instead of sigma_photoz_des.cuh

#include "utils/make_interp_1d.cuh"
#include "utils/make_interp_2d.cuh"

#include <cmath>
#include <optional>
#include <vector>

namespace y3_cuda {

  namespace p_op_gpu_detail {

    static constexpr double PI = 3.14159265358979323846;

    // GL node counts
    static constexpr int GL_N_SINGLE = 10;
    static constexpr int GL_N_HALF = 5;

    __host__ __device__ inline double
    R_lambda(double lam)
    {
      return pow(lam / 100.0, 0.2);
    }

    // DES Y3 richness bin centres: [20,30], [30,45], [45,60], [60,200]
    __host__ __device__ inline double
    lob_center(int bin)
    {
      double const centres[4] = {25.0, 37.5, 52.5, 130.0};
      return (bin >= 0 && bin < 4) ? centres[bin] : 25.0;
    }

    // Two-disk fractional overlap f_A(theta; theta_lob, theta_lt)
    __host__ __device__ inline double
    area_overlap(double theta, double theta_lob, double theta_lt)
    {
      if (theta > theta_lob + theta_lt) return 0.0;

      double const diff = fabs(theta_lob - theta_lt);
      if (theta <= diff) {
        if (theta_lt <= theta_lob) return 1.0;
        double const r = theta_lob / theta_lt;
        return r * r;
      }

      double const tt = theta, ll = theta_lt, lo = theta_lob;
      double c1 = (tt*tt + ll*ll - lo*lo) / (2.0 * tt * ll);
      double c2 = (tt*tt + lo*lo - ll*ll) / (2.0 * tt * lo);
      c1 = fmax(-1.0, fmin(1.0, c1));
      c2 = fmax(-1.0, fmin(1.0, c2));

      double sq = (-tt + ll + lo) * (tt + ll - lo)
                * ( tt - ll + lo) * (tt + ll + lo);
      sq = fmax(sq, 0.0);

      double const A = ll*ll * acos(c1)
                     + lo*lo * acos(c2)
                     - 0.5   * sqrt(sq);
      return A / (PI * ll * ll);
    }

    // Per-z exclusion angle: cos(theta_excl) = (chi_z^2 + chi_o^2 - R_excl^2)/(2*chi_z*chi_o)
    __host__ __device__ inline double
    theta_excl_at_z(double chi_z, double chi_o, double R_excl)
    {
      double const denom = 2.0 * chi_z * chi_o;
      if (denom < 1e-30) return 0.0;
      double cos_excl = (chi_z*chi_z + chi_o*chi_o - R_excl*R_excl) / denom;
      cos_excl = fmax(-1.0, fmin(1.0, cos_excl));
      if (cos_excl >= 1.0 - 1e-12) return 1e-6;
      return acos(cos_excl);
    }

  }  // namespace p_op_gpu_detail


  // =====================================================================
  // GPU integrand template for P[X] operators
  // =====================================================================
  template <typename WeightF>
  class P_operator_gpu {
  public:
    using grid_t       = y3_cluster::grid_t<3>;  // (zo_low, zo_high, lambda_bin)
    using grid_point_t = grid_t::value_type;

  private:
    using volume_t = quad::Volume<double, 3>;    // (lt, zt, lnM)

    std::optional<y3_cuda::HMF_t>                hmf_;
    std::optional<y3_cuda::DV_DO_DZ_t>           dv_do_dz_;
    std::optional<y3_cuda::MOR_SAT_ONLY_t> mor_;  // Correct MOR

    std::optional<quad::Interp2D> hmb_;     // haloModel/bias(lnM, z)
    std::optional<quad::Interp2D> xi_nl_;   // xi_nl(r, z)
    std::optional<quad::Interp1D> chi_;     // distances/d_c(z)

    // Current grid point state
    double zob_low_     = 0.0;
    double zob_high_    = 0.0;
    double zob_         = 0.0;
    int    current_bin_ = 0;
    double lob_center_  = 25.0;  // Per-bin lt upper bound

    double h0_ = 1.0;

  public:
    explicit P_operator_gpu(cosmosis::DataBlock& /*cfg*/)
    {
      // No special setup needed
    }

    void
    set_sample(cosmosis::DataBlock& sample)
    {
      hmf_.emplace(sample);
      dv_do_dz_.emplace(sample);
      mor_.emplace(sample);  // uses MOR_SAT_ONLY_t

      // Interpolators from datablock
      hmb_.emplace(make_Interp2D(sample, "haloModel", "lnM", "z", "bias"));
      xi_nl_.emplace(make_Interp2D(sample, "xi_nl", "r", "z", "xi_nl"));
      chi_.emplace(make_Interp1D(sample, "distances", "z", "d_c"));

      h0_ = sample.view<double>("cosmological_parameters", "h0");
    }

    void
    set_grid_point(grid_point_t const& pt)
    {
      zob_low_     = pt[0];
      zob_high_    = pt[1];
      zob_         = 0.5 * (zob_low_ + zob_high_);
      current_bin_ = static_cast<int>(pt[2]);
      lob_center_  = p_op_gpu_detail::lob_center(current_bin_);
    }

    size_t
    get_device_mem_footprint()
    {
      size_t size = 0;
      if (hmf_) size += hmf_->get_device_mem_footprint();
      if (dv_do_dz_) size += dv_do_dz_->get_device_mem_footprint();
      return size;
    }

    // Integrand at (lt, zt, lnM) -- inner theta loop via fixed GL quadrature
    __host__ __device__ double
    operator()(double lt, double zt, double lnM) const
    {
      using namespace p_op_gpu_detail;

      if (lt <= 0.0 || lt > lob_center_) {
        return 0.0;
      }

    
      SIGMA_PHOTOZ_TABLE_t sigma_z_model;
      double const s = sigma_z_model(zt);
      double const u = (zt - zob_) / s;
      if (fabs(u) >= 1.0) return 0.0;
      double const wz = 1.0 - u * u;

      // Outer weights (independent of theta)
      double const hmf_val = (*hmf_)(lnM, zt);
      double const dv_val  = (*dv_do_dz_)(zt);
      double const mor_val = (*mor_)(lt, lnM, zt);
      double const common  = hmf_val * dv_val * mor_val;

      if (common <= 0.0) return 0.0;

      // Geometry for inner theta integral
      double const lob       = lob_center_;
      double const chi_o     = chi_->clamp(zob_) * h0_;   // cMpc/h
      double const chi_z     = chi_->clamp(zt) * h0_;     // cMpc/h
      double const theta_lob = R_lambda(lob) * (1.0 + zob_) / chi_o;
      double const R_excl    = R_lambda(lob) * (1.0 + zob_);

      // theta_excl(zt) - per-z exclusion angle
      double const th_excl = theta_excl_at_z(chi_z, chi_o, R_excl);
      double const th_hi   = 2.0 * theta_lob;
      if (th_excl >= th_hi) return 0.0;

      // Inner theta integral via fixed GL quadrature
      double const theta_lt = R_lambda(lt) * (1.0 + zt) / chi_z;
      double const k_sig    = 2.5 / theta_lob;
      double const th0_sig  = 0.5 * theta_lob;


      // GL nodes and weights for N=5 - embedded for device compatibility
      const double gl_x5[5] = {
        -0.9061798459386640, -0.5384693101056831, 0.0,
         0.5384693101056831,  0.9061798459386640
      };
      const double gl_w5[5] = {
        0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
        0.4786286704993665, 0.2369268850561891
      };

      double th_sum = 0.0;

      // Clamp split point to valid range
      double const th_mid = fmax(th_excl, fmin(theta_lob, th_hi));

      // Panel 1: [th_excl, th_mid] (if th_mid > th_excl)
      if (th_mid > th_excl + 1e-10) {
        double const mid1  = 0.5 * (th_excl + th_mid);
        double const hlen1 = 0.5 * (th_mid - th_excl);
        for (int it = 0; it < GL_N_HALF; ++it) {
          double const th     = mid1 + hlen1 * gl_x5[it];
          double const w_gl   = hlen1 * gl_w5[it];
          double const sin_th = sin(th);
          double const fA     = area_overlap(th, theta_lob, theta_lt);
          if (fA <= 0.0) continue;

          double const cos_th  = cos(th);
          double const dchi_3d = sqrt(fmax(
              chi_z*chi_z + chi_o*chi_o - 2.0*chi_z*chi_o*cos_th, 0.0));
          double const xi_val  = xi_nl_->clamp(dchi_3d, zob_);
          double const sigmoid = 1.0 / (1.0 + exp(-k_sig * (th - th0_sig)));
          double const w_th    = w_gl * 2.0 * PI * sin_th;

          th_sum += w_th * fA * WeightF::weight(xi_val, sigmoid);
        }
      }

      // Panel 2: [th_mid, th_hi] (if th_hi > th_mid)
      if (th_hi > th_mid + 1e-10) {
        double const mid2  = 0.5 * (th_mid + th_hi);
        double const hlen2 = 0.5 * (th_hi - th_mid);
        for (int it = 0; it < GL_N_HALF; ++it) {
          double const th     = mid2 + hlen2 * gl_x5[it];
          double const w_gl   = hlen2 * gl_w5[it];
          double const sin_th = sin(th);
          double const fA     = area_overlap(th, theta_lob, theta_lt);
          if (fA <= 0.0) continue;

          double const cos_th  = cos(th);
          double const dchi_3d = sqrt(fmax(
              chi_z*chi_z + chi_o*chi_o - 2.0*chi_z*chi_o*cos_th, 0.0));
          double const xi_val  = xi_nl_->clamp(dchi_3d, zob_);
          double const sigmoid = 1.0 / (1.0 + exp(-k_sig * (th - th0_sig)));
          double const w_th    = w_gl * 2.0 * PI * sin_th;

          th_sum += w_th * fA * WeightF::weight(xi_val, sigmoid);
        }
      }

      // Multiply by bias if needed
      double const b_mz = WeightF::uses_bias ? hmb_->clamp(lnM, zt) : 1.0;
      return wz * lt * common * b_mz * th_sum;
    }

    static char const* module_label() { return WeightF::module_label(); }

    // Create 3D integration volumes from ini config
    // NOTE: lt_high should be set to max(lob_centers) = 130 in the ini.
    // The integrand will return 0 for lt > lob_center[bin].
    static std::vector<volume_t>
    make_integration_volumes(cosmosis::DataBlock& cfg)
    {
      char const* label = module_label();

      double lt_low  = cfg.view<double>(label, "lt_low");
      double lt_high = cfg.view<double>(label, "lt_high");
      double zt_low  = cfg.view<double>(label, "zt_low");
      double zt_high = cfg.view<double>(label, "zt_high");
      double lnm_low  = cfg.view<double>(label, "lnm_low");
      double lnm_high = cfg.view<double>(label, "lnm_high");

      volume_t vol;
      vol.lows[0]  = lt_low;   vol.highs[0] = lt_high;
      vol.lows[1]  = zt_low;   vol.highs[1] = zt_high;
      vol.lows[2]  = lnm_low;  vol.highs[2] = lnm_high;

      return {vol};
    }

    // Create grid points from ini config
    static grid_t
    make_grid_points(cosmosis::DataBlock& cfg)
    {
      char const* label = module_label();

      auto zo_low_vec  = get_vector_double(cfg, label, "zo_low");
      auto zo_high_vec = get_vector_double(cfg, label, "zo_high");
      auto bin_vec     = get_vector_double(cfg, label, "lambda_bin");

      size_t n = zo_low_vec.size();
      if (zo_high_vec.size() != n || bin_vec.size() != n) {
        throw std::runtime_error(
          std::string(label) + ": grid arrays must have same length");
      }

      grid_t result;
      result.set_names({"zo_low", "zo_high", "lambda_bin"});
      for (size_t i = 0; i < n; ++i) {
        result.push_back({zo_low_vec[i], zo_high_vec[i], bin_vec[i]});
      }
      return result;
    }
  };

  // ---------------------------------------------------------------------
  // Weight functors for GPU 
  // ---------------------------------------------------------------------

  struct BSelMargP1GPUWeight {
    static char const* module_label() { return "bSelMargP1GPU"; }
    static constexpr bool uses_bias = false;
    __host__ __device__ static double weight(double /*xi*/, double /*sigmoid*/)
    { return 1.0; }
  };

  struct BSelMargI1GPUWeight {
    static char const* module_label() { return "bSelMargI1GPU"; }
    static constexpr bool uses_bias = true;
    __host__ __device__ static double weight(double xi, double sigmoid)
    { return xi * sigmoid; }
  };

  struct BSelMargI2GPUWeight {
    static char const* module_label() { return "bSelMargI2GPU"; }
    static constexpr bool uses_bias = true;
    __host__ __device__ static double weight(double xi, double /*sigmoid*/)
    { return xi; }
  };

  // J = I2 - I1, computed directly to avoid cancellation
  struct BSelMargJGPUWeight {
    static char const* module_label() { return "bSelMargJGPU"; }
    static constexpr bool uses_bias = true;
    __host__ __device__ static double weight(double xi, double sigmoid)
    { return xi * (1.0 - sigmoid); }
  };

}  // namespace y3_cuda

#endif
