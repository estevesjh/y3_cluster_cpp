// Miscentred one-halo shear via the radial_series strategy -- C++ backend.
//
// Same scientific contract as the Python backend
// (../python/shear1h_radial_series.py) and the same committed derived
// data (data/radial_series/, text export): the offline unit-profile
// tables U_ell(ln x, ln x_mis), the exact fixed-GL redshift contraction
// W_ij(lnM) (SelGLCore, reused untouched), the plain central moments of
// y = ln r_s(M), and the per-(bin, R) assembly
//
//   O_ij(R) ~= N_ij A0(ybar) [ u_mix,0 + mu2 u_mix,2 (+ mu3 u_mix,3) ]
//   u_mix,ell = (1 - f_mis) U_ell^cen + f_mis Omega_m U_ell^mis
//
// with A0(y) = 2 e^y delta_c rho_crit 1e-12 and the NFW_DSIGMA_MIS
// conventions (c = 4, rho_crit = RHOC, gamma kernel).  IMPORTANT: c = 4 is
// fixed for every mass and redshift.  There is no concentration--mass or
// concentration--redshift evolution in this profile family.  The redshift
// contraction is exact only for the population weights; it does not restore
// the missing c(M, z) dependence.  This backend is therefore a fixed-shape
// approximation and must not be treated as an exact replacement for the
// production haloModel/dSigma_nfw profile when that concentration varies.
// No derivative is recomputed inside an MCMC sample; evaluate() only
// interpolates the offline tables and restores the analytic amplitude.
//
// Differences vs the Python backend, by design:
//   - table interpolation is GSL bilinear (Interp2D/Interp1D, the
//     production convention) instead of cubic splines; the measured
//     backend difference is ~1e-4 relative (compare_backends.py),
//     far below the ~0.45% ell<=2 truncation tolerance;
//   - only vals is published (the evaluator template writes grid-shaped
//     arrays; the per-bin norm/y_eff/mu2/mu3 diagnostics of the Python
//     module have no slot in that contract).
//
// This file follows the approved plan's rules for new C++ work: it
// instantiates the immutable CosmoSISScalarEvaluatorModule template
// from a thin driver and composes the immutable SelGLCore; nothing
// under src/models or src/utils is modified.
#ifndef Y3_CLUSTER_CPP_DES_Y3_SHEAR1H_RADIAL_SERIES_T_HH
#define Y3_CLUSTER_CPP_DES_Y3_SHEAR1H_RADIAL_SERIES_T_HH

#include "cosmosis/datablock/datablock.hh"

#include "models/n_operator_sel_gl_t.hh"
#include "models/nfw_dsigma_mis.hh"
#include "utils/interp_1d.hh"
#include "utils/interp_2d.hh"
#include "utils/make_grid_points.hh"
#include "utils/read_vector.hh"

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace y3_cluster {
  namespace des_y3 {

    // Fixed-convention helpers shared with the offline generator
    // (nfw_profile_family.py).  These intentionally use c=4 for all M and z;
    // changing this to a varying concentration would invalidate the current
    // U_ell tables.  The unit test cross-checks the composite
    // amplitude against NFW_DSIGMA_MIS itself so the two cannot drift
    // apart silently.
    inline double
    delta_c_nfw()
    {
      return (200.0 * CONC * CONC * CONC / 3.0) /
             (std::log(1.0 + CONC) - CONC / (1.0 + CONC));
    }

    inline double
    y_of_lnM(double lnM)
    {
      double const r_200 =
        std::cbrt(3.0 * std::exp(lnM) / (800.0 * M_PI * RHOC));
      return std::log(r_200 / CONC);
    }

    // A0(y) in Msun/(h pc^2) per unit u: DSigma = A_sample * A0 * u.
    inline double
    A0_of_y(double y)
    {
      return 2.0 * std::exp(y) * delta_c_nfw() * RHOC * 1.0e-12;
    }

    // Loader/interpolator for the committed text tables.  Constructed
    // once at module load; the ell index selects U_ell.  Queries are
    // clamped to the table domain, matching Interp2D::clamp semantics
    // everywhere else in the pipeline.
    class RadialSeriesTable {
     public:
      explicit RadialSeriesTable(
        std::string const& stem =
          "radial_series/radial_series_nfw_mis_gamma_v1")
      {
        auto const lnx = read_vector(stem + "_lnx.txt");
        auto const lnxm = read_vector(stem + "_lnxm.txt");
        for (int ell = 0; ell != 4; ++ell) {
          std::string const tag = "_u" + std::to_string(ell);
          mis_.emplace_back(lnx, lnxm,
                            read_vector(stem + tag + "_mis.txt"));
          cen_.emplace_back(lnx, read_vector(stem + tag + "_cen.txt"));
        }
      }

      double
      u_cen(int ell, double lnx) const
      {
        return cen_[ell].clamp(lnx);
      }

      double
      u_mis(int ell, double lnx, double lnxm) const
      {
        return mis_[ell].clamp(lnx, lnxm);
      }

      double
      u_mix(int ell, double lnx, double lnxm, double f_mis,
            double rho_mult) const
      {
        return (1.0 - f_mis) * u_cen(ell, lnx) +
               f_mis * rho_mult * u_mis(ell, lnx, lnxm);
      }

      // The full series for one query: norm * A0(ybar) *
      // [u0 + mu2 u2 (+ mu3 u3)], everything in one place so the unit
      // test and the module share the exact arithmetic.
      double
      series(double R, double r_mis, double norm, double ybar, double mu2,
             double mu3, double f_mis, double rho_mult, int ell_max) const
      {
        double const lnx = std::log(R) - ybar;
        double const lnxm = std::log(r_mis) - ybar;
        double acc = u_mix(0, lnx, lnxm, f_mis, rho_mult) +
                     mu2 * u_mix(2, lnx, lnxm, f_mis, rho_mult);
        if (ell_max >= 3)
          acc += mu3 * u_mix(3, lnx, lnxm, f_mis, rho_mult);
        return norm * A0_of_y(ybar) * acc;
      }

     private:
      std::vector<Interp2D> mis_;
      std::vector<Interp1D> cen_;
    };

    // ---- Shear1hRadialSeries: the CosmoSIS evaluator ------------------
    //
    // Shear1hMisSel grid semantics: bin_index x r_perp cartesian
    // product, bin slow / R fast; richness bin = bin_index %
    // lob_centers.size().  Optional ini knobs: ell_max (2, the
    // validated default, or 3), table_stem.
    class Shear1hRadialSeries {
     public:
      using grid_t = y3_cluster::grid_t<2>;
      using grid_point_t = grid_t::value_type;
      static constexpr std::size_t n_outputs = 1;

      explicit Shear1hRadialSeries(cosmosis::DataBlock& cfg)
        : core_(cfg, module_label())
        , ell_max_(cfg.has_val(module_label(), "ell_max")
                     ? cfg.view<int>(module_label(), "ell_max")
                     : 2)
        , table_(cfg.has_val(module_label(), "table_stem")
                   ? cfg.view<std::string>(module_label(), "table_stem")
                   : "radial_series/radial_series_nfw_mis_gamma_v1")
      {
        if (ell_max_ != 2 && ell_max_ != 3)
          throw std::runtime_error(
            "Shear1hRadialSeries: ell_max must be 2 or 3");
        lob_centers_ =
          y3_cluster_sel_weights::mis_detail::read_lob_centers(
            cfg, module_label());
        if (lob_centers_.empty())
          throw std::runtime_error(
            "Shear1hRadialSeries: lob_centers is empty");
      }

      void
      set_sample(cosmosis::DataBlock& s)
      {
        namespace w = y3_cluster_sel_weights;

        core_.build_weights(s, /*include_sci=*/true);

        f_mis_ = w::mis_detail::read_mis_param(
          s, "f_mis", w::mis_detail::F_MIS_DEFAULT);
        double const tau_mis = w::mis_detail::read_mis_param(
          s, "tau_mis", w::mis_detail::TAU_MIS_DEFAULT);
        rho_mult_ = s.view<double>("cosmological_parameters", "omega_M");

        // Plain central moments of y = ln r_s(M) under each bin's
        // weight.  SelGLCore is immutable and only carries the lnM
        // moments through mu2, so the y moments (incl. mu3) are built
        // here from its public weights and nodes.
        std::size_t const nb = core_.n_bins();
        auto const& xs = core_.lnm_x();
        auto const& ws = core_.lnm_w();
        std::vector<double> y(xs.size());
        for (std::size_t k = 0; k != xs.size(); ++k)
          y[k] = y_of_lnM(xs[k]);

        norm_.assign(nb, 0.0);
        ybar_.assign(nb, 0.0);
        mu2_.assign(nb, 0.0);
        mu3_.assign(nb, 0.0);
        r_mis_.assign(nb, 0.0);
        for (std::size_t b = 0; b != nb; ++b) {
          auto const& Wb = core_.weights(b);
          double n0 = 0.0, n1 = 0.0;
          for (std::size_t k = 0; k != xs.size(); ++k) {
            n0 += ws[k] * Wb[k];
            n1 += ws[k] * Wb[k] * y[k];
          }
          norm_[b] = n0;
          ybar_[b] = (n0 != 0.0) ? n1 / n0 : y[xs.size() / 2];
          double m2 = 0.0, m3 = 0.0;
          for (std::size_t k = 0; k != xs.size(); ++k) {
            double const d = y[k] - ybar_[b];
            m2 += ws[k] * Wb[k] * d * d;
            m3 += ws[k] * Wb[k] * d * d * d;
          }
          mu2_[b] = (n0 != 0.0) ? m2 / n0 : 0.0;
          mu3_[b] = (n0 != 0.0) ? m3 / n0 : 0.0;
          r_mis_[b] = tau_mis * w::mis_detail::R_lambda(
                                  lob_centers_[b % lob_centers_.size()]);
        }
      }

      std::array<double, n_outputs>
      evaluate(grid_point_t const& pt) const
      {
        int const b = static_cast<int>(pt[0]);
        double const R = pt[1];
        if (b < 0 || static_cast<std::size_t>(b) >= core_.n_bins())
          throw std::out_of_range(
            "Shear1hRadialSeries: bin_index outside sel_function range");
        return {table_.series(R, r_mis_[b], norm_[b], ybar_[b], mu2_[b],
                              mu3_[b], f_mis_, rho_mult_, ell_max_)};
      }

      static char const*
      module_label()
      {
        return "Shear1hRadialSeries";
      }

      // NOTE: hardcoded, deliberately NOT an ini knob -- CosmoSIS
      // [DEFAULT] blocks propagate keys like output_section into every
      // module section, which would silently redirect the write.  Same
      // section as the Python backend: the two are alternative backends
      // of one stage and are never run together.
      static std::array<char const*, n_outputs>
      output_sections()
      {
        return {"shear1h_radial_series"};
      }

      static grid_t
      make_grid_points(cosmosis::DataBlock& cfg)
      {
        return y3_cluster::make_grid_points_cartesian_product(
          cfg, module_label(), "bin_index", "r_perp");
      }

     private:
      nosel_gl_detail::SelGLCore core_;
      int ell_max_;
      RadialSeriesTable table_;
      std::vector<double> lob_centers_;

      double f_mis_{y3_cluster_sel_weights::mis_detail::F_MIS_DEFAULT};
      double rho_mult_{1.0};
      std::vector<double> norm_, ybar_, mu2_, mu3_, r_mis_;
    };

  } // namespace des_y3
} // namespace y3_cluster

#endif
