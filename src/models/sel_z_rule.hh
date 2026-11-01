// SelZRule -- the true-redshift quadrature rule shared by the fixed-GL
// selection-weighted evaluators (SelGLCore, SelGlWeights, Shear1h2hMax and
// its CUDA twin), and SelFunctionZ_t, the S_ij(lnM, z) reader that goes
// with it.
//
// The z integrand n(M,z) dV/dz Omega(z) S_ij(lnM,z) [Sigma_crit^-1(z)] is
// NOT smooth on [zt_low, zt_high]:
//
//   S_ij = S_i(lnM, z) * S_j(z) * 1[z outside (zt_excl_low, zt_excl_high)]
//   S_j  = Phi((zob_max - z)/sigma_z) - Phi((zob_min - z)/sigma_z)
//
// steps at every photo-z bin edge over a width sigma_z (0.001 for the
// true-z Buzzard bins, a jump in practice), drops to zero across the
// simulation-box seam, and OMEGA_Z_DES is piecewise at 0.504 and 0.7. A
// single Gauss-Legendre panel over [zt_low, zt_high] converges slowly and
// non-monotonically across steps (GL-64: +3.4%/-1.9%/+0.8% in the three
// Buzzard z bins). Sampling the step on sel_function's linear z table and
// interpolating it bilinearly adds an O(dz) ramp error that no node count
// removes (~1% in z2 at n_z_shared = 128).
//
// The "composite" rule (default) therefore
//   1. reads the factorised selection published by sel_function
//      (S_rich_stack = S_i alone, smooth in z, plus the closed-form S_j
//      parameters and the seam), and evaluates S_j and the seam exactly
//      at every node -- never interpolating a step;
//   2. integrates z with composite Gauss-Legendre, n_z_panel nodes per
//      panel, with panel breaks at every bin edge's step zob +- L sigma_z
//      (L = z_step_width), at the seam edges and at the OMEGA_Z_DES
//      breakpoints, over the union of the bins' supports
//      [zob_min - L sigma_z, zob_max + L sigma_z] clipped to
//      [zt_low, zt_high]; panels inside the seam are dropped.
//   3. interpolates S_rich with the tensor-product natural cubic spline
//      of its table (sel_interp = spline, default), evaluated directly on
//      the (lnM node x z node) grid in two 1-D passes (tabulate()). With
//      the step factored out S_rich is smooth in (lnM, z), and bilinear
//      kinks at every table cell were the next error: 1.6e-3 in the
//      60-200 bins at the production n_lnm = 192, sigma_z = 0.01; the
//      spline brings it to ~3e-4 at no measurable cost (the two passes
//      are O(table + nodes) per bin, cheaper than pointwise bilinear).
// Each step then sits mid-panel and hands its neighbours a Phi(-L)
// remnant; every panel integrand is smooth. Same construction as the
// lambda_tr rule of CLensPy PR #10 (estevesjh/CLensPy), there applied to
// the richness-bin steps.
//
// Measured on the Buzzard truth point (OUTPUT/zint_fix_*): at defaults
// (8 nodes/panel, L = 5) every bin is within 5e-4 of a dense reference
// of the same integrand for sigma_z = 0.001, 0.01, 0.03 and
// 0.01 (1 + z), with 56-96 z nodes vs the global rule's 64.
//
// Fallback: z_rule = global, or a block without the factorised keys
// (an older sel_function), gives the previous single GL panel of n_z
// nodes on the interpolated S_stack table, bit for bit.
//
// Ini knobs (evaluator section): z_rule (composite | global, default
// composite), n_z_panel (8), z_step_width (5.0), sel_interp (spline |
// bilinear, composite only), n_z (64, global only).
#ifndef Y3_CLUSTER_CPP_SEL_Z_RULE_HH
#define Y3_CLUSTER_CPP_SEL_Z_RULE_HH

#include "cosmosis/datablock/datablock.hh"
#include "cosmosis/datablock/ndarray.hh"

#include "models/omega_z_des.hh"
#include "models/sel_function_t.hh"
#include "utils/interp_2d.hh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace y3_cluster {

  namespace sel_z_detail {

    // Gauss-Legendre rule on [-1, 1] (Newton on P_N; same solver as the
    // evaluators' gl_nodes), cached by order. Returns the nodes and the
    // weight denominators (1 - x^2) P_N'(x)^2, so that a panel's weights
    // 2 h / den round exactly as gl_nodes' do (global rule bit for bit).
    inline std::pair<std::vector<double>, std::vector<double>> const&
    gl_unit(std::size_t N)
    {
      static std::map<std::size_t,
                      std::pair<std::vector<double>, std::vector<double>>>
        cache;
      auto it = cache.find(N);
      if (it != cache.end())
        return it->second;
      constexpr double PI = 3.14159265358979323846;
      std::vector<double> xs(N), ws(N);
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
          z = z1 - p1 / pp;
        } while (std::abs(z - z1) > 1e-14);
        xs[i] = -z;
        xs[N - 1 - i] = z;
        double const den = (1.0 - z * z) * pp * pp;
        ws[i] = den;
        ws[N - 1 - i] = den;
      }
      return cache.emplace(N, std::make_pair(xs, ws)).first->second;
    }

    // Natural cubic spline through (x, y_r) for many rows r sharing x:
    // second derivatives by the Thomas algorithm, the factorisation
    // shared by every row. The numpy twin (datablock_models.
    // NaturalSplineGrid) performs the same operations in the same order.
    class NaturalSplineGrid {
    public:
      explicit NaturalSplineGrid(std::vector<double> x) : x_(std::move(x))
      {
        std::size_t const n = x_.size();
        if (n < 3)
          throw std::invalid_argument("NaturalSplineGrid: needs >= 3 knots");
        h_.resize(n - 1);
        for (std::size_t i = 0; i + 1 < n; ++i)
          h_[i] = x_[i + 1] - x_[i];
        // Interior system rows i = 1..n-2: a M_{i-1} + b M_i + c M_{i+1}.
        cp_.assign(n, 0.0);
        den_.assign(n, 1.0);
        for (std::size_t i = 1; i + 1 < n; ++i) {
          double const a = h_[i - 1], b = 2.0 * (h_[i - 1] + h_[i]);
          den_[i] = b - a * cp_[i - 1];
          cp_[i] = h_[i] / den_[i];
        }
      }

      // Second derivatives of the row y (length n).
      std::vector<double>
      moments(double const* y) const
      {
        std::size_t const n = x_.size();
        std::vector<double> dp(n, 0.0), m(n, 0.0);
        for (std::size_t i = 1; i + 1 < n; ++i) {
          double const r = 6.0 * ((y[i + 1] - y[i]) / h_[i] -
                                  (y[i] - y[i - 1]) / h_[i - 1]);
          dp[i] = (r - h_[i - 1] * dp[i - 1]) / den_[i];
        }
        for (std::size_t i = n - 2; i >= 1; --i) {
          m[i] = dp[i] - cp_[i] * m[i + 1];
          if (i == 1)
            break;
        }
        return m;
      }

      // Cell index and the four weights of each query point (clamped to
      // the knot range): value = wa y_i + wb y_{i+1} + wc M_i + wd M_{i+1}.
      struct Stencil {
        std::size_t i;
        double wa, wb, wc, wd;
      };
      std::vector<Stencil>
      stencils(std::vector<double> const& xq) const
      {
        std::size_t const n = x_.size();
        std::vector<Stencil> out(xq.size());
        for (std::size_t k = 0; k != xq.size(); ++k) {
          double const x = std::clamp(xq[k], x_.front(), x_.back());
          std::size_t i =
            std::upper_bound(x_.begin(), x_.end(), x) - x_.begin();
          i = std::min(std::max<std::size_t>(i, 1), n - 1) - 1;
          double const h = h_[i];
          double const A = (x_[i + 1] - x) / h, B = (x - x_[i]) / h;
          double const h26 = h * h / 6.0;
          out[k] = {i, A, B, (A * A * A - A) * h26, (B * B * B - B) * h26};
        }
        return out;
      }

      std::vector<double> const& x() const { return x_; }

    private:
      std::vector<double> x_, h_, cp_, den_;
    };

    inline double
    phi(double x)
    {
      return 0.5 * std::erfc(-x / std::sqrt(2.0));
    }

  } // namespace sel_z_detail

  // Closed-form true-z factor of every bin, as sel_function publishes it.
  struct SelZBins {
    std::vector<double> zob_min, zob_max, sigma_z;
    bool one_plus_z = false;
    double excl_lo = 0.0, excl_hi = -1.0;

    // True when the block carries the factorised selection.
    static bool
    available(cosmosis::DataBlock& s)
    {
      return s.has_val("sel_function", "S_rich_stack") &&
             s.has_val("sel_function", "sigma_z") &&
             s.has_val("sel_function", "zob_min") &&
             s.has_val("sel_function", "zob_max");
    }

    static SelZBins
    from_block(cosmosis::DataBlock& s)
    {
      using doubles = std::vector<double>;
      SelZBins out;
      out.zob_min = s.view<doubles>("sel_function", "zob_min");
      out.zob_max = s.view<doubles>("sel_function", "zob_max");
      out.sigma_z = s.view<doubles>("sel_function", "sigma_z");
      if (out.zob_max.size() != out.zob_min.size() ||
          out.sigma_z.size() != out.zob_min.size())
        throw std::runtime_error(
          "SelZBins: sel_function zob_min/zob_max/sigma_z sizes differ");
      if (s.has_val("sel_function", "sigma_z_one_plus_z"))
        out.one_plus_z = s.view<int>("sel_function", "sigma_z_one_plus_z");
      if (s.has_val("sel_function", "zt_excl_low"))
        out.excl_lo = s.view<double>("sel_function", "zt_excl_low");
      if (s.has_val("sel_function", "zt_excl_high"))
        out.excl_hi = s.view<double>("sel_function", "zt_excl_high");
      return out;
    }

    bool has_excl() const { return excl_hi > excl_lo; }

    double
    width(int b, double z) const
    {
      return sigma_z[b] * (one_plus_z ? 1.0 + z : 1.0);
    }

    // S_j(z) * 1[z outside the seam], exactly.
    double
    operator()(int b, double z) const
    {
      if (has_excl() && z > excl_lo && z < excl_hi)
        return 0.0;
      double const w = width(b, z);
      return sel_z_detail::phi((zob_max[b] - z) / w) -
             sel_z_detail::phi((zob_min[b] - z) / w);
    }

    // Composite-rule panels: see the header comment.
    std::vector<std::pair<double, double>>
    panels(double zt_lo, double zt_hi, double L) const
    {
      std::vector<double> br;
      double lo = zt_hi, hi = zt_lo;
      for (std::size_t b = 0; b != zob_min.size(); ++b) {
        int const bi = static_cast<int>(b);
        double const wl = L * width(bi, zob_min[b]);
        double const wh = L * width(bi, zob_max[b]);
        br.insert(br.end(), {zob_min[b] - wl, zob_min[b] + wl,
                             zob_max[b] - wh, zob_max[b] + wh});
        lo = std::min(lo, zob_min[b] - wl);
        hi = std::max(hi, zob_max[b] + wh);
      }
      lo = std::max(lo, zt_lo);
      hi = std::min(hi, zt_hi);
      if (!(hi > lo))
        return {};
      if (has_excl())
        br.insert(br.end(), {excl_lo, excl_hi});
      for (double const z : OMEGA_Z_DES::breakpoints)
        br.push_back(z);
      br.push_back(lo);
      br.push_back(hi);
      for (double& z : br)
        z = std::clamp(z, lo, hi);
      std::sort(br.begin(), br.end());
      std::vector<std::pair<double, double>> out;
      for (std::size_t i = 0; i + 1 < br.size(); ++i) {
        double const a = br[i], c = br[i + 1];
        if (!(c - a > 1e-12))
          continue;
        if (has_excl() && a >= excl_lo - 1e-12 && c <= excl_hi + 1e-12)
          continue;
        out.emplace_back(a, c);
      }
      return out;
    }
  };

  // S_ij(lnM, z) for one bin: the factorised form when available (smooth
  // S_rich table x exact S_j x seam), else the interpolated S_stack table.
  class SelFunctionZ_t {
  public:
    SelFunctionZ_t(cosmosis::DataBlock& s, int bin, SelZBins const* zbins,
                   bool bicubic = true)
      : bin_(bin), zbins_(zbins)
    {
      if (zbins_)
        rich_.emplace_back(make_rich(s, bin, bicubic));
      else
        table_.emplace_back(s, bin);
    }

    double
    operator()(double lnM, double zt) const
    {
      if (!zbins_)
        return table_.front()(lnM, zt);
      if (lnM < lnm_lo_ || lnM > lnm_hi_ || zt < z_lo_ || zt > z_hi_)
        return 0.0;
      double const sz = (*zbins_)(bin_, zt);
      return sz == 0.0 ? 0.0 : rich_.front().clamp(lnM, zt) * sz;
    }

  private:
    int bin_;
    SelZBins const* zbins_;
    std::vector<SelFunction_t> table_;
    std::vector<Interp2D> rich_;
    double lnm_lo_{}, lnm_hi_{}, z_lo_{}, z_hi_{};

    Interp2D
    make_rich(cosmosis::DataBlock& s, int bin, bool bicubic)
    {
      using doubles = std::vector<double>;
      auto const& lnm = s.view<doubles>("sel_function", "lnM");
      auto const& z = s.view<doubles>("sel_function", "z");
      auto const& nd =
        s.view<cosmosis::ndarray<double>>("sel_function", "S_rich_stack");
      auto ext = nd.extents();
      if (ext.size() != 3 || ext[1] != z.size() || ext[2] != lnm.size())
        throw std::runtime_error(
          "sel_function/S_rich_stack must be (n_bins, n_z, n_lnm)");
      if (bin < 0 || static_cast<std::size_t>(bin) >= ext[0])
        throw std::runtime_error("sel_function: bin_index out of range");
      lnm_lo_ = *std::min_element(lnm.begin(), lnm.end());
      lnm_hi_ = *std::max_element(lnm.begin(), lnm.end());
      z_lo_ = *std::min_element(z.begin(), z.end());
      z_hi_ = *std::max_element(z.begin(), z.end());
      std::size_t const plane = ext[1] * ext[2];
      std::vector<double> vals(nd.begin() + bin * plane,
                               nd.begin() + (bin + 1) * plane);
      return Interp2D(lnm, z, vals, bicubic);
    }
  };

  class SelZRule {
  public:
    SelZRule(cosmosis::DataBlock& cfg, char const* label)
      : n_z_(cfg.has_val(label, "n_z") ? cfg.view<int>(label, "n_z") : 64)
      , n_panel_(cfg.has_val(label, "n_z_panel")
                   ? cfg.view<int>(label, "n_z_panel") : 8)
      , L_(cfg.has_val(label, "z_step_width")
             ? cfg.view<double>(label, "z_step_width") : 5.0)
      , zt_lo_(cfg.view<double>(label, "zt_low"))
      , zt_hi_(cfg.view<double>(label, "zt_high"))
      , label_(label)
    {
      std::string const rule = cfg.has_val(label, "z_rule")
                                 ? cfg.view<std::string>(label, "z_rule")
                                 : "composite";
      if (rule != "composite" && rule != "global")
        throw std::invalid_argument(std::string(label) +
                                    ": z_rule must be composite or global");
      composite_ = (rule == "composite");
      std::string const interp =
        cfg.has_val(label, "sel_interp")
          ? cfg.view<std::string>(label, "sel_interp") : "spline";
      if (interp != "spline" && interp != "bilinear")
        throw std::invalid_argument(
          std::string(label) + ": sel_interp must be spline or bilinear");
      spline_ = (interp == "spline");
      if (n_z_ < 1 || n_panel_ < 1 || !(L_ > 0.0))
        throw std::invalid_argument(std::string(label) +
                                    ": n_z, n_z_panel, z_step_width > 0");
      build_global();
    }

    // Per sample: (re)build the nodes. Returns true if they changed.
    bool
    update(cosmosis::DataBlock& s)
    {
      if (!composite_ || !SelZBins::available(s)) {
        static bool warned = false;   // once per process
        if (composite_ && !warned) {
          std::fprintf(stderr,
                       "[%s] z_rule = composite needs the factorised "
                       "sel_function outputs (S_rich_stack, sigma_z); "
                       "falling back to the global GL rule\n",
                       label_.c_str());
          warned = true;
        }
        bool const changed = factored_;
        if (factored_)
          build_global();
        factored_ = false;
        return changed;
      }
      zbins_ = SelZBins::from_block(s);
      auto const pans = zbins_.panels(zt_lo_, zt_hi_, L_);
      if (factored_ && pans == panels_)
        return false;
      panels_ = pans;
      factored_ = true;
      auto const& [u, v] = sel_z_detail::gl_unit(n_panel_);
      xs_.clear();
      ws_.clear();
      for (auto const& [a, c] : panels_)
        for (std::size_t i = 0; i != u.size(); ++i) {
          xs_.push_back(0.5 * (a + c) + 0.5 * (c - a) * u[i]);
          ws_.push_back(2.0 * (0.5 * (c - a)) / v[i]);
        }
      return true;
    }

    SelFunctionZ_t
    sel(cosmosis::DataBlock& s, int bin) const
    {
      return SelFunctionZ_t(s, bin, factored_ ? &zbins_ : nullptr,
                            /*bicubic=*/false);
    }

    // S_ij(lnM_k, z_q) of bin b on (lnm_nodes x this rule's z nodes),
    // layout [k * size() + q]. Global/fallback and sel_interp = bilinear:
    // pointwise bilinear (the global path is the old SelFunction_t, bit
    // for bit). Composite + spline: tensor natural cubic spline of
    // S_rich (lnM pass, then z pass) x exact S_j x seam; zero outside
    // the table, like SelFunction_t.
    std::vector<double>
    tabulate(cosmosis::DataBlock& s, int b,
             std::vector<double> const& lnm_nodes) const
    {
      std::size_t const nk = lnm_nodes.size(), nq = xs_.size();
      std::vector<double> out(nk * nq, 0.0);
      if (!factored_ || !spline_) {
        auto const sel = this->sel(s, b);
        for (std::size_t k = 0; k != nk; ++k)
          for (std::size_t q = 0; q != nq; ++q)
            out[k * nq + q] = sel(lnm_nodes[k], xs_[q]);
        return out;
      }
      using doubles = std::vector<double>;
      auto const& lnm = s.view<doubles>("sel_function", "lnM");
      auto const& zt = s.view<doubles>("sel_function", "z");
      auto const& nd =
        s.view<cosmosis::ndarray<double>>("sel_function", "S_rich_stack");
      auto ext = nd.extents();
      if (ext.size() != 3 || ext[1] != zt.size() || ext[2] != lnm.size())
        throw std::runtime_error(
          "sel_function/S_rich_stack must be (n_bins, n_z, n_lnm)");
      if (b < 0 || static_cast<std::size_t>(b) >= ext[0])
        throw std::runtime_error("sel_function: bin_index out of range");
      std::size_t const nzt = zt.size(), nmt = lnm.size();
      double const* plane = &*(nd.begin() + b * nzt * nmt);

      sel_z_detail::NaturalSplineGrid const sm(lnm), sz(zt);
      auto const stm = sm.stencils(lnm_nodes);
      auto const stz = sz.stencils(xs_);
      // lnM pass: every table z-row at the lnM nodes -> T[r][k].
      std::vector<double> T(nzt * nk);
      for (std::size_t r = 0; r != nzt; ++r) {
        double const* y = plane + r * nmt;
        auto const m = sm.moments(y);
        for (std::size_t k = 0; k != nk; ++k) {
          auto const& w = stm[k];
          T[r * nk + k] = w.wa * y[w.i] + w.wb * y[w.i + 1] +
                          w.wc * m[w.i] + w.wd * m[w.i + 1];
        }
      }
      // z pass: every lnM node's column at the z nodes, x exact S_j.
      std::vector<double> sj(nq);
      for (std::size_t q = 0; q != nq; ++q)
        sj[q] = zbins_(b, xs_[q]);
      std::vector<double> col(nzt);
      double const mlo = lnm.front(), mhi = lnm.back();
      double const zlo = zt.front(), zhi = zt.back();
      for (std::size_t k = 0; k != nk; ++k) {
        if (lnm_nodes[k] < mlo || lnm_nodes[k] > mhi)
          continue;
        for (std::size_t r = 0; r != nzt; ++r)
          col[r] = T[r * nk + k];
        auto const m = sz.moments(col.data());
        for (std::size_t q = 0; q != nq; ++q) {
          if (sj[q] == 0.0 || xs_[q] < zlo || xs_[q] > zhi)
            continue;
          auto const& w = stz[q];
          out[k * nq + q] = (w.wa * col[w.i] + w.wb * col[w.i + 1] +
                             w.wc * m[w.i] + w.wd * m[w.i + 1]) *
                            sj[q];
        }
      }
      return out;
    }

    std::vector<double> const& xs() const { return xs_; }
    std::vector<double> const& ws() const { return ws_; }
    std::size_t size() const { return xs_.size(); }
    bool factored() const { return factored_; }
    std::vector<std::pair<double, double>> const& panels() const
    {
      return panels_;
    }

  private:
    std::size_t n_z_, n_panel_;
    double L_, zt_lo_, zt_hi_;
    std::string label_;
    bool composite_ = true, factored_ = false, spline_ = true;
    SelZBins zbins_;
    std::vector<std::pair<double, double>> panels_;
    std::vector<double> xs_, ws_;

    void
    build_global()
    {
      auto const& [u, v] = sel_z_detail::gl_unit(n_z_);
      double const mid = 0.5 * (zt_lo_ + zt_hi_);
      double const h = 0.5 * (zt_hi_ - zt_lo_);
      xs_.resize(n_z_);
      ws_.resize(n_z_);
      for (std::size_t i = 0; i != n_z_; ++i) {
        xs_[i] = (u[i] < 0.0) ? mid - h * (-u[i]) : mid + h * u[i];
        ws_[i] = 2.0 * h / v[i];
      }
      panels_.assign(1, {zt_lo_, zt_hi_});
    }
  };

} // namespace y3_cluster

#endif
